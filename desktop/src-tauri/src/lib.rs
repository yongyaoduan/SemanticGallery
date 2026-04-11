use std::env;
use std::fs;
use std::io::{BufRead, BufReader};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

use tauri::{AppHandle, Emitter, Manager, State};
use tauri_plugin_dialog::DialogExt;

const SETUP_PREFIX: &str = "SETUP ";
const SETUP_PROGRESS_EVENT: &str = "semanticgallery://setup-progress";
const SETUP_LOG_EVENT: &str = "semanticgallery://setup-log";
const SETUP_ERROR_EVENT: &str = "semanticgallery://setup-error";
const TEMPLATE_ENTRIES: [&str; 6] = [
    "desktop_runtime",
    "deployment",
    "scripts",
    "tools",
    "requirements.txt",
    "mlx_pipeline.py",
];

#[derive(Default)]
struct SidecarRuntime {
    starting: bool,
    base_url: Option<String>,
    last_error: Option<String>,
    child: Option<Child>,
}

struct SidecarShared {
    runtime: Mutex<SidecarRuntime>,
    ready: Condvar,
}

struct SidecarState {
    shared: Arc<SidecarShared>,
}

impl Default for SidecarState {
    fn default() -> Self {
        Self {
            shared: Arc::new(SidecarShared {
                runtime: Mutex::new(SidecarRuntime::default()),
                ready: Condvar::new(),
            }),
        }
    }
}

fn parse_sidecar_port(line: &str) -> Option<u16> {
    let (_, url) = line.split_once(' ')?;
    let (_, port_text) = url.rsplit_once(':')?;
    port_text.parse::<u16>().ok()
}

fn parse_setup_payload(line: &str) -> Option<serde_json::Value> {
    line.strip_prefix(SETUP_PREFIX)
        .and_then(|payload| serde_json::from_str(payload).ok())
}

fn copy_file(source: &Path, target: &Path) -> Result<(), String> {
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("Failed to create {}: {error}", parent.display()))?;
    }
    fs::copy(source, target)
        .map_err(|error| format!("Failed to copy {} to {}: {error}", source.display(), target.display()))?;
    let permissions = fs::metadata(source)
        .map_err(|error| format!("Failed to read {}: {error}", source.display()))?
        .permissions();
    fs::set_permissions(target, permissions)
        .map_err(|error| format!("Failed to set permissions for {}: {error}", target.display()))?;
    Ok(())
}

fn copy_tree(source: &Path, target: &Path) -> Result<(), String> {
    fs::create_dir_all(target)
        .map_err(|error| format!("Failed to create {}: {error}", target.display()))?;
    for entry in fs::read_dir(source)
        .map_err(|error| format!("Failed to read {}: {error}", source.display()))?
    {
        let entry = entry.map_err(|error| format!("Failed to read {} entry: {error}", source.display()))?;
        let source_path = entry.path();
        let target_path = target.join(entry.file_name());
        if source_path.is_dir() {
            copy_tree(&source_path, &target_path)?;
        } else if source_path.is_file() {
            copy_file(&source_path, &target_path)?;
        }
    }
    Ok(())
}

fn sync_runtime_template(template_root: &Path, runtime_root: &Path) -> Result<(), String> {
    fs::create_dir_all(runtime_root)
        .map_err(|error| format!("Failed to create {}: {error}", runtime_root.display()))?;
    for entry in TEMPLATE_ENTRIES {
        let source = template_root.join(entry);
        let target = runtime_root.join(entry);
        if source.is_dir() {
            copy_tree(&source, &target)?;
        } else if source.is_file() {
            copy_file(&source, &target)?;
        } else {
            return Err(format!(
                "The desktop runtime template is missing {}.",
                source.display()
            ));
        }
    }
    Ok(())
}

fn find_on_path(binary_name: &str) -> Option<PathBuf> {
    let path_value = env::var_os("PATH")?;
    for root in env::split_paths(&path_value) {
        let candidate = root.join(binary_name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

fn is_runtime_template_root(candidate: &Path) -> bool {
    candidate.join("desktop_runtime").is_dir()
        && candidate.join("deployment").is_dir()
        && candidate.join("scripts").is_dir()
        && candidate.join("tools").is_dir()
        && candidate.join("requirements.txt").is_file()
        && candidate.join("mlx_pipeline.py").is_file()
}

fn locate_runtime_template_root(resource_dir: &Path) -> Option<PathBuf> {
    if is_runtime_template_root(resource_dir) {
        return Some(resource_dir.to_path_buf());
    }

    let nested_candidate = resource_dir.join("_up_").join("_up_");
    if is_runtime_template_root(&nested_candidate) {
        return Some(nested_candidate);
    }

    None
}

fn resolve_runtime_template_dir(app: &AppHandle) -> Result<PathBuf, String> {
    if let Ok(override_dir) = env::var("SEMANTICGALLERY_RUNTIME_TEMPLATE_DIR") {
        let path = PathBuf::from(override_dir);
        return Ok(path.canonicalize().unwrap_or(path));
    }

    if let Ok(resource_dir) = app.path().resource_dir() {
        if let Some(template_root) = locate_runtime_template_root(&resource_dir) {
            return Ok(template_root);
        }
    }

    let dev_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    if is_runtime_template_root(&dev_root) {
        return Ok(dev_root.canonicalize().unwrap_or(dev_root));
    }

    Err("The desktop runtime template could not be resolved.".into())
}

fn resolve_uv_binary(app: &AppHandle, template_root: &Path) -> Result<PathBuf, String> {
    if let Ok(override_path) = env::var("SEMANTICGALLERY_UV_BINARY") {
        let path = PathBuf::from(override_path);
        return Ok(path.canonicalize().unwrap_or(path));
    }

    if let Ok(resource_dir) = app.path().resource_dir() {
        let candidate = resource_dir.join("uv");
        if candidate.is_file() {
            return Ok(candidate);
        }
        let nested_candidate = resource_dir.join("resources").join("uv");
        if nested_candidate.is_file() {
            return Ok(nested_candidate);
        }
    }

    let bundled_candidate = template_root.join("desktop_runtime").join("resources").join("uv");
    if bundled_candidate.is_file() {
        return Ok(bundled_candidate);
    }

    find_on_path("uv").ok_or_else(|| {
        "The desktop app could not find a bundled uv helper and no system uv binary is available."
            .into()
    })
}

fn resolve_bundled_resources_dir(app: &AppHandle, template_root: &Path) -> Option<PathBuf> {
    if let Ok(override_dir) = env::var("SEMANTICGALLERY_BUNDLED_RESOURCES_DIR") {
        let path = PathBuf::from(override_dir);
        return Some(path.canonicalize().unwrap_or(path));
    }

    if let Ok(resource_dir) = app.path().resource_dir() {
        let nested_candidate = resource_dir.join("resources");
        if nested_candidate.exists() {
            return Some(nested_candidate);
        }
        return Some(resource_dir);
    }

    let fallback = template_root.join("desktop_runtime").join("resources");
    if fallback.exists() {
        return Some(fallback);
    }

    None
}

fn reserve_sidecar_port() -> Result<u16, String> {
    let listener = TcpListener::bind("127.0.0.1:0")
        .map_err(|error| format!("Failed to reserve a sidecar port: {error}"))?;
    let port = listener
        .local_addr()
        .map_err(|error| format!("Failed to inspect the reserved sidecar port: {error}"))?
        .port();
    drop(listener);
    Ok(port)
}

fn build_bootstrap_args(
    runtime_root: &Path,
    index_db_path: &Path,
    port: u16,
    bundled_resources_dir: Option<&Path>,
) -> Vec<String> {
    let mut args = vec![
        "run".into(),
        "--python".into(),
        "3.12".into(),
        "--no-project".into(),
        "--isolated".into(),
        "-m".into(),
        "desktop_runtime.runtime_bootstrap".into(),
        "--host".into(),
        "127.0.0.1".into(),
        "--port".into(),
        port.to_string(),
        "--workspace-root".into(),
        runtime_root.as_os_str().to_string_lossy().into_owned(),
        "--index-db".into(),
        index_db_path.as_os_str().to_string_lossy().into_owned(),
    ];

    if let Some(resources_dir) = bundled_resources_dir {
        args.push("--bundled-resources-dir".into());
        args.push(resources_dir.as_os_str().to_string_lossy().into_owned());
    }

    args
}

fn emit_setup_log(app: &AppHandle, line: &str) {
    if line.trim().is_empty() {
        return;
    }
    let payload = serde_json::json!({ "line": line });
    let _ = app.emit(SETUP_LOG_EVENT, payload);
}

fn mark_start_failure(shared: &Arc<SidecarShared>, app: &AppHandle, message: String) {
    {
        let mut runtime = shared.runtime.lock().expect("sidecar runtime lock poisoned");
        runtime.starting = false;
        runtime.base_url = None;
        runtime.last_error = Some(message.clone());
    }
    shared.ready.notify_all();
    let _ = app.emit(SETUP_ERROR_EVENT, serde_json::json!({ "message": message }));
}

fn spawn_output_threads(
    app: AppHandle,
    shared: Arc<SidecarShared>,
    child: &mut Child,
) -> Result<(), String> {
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "The sidecar bootstrap did not expose stdout.".to_string())?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| "The sidecar bootstrap did not expose stderr.".to_string())?;

    let stdout_app = app.clone();
    let stdout_shared = shared.clone();
    thread::spawn(move || {
        let reader = BufReader::new(stdout);
        for line_result in reader.lines() {
            let line = match line_result {
                Ok(line) => line,
                Err(error) => {
                    mark_start_failure(
                        &stdout_shared,
                        &stdout_app,
                        format!("The sidecar bootstrap stopped streaming stdout: {error}"),
                    );
                    return;
                }
            };

            if let Some(payload) = parse_setup_payload(&line) {
                let _ = stdout_app.emit(SETUP_PROGRESS_EVENT, payload);
                continue;
            }

            if line.starts_with("READY ") {
                if parse_sidecar_port(&line).is_none() {
                    mark_start_failure(
                        &stdout_shared,
                        &stdout_app,
                        "The sidecar reported an invalid ready URL.".into(),
                    );
                    return;
                }
                {
                    let mut runtime = stdout_shared.runtime.lock().expect("sidecar runtime lock poisoned");
                    runtime.base_url = Some(line["READY ".len()..].to_string());
                    runtime.last_error = None;
                    runtime.starting = false;
                }
                stdout_shared.ready.notify_all();
                continue;
            }

            emit_setup_log(&stdout_app, &line);
        }

        let should_fail = {
            let runtime = stdout_shared.runtime.lock().expect("sidecar runtime lock poisoned");
            runtime.starting && runtime.base_url.is_none()
        };
        if should_fail {
            mark_start_failure(
                &stdout_shared,
                &stdout_app,
                "The sidecar exited before it reported a ready URL.".into(),
            );
        }
    });

    let stderr_app = app;
    thread::spawn(move || {
        let reader = BufReader::new(stderr);
        for line_result in reader.lines() {
            match line_result {
                Ok(line) => emit_setup_log(&stderr_app, &line),
                Err(error) => {
                    let _ = stderr_app.emit(
                        SETUP_LOG_EVENT,
                        serde_json::json!({ "line": format!("Failed to read sidecar stderr: {error}") }),
                    );
                    return;
                }
            }
        }
    });

    Ok(())
}

fn start_sidecar(app: &AppHandle, shared: Arc<SidecarShared>) -> Result<(), String> {
    let template_root = resolve_runtime_template_dir(app)?;
    let support_root = app
        .path()
        .app_data_dir()
        .map_err(|error| format!("Failed to resolve the app support directory: {error}"))?;
    let runtime_root = support_root.join("runtime");
    let index_db_path = support_root.join("index.sqlite3");
    let bundled_resources_dir = resolve_bundled_resources_dir(app, &template_root);
    let uv_binary = resolve_uv_binary(app, &template_root)?;
    let port = reserve_sidecar_port()?;

    sync_runtime_template(&template_root, &runtime_root)?;

    let args = build_bootstrap_args(
        &runtime_root,
        &index_db_path,
        port,
        bundled_resources_dir.as_deref(),
    );

    let mut child = Command::new(&uv_binary)
        .args(&args)
        .current_dir(&runtime_root)
        .env("PYTHONUNBUFFERED", "1")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("Failed to start the sidecar bootstrap: {error}"))?;

    spawn_output_threads(app.clone(), shared.clone(), &mut child)?;

    let mut runtime = shared.runtime.lock().expect("sidecar runtime lock poisoned");
    runtime.child = Some(child);
    Ok(())
}

fn ensure_sidecar(app: &AppHandle, state: &SidecarState) -> Result<String, String> {
    let shared = state.shared.clone();
    {
        let mut runtime = shared.runtime.lock().expect("sidecar runtime lock poisoned");
        let mut child_exited = false;
        if let Some(child) = runtime.child.as_mut() {
            if let Ok(Some(_)) = child.try_wait() {
                child_exited = true;
            }
        }
        if child_exited {
            runtime.child = None;
            runtime.base_url = None;
            runtime.starting = false;
        }

        if let Some(base_url) = runtime.base_url.clone() {
            return Ok(base_url);
        }

        if runtime.starting {
            while runtime.starting && runtime.base_url.is_none() && runtime.last_error.is_none() {
                runtime = shared
                    .ready
                    .wait(runtime)
                    .expect("sidecar runtime lock poisoned");
            }
            if let Some(base_url) = runtime.base_url.clone() {
                return Ok(base_url);
            }
            return Err(
                runtime
                    .last_error
                    .clone()
                    .unwrap_or_else(|| "The sidecar did not finish starting.".into()),
            );
        }

        runtime.starting = true;
        runtime.last_error = None;
    }

    if let Err(error) = start_sidecar(app, shared.clone()) {
        mark_start_failure(&shared, app, error.clone());
        return Err(error);
    }

    let mut runtime = shared.runtime.lock().expect("sidecar runtime lock poisoned");
    while runtime.starting && runtime.base_url.is_none() && runtime.last_error.is_none() {
        runtime = shared
            .ready
            .wait(runtime)
            .expect("sidecar runtime lock poisoned");
    }
    if let Some(base_url) = runtime.base_url.clone() {
        return Ok(base_url);
    }
    Err(
        runtime
            .last_error
            .clone()
            .unwrap_or_else(|| "The sidecar did not finish starting.".into()),
    )
}

fn stop_sidecar(state: &SidecarState) {
    let child = {
        let mut runtime = state.shared.runtime.lock().expect("sidecar runtime lock poisoned");
        runtime.base_url = None;
        runtime.starting = false;
        runtime.last_error = None;
        runtime.child.take()
    };
    if let Some(mut child) = child {
        let _ = child.kill();
        let _ = child.wait();
    }
}

#[tauri::command]
fn pick_folder(app: AppHandle) -> Result<Option<String>, String> {
    let folder = app.dialog().file().blocking_pick_folder();
    Ok(folder.map(|path| path.to_string()))
}

#[tauri::command]
fn sidecar_base_url(app: AppHandle, state: State<'_, SidecarState>) -> Result<String, String> {
    ensure_sidecar(&app, state.inner())
}

pub fn run() {
    let app = tauri::Builder::default()
        .manage(SidecarState::default())
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![pick_folder, sidecar_base_url])
        .build(tauri::generate_context!())
        .expect("failed to build SemanticGallery desktop shell");

    app.run(|app_handle, event| {
        if matches!(event, tauri::RunEvent::Exit | tauri::RunEvent::ExitRequested { .. }) {
            stop_sidecar(&app_handle.state::<SidecarState>());
        }
    });
}

#[cfg(test)]
mod tests {
    use super::{
        build_bootstrap_args, copy_file, is_runtime_template_root, locate_runtime_template_root,
        parse_setup_payload, parse_sidecar_port, sync_runtime_template,
    };
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_root(name: &str) -> PathBuf {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock went backwards")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("semanticgallery-{name}-{suffix}"));
        fs::create_dir_all(&root).expect("failed to create temp root");
        root
    }

    #[test]
    fn parse_sidecar_port_reads_ready_line() {
        let value = parse_sidecar_port("READY http://127.0.0.1:38291");
        assert_eq!(value, Some(38291));
    }

    #[test]
    fn parse_setup_payload_reads_json_progress_event() {
        let payload =
            parse_setup_payload("SETUP {\"task\":\"prepare-base-model\",\"phase\":\"finish\",\"current\":3,\"total\":5}")
                .expect("payload should parse");
        assert_eq!(payload["task"], "prepare-base-model");
        assert_eq!(payload["phase"], "finish");
        assert_eq!(payload["current"], 3);
    }

    #[test]
    fn build_bootstrap_args_passes_runtime_and_index_paths() {
        let args = build_bootstrap_args(
            PathBuf::from("/tmp/runtime").as_path(),
            PathBuf::from("/tmp/index.sqlite3").as_path(),
            38291,
            Some(PathBuf::from("/tmp/resources").as_path()),
        );

        assert_eq!(
            args,
            vec![
                "run",
                "--python",
                "3.12",
                "--no-project",
                "--isolated",
                "-m",
                "desktop_runtime.runtime_bootstrap",
                "--host",
                "127.0.0.1",
                "--port",
                "38291",
                "--workspace-root",
                "/tmp/runtime",
                "--index-db",
                "/tmp/index.sqlite3",
                "--bundled-resources-dir",
                "/tmp/resources",
            ]
        );
    }

    #[test]
    fn copy_file_preserves_script_permissions() {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            let root = temp_root("copy-file");
            let source = root.join("source.sh");
            let target = root.join("target.sh");
            fs::write(&source, "#!/bin/bash\n").expect("failed to write source");
            let mut permissions = fs::metadata(&source).expect("failed to read metadata").permissions();
            permissions.set_mode(0o755);
            fs::set_permissions(&source, permissions).expect("failed to set permissions");

            copy_file(&source, &target).expect("copy should succeed");

            let copied = fs::metadata(&target).expect("failed to read copied metadata");
            assert_eq!(copied.permissions().mode() & 0o777, 0o755);

            let _ = fs::remove_dir_all(root);
        }
    }

    #[test]
    fn sync_runtime_template_copies_required_entries() {
        let template_root = temp_root("template");
        let runtime_root = temp_root("runtime");

        fs::create_dir_all(template_root.join("desktop_runtime")).expect("failed to create desktop_runtime");
        fs::create_dir_all(template_root.join("deployment")).expect("failed to create deployment");
        fs::create_dir_all(template_root.join("scripts")).expect("failed to create scripts");
        fs::create_dir_all(template_root.join("tools")).expect("failed to create tools");
        fs::write(template_root.join("desktop_runtime").join("__init__.py"), "").expect("failed to write file");
        fs::write(template_root.join("deployment").join("__init__.py"), "").expect("failed to write file");
        fs::write(template_root.join("scripts").join("prepare_data.sh"), "#!/bin/bash\n").expect("failed to write file");
        fs::write(template_root.join("tools").join("train.py"), "").expect("failed to write file");
        fs::write(template_root.join("requirements.txt"), "fastapi\n").expect("failed to write file");
        fs::write(template_root.join("mlx_pipeline.py"), "VALUE = 1\n").expect("failed to write file");

        sync_runtime_template(&template_root, &runtime_root).expect("sync should succeed");

        assert!(runtime_root.join("desktop_runtime").join("__init__.py").is_file());
        assert!(runtime_root.join("deployment").join("__init__.py").is_file());
        assert!(runtime_root.join("scripts").join("prepare_data.sh").is_file());
        assert!(runtime_root.join("tools").join("train.py").is_file());
        assert!(runtime_root.join("requirements.txt").is_file());
        assert!(runtime_root.join("mlx_pipeline.py").is_file());

        let _ = fs::remove_dir_all(template_root);
        let _ = fs::remove_dir_all(runtime_root);
    }

    #[test]
    fn locate_runtime_template_root_accepts_nested_up_segments() {
        let resource_root = temp_root("resources");
        let nested_root = resource_root.join("_up_").join("_up_");
        fs::create_dir_all(nested_root.join("desktop_runtime")).expect("failed to create desktop_runtime");
        fs::create_dir_all(nested_root.join("deployment")).expect("failed to create deployment");
        fs::create_dir_all(nested_root.join("scripts")).expect("failed to create scripts");
        fs::create_dir_all(nested_root.join("tools")).expect("failed to create tools");
        fs::write(nested_root.join("requirements.txt"), "").expect("failed to write requirements");
        fs::write(nested_root.join("mlx_pipeline.py"), "").expect("failed to write mlx_pipeline");

        let located = locate_runtime_template_root(&resource_root).expect("template root should be found");

        assert!(is_runtime_template_root(&located));
        assert_eq!(located, nested_root);

        let _ = fs::remove_dir_all(resource_root);
    }
}
