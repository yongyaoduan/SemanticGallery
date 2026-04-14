mod desktop_core;

use std::env;
use std::ffi::OsStr;
use std::fs;
use std::io::{BufRead, BufReader};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::sync_channel;
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Manager, State};
use tauri_plugin_dialog::DialogExt;

const SETUP_PREFIX: &str = "SETUP ";
const SETUP_PROGRESS_EVENT: &str = "semanticgallery://setup-progress";
const SETUP_LOG_EVENT: &str = "semanticgallery://setup-log";
const SETUP_ERROR_EVENT: &str = "semanticgallery://setup-error";
const RUNTIME_READY_EVENT: &str = "semanticgallery://runtime-ready";
const SETUP_LOG_LIMIT: usize = 24;
const START_FAILURE_LOG_CONTEXT_LIMIT: usize = 6;
const UNINSTALLER_APP_NAME: &str = "Uninstall SemanticGallery.app";
const AUTOMATION_TOOLS_ENV: &str = "SEMANTICGALLERY_AUTOMATION_TOOLS";
const AUTOMATION_TOOLS_TEST_MODE_ARG: &str = "--semanticgallery-test-mode";
const AUTOMATION_REQUEST_FILE: &str = "automation-request.json";
const AUTOMATION_RESPONSE_FILE: &str = "automation-response.json";
const AUTOMATION_POLL_INTERVAL_MS: u64 = 150;
const TEMPLATE_ENTRIES: [&str; 6] = [
    "desktop_runtime",
    "deployment",
    "scripts",
    "tools",
    "requirements.txt",
    "mlx_pipeline.py",
];
const SETUP_STEPS: [(&str, &str); 6] = [
    ("sync-runtime", "App files"),
    ("check-runtime", "Python runtime"),
    ("prepare-dependencies", "Dependencies"),
    ("prepare-base-model", "Base model"),
    ("prepare-public-anchor", "Public anchor"),
    ("finish-setup", "Finalize"),
];

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct SetupStepSnapshot {
    task: String,
    label: String,
    status: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
enum AutomationRequestKind {
    SelectFolder,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct AutomationRequest {
    id: String,
    kind: AutomationRequestKind,
    folder_path: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct AutomationResponse {
    id: String,
    status: String,
    message: String,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct BootstrapSnapshot {
    status: String,
    current_step: u32,
    total_steps: u32,
    message: String,
    onboarding_required: bool,
    steps: Vec<SetupStepSnapshot>,
    logs: Vec<String>,
}

impl Default for BootstrapSnapshot {
    fn default() -> Self {
        Self {
            status: "idle".into(),
            current_step: 0,
            total_steps: SETUP_STEPS.len() as u32,
            message: "Install the local runtime to continue.".into(),
            onboarding_required: true,
            steps: setup_steps(),
            logs: Vec::new(),
        }
    }
}

#[derive(Default)]
struct SidecarRuntime {
    starting: bool,
    cancel_requested: bool,
    base_url: Option<String>,
    last_error: Option<String>,
    child: Option<Child>,
    bootstrap: BootstrapSnapshot,
}

struct SidecarShared {
    runtime: Mutex<SidecarRuntime>,
    ready: Condvar,
}

struct SidecarState {
    shared: Arc<SidecarShared>,
}

enum StartError {
    Failed(String),
    Cancelled,
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

fn setup_steps() -> Vec<SetupStepSnapshot> {
    SETUP_STEPS
        .iter()
        .map(|(task, label)| SetupStepSnapshot {
            task: (*task).into(),
            label: (*label).into(),
            status: "idle".into(),
        })
        .collect()
}

fn onboarding_marker_path(app_data_dir: &Path) -> PathBuf {
    app_data_dir.join("onboarding-complete")
}

fn automation_tools_marker_path(app_data_dir: &Path) -> PathBuf {
    app_data_dir.join("automation-tools-enabled")
}

fn automation_request_path(app_data_dir: &Path) -> PathBuf {
    app_data_dir.join(AUTOMATION_REQUEST_FILE)
}

fn automation_response_path(app_data_dir: &Path) -> PathBuf {
    app_data_dir.join(AUTOMATION_RESPONSE_FILE)
}

fn automation_tools_enabled_for_dir(app_data_dir: &Path) -> bool {
    automation_tools_marker_path(app_data_dir).is_file()
        || automation_tools_env_value_enabled(env::var_os(AUTOMATION_TOOLS_ENV).as_deref())
        || automation_tools_enabled_from_args(env::args_os())
}

fn automation_tools_env_value_enabled(value: Option<&OsStr>) -> bool {
    value.is_some_and(|raw| {
        let normalized = raw.to_string_lossy().trim().to_ascii_lowercase();
        !normalized.is_empty() && normalized != "0" && normalized != "false" && normalized != "no"
    })
}

fn automation_tools_enabled_from_args<I, S>(args: I) -> bool
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    args.into_iter()
        .any(|arg| arg.as_ref().to_string_lossy() == AUTOMATION_TOOLS_TEST_MODE_ARG)
}

fn runtime_ready_marker_path(app_data_dir: &Path) -> PathBuf {
    app_data_dir.join("runtime-ready")
}

fn take_automation_request(path: &Path) -> Result<Option<AutomationRequest>, String> {
    if !path.is_file() {
        return Ok(None);
    }
    let raw = fs::read_to_string(path)
        .map_err(|error| format!("Failed to read {}: {error}", path.display()))?;
    fs::remove_file(path).map_err(|error| format!("Failed to clear {}: {error}", path.display()))?;
    if raw.trim().is_empty() {
        return Ok(None);
    }
    serde_json::from_str(&raw)
        .map(Some)
        .map_err(|error| format!("Failed to parse {}: {error}", path.display()))
}

fn write_automation_response(path: &Path, response: &AutomationResponse) -> Result<(), String> {
    let encoded =
        serde_json::to_vec(response).map_err(|error| format!("Failed to encode the automation response: {error}"))?;
    fs::write(path, encoded).map_err(|error| format!("Failed to write {}: {error}", path.display()))
}

fn runtime_files_look_ready(app_data_dir: &Path) -> bool {
    let runtime_root = app_data_dir.join("runtime");
    runtime_root.join(".venv").join("bin").join("python").is_file()
        && runtime_root
            .join(".cache")
            .join("mlx")
            .join("siglip2-base-patch16-224-f32")
            .join("config.json")
            .is_file()
        && runtime_root
            .join(".cache")
            .join("semanticgallery")
            .join("stage2_public_anchor")
            .join("extracted")
            .join("flickr30k")
            .join("captions.txt")
            .is_file()
        && runtime_root
            .join(".cache")
            .join("semanticgallery")
            .join("stage2_public_anchor")
            .join("extracted")
            .join("screen2words")
            .join("manifest.jsonl")
            .is_file()
}

fn runtime_dependencies_look_ready(app_data_dir: &Path) -> bool {
    let runtime_root = app_data_dir.join("runtime");
    let python_bin = runtime_root.join(".venv").join("bin").join("python");
    if !python_bin.is_file() {
        return false;
    }

    Command::new(python_bin)
        .current_dir(runtime_root)
        .arg("-c")
        .arg(
            "import datasets, fastapi, huggingface_hub, jinja2, mlx, mlx_embeddings, multipart, pillow_heif, tqdm, uvicorn",
        )
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

fn runtime_ready_for_dir(app_data_dir: &Path) -> bool {
    runtime_ready_marker_path(app_data_dir).is_file()
        && runtime_files_look_ready(app_data_dir)
        && runtime_dependencies_look_ready(app_data_dir)
}

fn onboarding_required_for_dir(app_data_dir: &Path) -> bool {
    !onboarding_marker_path(app_data_dir).is_file() || !runtime_ready_for_dir(app_data_dir)
}

fn onboarding_required(app: &AppHandle) -> bool {
    app.path()
        .app_data_dir()
        .ok()
        .map_or(true, |app_data_dir| onboarding_required_for_dir(&app_data_dir))
}

fn snapshot_for_app(app: &AppHandle, snapshot: BootstrapSnapshot) -> BootstrapSnapshot {
    BootstrapSnapshot {
        onboarding_required: onboarding_required(app),
        ..snapshot
    }
}

fn complete_onboarding_snapshot(snapshot: BootstrapSnapshot) -> BootstrapSnapshot {
    BootstrapSnapshot {
        onboarding_required: false,
        ..snapshot
    }
}

fn reset_bootstrap(runtime: &mut SidecarRuntime) {
    runtime.bootstrap = BootstrapSnapshot {
        status: "running".into(),
        current_step: 0,
        total_steps: SETUP_STEPS.len() as u32,
        message: "Preparing the local desktop runtime.".into(),
        onboarding_required: true,
        steps: setup_steps(),
        logs: vec!["Starting SemanticGallery desktop runtime.".into()],
    };
}

fn is_uvicorn_access_log(line: &str) -> bool {
    line.starts_with("INFO: 127.0.0.1:")
        && (line.contains("\"OPTIONS ")
            || line.contains("\"GET ")
            || line.contains("\"POST ")
            || line.contains("\"HEAD "))
}

fn normalize_setup_log_line(line: &str) -> Option<String> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return None;
    }

    if trimmed.starts_with("INFO: Uvicorn running on ")
        || trimmed.starts_with("INFO:     Uvicorn running on ")
        || trimmed.starts_with("INFO: Started server process")
        || trimmed.starts_with("INFO:     Started server process")
        || trimmed.starts_with("INFO: Waiting for application startup.")
        || trimmed.starts_with("INFO:     Waiting for application startup.")
        || trimmed.starts_with("INFO: Application startup complete.")
        || trimmed.starts_with("INFO:     Application startup complete.")
        || trimmed.contains("DeprecationWarning:")
        || trimmed.starts_with("Read more about it in the")
        || trimmed.contains("FastAPI docs for Lifespan Events")
        || trimmed == "@app.on_event(\"startup\")"
        || is_uvicorn_access_log(trimmed)
    {
        return None;
    }

    Some(trimmed.to_string())
}

fn append_runtime_log(runtime: &mut SidecarRuntime, line: &str) {
    if line.trim().is_empty() {
        return;
    }
    if runtime.bootstrap.logs.last().is_some_and(|last| last == line) {
        return;
    }
    runtime.bootstrap.logs.push(line.to_string());
    if runtime.bootstrap.logs.len() > SETUP_LOG_LIMIT {
        let drop_count = runtime.bootstrap.logs.len() - SETUP_LOG_LIMIT;
        runtime.bootstrap.logs.drain(0..drop_count);
    }
}

fn build_start_failure_message(runtime: &SidecarRuntime, fallback: &str) -> String {
    let recent_logs = runtime
        .bootstrap
        .logs
        .iter()
        .rev()
        .filter(|line| line.as_str() != fallback && line.as_str() != "Starting SemanticGallery desktop runtime.")
        .take(START_FAILURE_LOG_CONTEXT_LIMIT)
        .cloned()
        .collect::<Vec<_>>();

    if recent_logs.is_empty() {
        return fallback.into();
    }

    let context = recent_logs
        .into_iter()
        .rev()
        .collect::<Vec<_>>()
        .join("\n");
    format!("{fallback}\nRecent runtime log:\n{context}")
}

fn apply_setup_progress(runtime: &mut SidecarRuntime, payload: &serde_json::Value) {
    runtime.bootstrap.status = "running".into();

    if let Some(current) = payload.get("current").and_then(|value| value.as_u64()) {
        runtime.bootstrap.current_step = current as u32;
    }
    if let Some(total) = payload.get("total").and_then(|value| value.as_u64()) {
        runtime.bootstrap.total_steps = total as u32;
    }
    if let Some(message) = payload.get("message").and_then(|value| value.as_str()) {
        runtime.bootstrap.message = message.to_string();
    }
    if let Some(task) = payload.get("task").and_then(|value| value.as_str()) {
        let phase = payload
            .get("phase")
            .and_then(|value| value.as_str())
            .unwrap_or("start");
        for step in &mut runtime.bootstrap.steps {
            if step.task == task {
                step.status = if phase == "finish" { "done" } else { "running" }.into();
            }
        }
    }
}

fn bootstrap_snapshot(state: &SidecarState) -> BootstrapSnapshot {
    state
        .shared
        .runtime
        .lock()
        .expect("sidecar runtime lock poisoned")
        .bootstrap
        .clone()
}

fn run_automation_request(app: &AppHandle, request: AutomationRequest) -> Result<AutomationResponse, String> {
    match request.kind {
        AutomationRequestKind::SelectFolder => {
            let folder_path = request
                .folder_path
                .as_deref()
                .ok_or_else(|| "The automation request did not include a folder path.".to_string())?;
            let base_url = ensure_sidecar(&app, app.state::<SidecarState>().inner())?;
            let payload = desktop_core::automation_select_folder(
                app.state::<desktop_core::DesktopCoreState>().inner.clone(),
                app,
                &base_url,
                folder_path,
            )?;
            Ok(AutomationResponse {
                id: request.id,
                status: "ok".into(),
                message: format!(
                    "Loaded {} (refreshed={}, skipped={}).",
                    payload.runtime.active_folder.unwrap_or_else(|| folder_path.to_string()),
                    payload.refreshed,
                    payload.skipped
                ),
            })
        }
    }
}

fn spawn_automation_request_watcher(app: &AppHandle) -> Result<(), String> {
    let app_data_dir = app
        .path()
        .app_data_dir()
        .map_err(|error| format!("Failed to resolve the app support directory: {error}"))?;
    if !automation_tools_enabled_for_dir(&app_data_dir) {
        return Ok(());
    }

    fs::create_dir_all(&app_data_dir)
        .map_err(|error| format!("Failed to create {}: {error}", app_data_dir.display()))?;
    let request_path = automation_request_path(&app_data_dir);
    let response_path = automation_response_path(&app_data_dir);
    let _ = fs::remove_file(&request_path);
    let _ = fs::remove_file(&response_path);
    let app_handle = app.clone();

    thread::spawn(move || loop {
        match take_automation_request(&request_path) {
            Ok(Some(request)) => {
                let response = match run_automation_request(&app_handle, request.clone()) {
                    Ok(response) => response,
                    Err(message) => AutomationResponse {
                        id: request.id,
                        status: "error".into(),
                        message,
                    },
                };
                let _ = write_automation_response(&response_path, &response);
            }
            Ok(None) => {}
            Err(message) => {
                let _ = write_automation_response(
                    &response_path,
                    &AutomationResponse {
                        id: "automation-request".into(),
                        status: "error".into(),
                        message,
                    },
                );
            }
        }
        thread::sleep(Duration::from_millis(AUTOMATION_POLL_INTERVAL_MS));
    });
    Ok(())
}

fn cancellation_requested(shared: &Arc<SidecarShared>) -> bool {
    shared
        .runtime
        .lock()
        .expect("sidecar runtime lock poisoned")
        .cancel_requested
}

fn check_cancellation(shared: &Arc<SidecarShared>) -> Result<(), StartError> {
    if cancellation_requested(shared) {
        return Err(StartError::Cancelled);
    }
    Ok(())
}

fn emit_setup_log(shared: &Arc<SidecarShared>, app: &AppHandle, line: &str) {
    let Some(line) = normalize_setup_log_line(line) else {
        return;
    };
    {
        let mut runtime = shared.runtime.lock().expect("sidecar runtime lock poisoned");
        append_runtime_log(&mut runtime, &line);
    }
    let payload = serde_json::json!({ "line": line });
    let _ = app.emit(SETUP_LOG_EVENT, payload);
}

fn emit_setup_progress(shared: &Arc<SidecarShared>, app: &AppHandle, payload: serde_json::Value) {
    let message = payload
        .get("message")
        .and_then(|value| value.as_str())
        .map(str::to_string);

    {
        let mut runtime = shared.runtime.lock().expect("sidecar runtime lock poisoned");
        apply_setup_progress(&mut runtime, &payload);
        if let Some(message) = message.as_deref() {
            append_runtime_log(&mut runtime, message);
        }
    }

    let _ = app.emit(SETUP_PROGRESS_EVENT, payload);
    if let Some(message) = message {
        let _ = app.emit(SETUP_LOG_EVENT, serde_json::json!({ "line": message }));
    }
}

fn mark_runtime_ready(shared: &Arc<SidecarShared>, app: &AppHandle, base_url: String) {
    if let Ok(app_data_dir) = app.path().app_data_dir() {
        let _ = fs::create_dir_all(&app_data_dir);
        let _ = fs::write(runtime_ready_marker_path(&app_data_dir), b"ready");
    }
    {
        let mut runtime = shared.runtime.lock().expect("sidecar runtime lock poisoned");
        runtime.base_url = Some(base_url.clone());
        runtime.last_error = None;
        runtime.starting = false;
        runtime.bootstrap.status = "ready".into();
        runtime.bootstrap.current_step = runtime.bootstrap.total_steps;
        runtime.bootstrap.message = "Desktop runtime is ready.".into();
        for step in &mut runtime.bootstrap.steps {
            step.status = "done".into();
        }
        append_runtime_log(&mut runtime, "Desktop runtime is ready.");
    }

    shared.ready.notify_all();
    let _ = app.emit(RUNTIME_READY_EVENT, serde_json::json!({ "baseUrl": base_url }));
}

fn mark_start_failure(shared: &Arc<SidecarShared>, app: &AppHandle, message: String) {
    {
        let mut runtime = shared.runtime.lock().expect("sidecar runtime lock poisoned");
        runtime.starting = false;
        runtime.cancel_requested = false;
        runtime.base_url = None;
        runtime.last_error = Some(message.clone());
        runtime.bootstrap.status = "failed".into();
        runtime.bootstrap.message = message.clone();
        append_runtime_log(&mut runtime, &message);
    }
    shared.ready.notify_all();
    let _ = app.emit(SETUP_ERROR_EVENT, serde_json::json!({ "message": message }));
}

fn mark_start_cancelled(shared: &Arc<SidecarShared>) -> BootstrapSnapshot {
    let snapshot = {
        let mut runtime = shared.runtime.lock().expect("sidecar runtime lock poisoned");
        runtime.starting = false;
        runtime.cancel_requested = false;
        runtime.base_url = None;
        runtime.last_error = None;
        runtime.child = None;
        runtime.bootstrap.status = "idle".into();
        runtime.bootstrap.current_step = 0;
        runtime.bootstrap.message =
            "Installation cancelled. Install and Start whenever you want to continue.".into();
        runtime.bootstrap.steps = setup_steps();
        append_runtime_log(&mut runtime, "Installation cancelled.");
        runtime.bootstrap.clone()
    };
    shared.ready.notify_all();
    snapshot
}

fn parse_sidecar_port(line: &str) -> Option<u16> {
    let (_, url) = line.split_once(' ')?;
    let (_, port_text) = url.rsplit_once(':')?;
    port_text.parse::<u16>().ok()
}

fn command_matches_sidecar(command_line: &str, runtime_root: &Path) -> bool {
    command_line.contains("desktop_runtime.sidecar_main")
        && command_line.contains(runtime_root.as_os_str().to_string_lossy().as_ref())
}

fn sidecar_pids_for_runtime(runtime_root: &Path) -> Result<Vec<u32>, String> {
    let output = Command::new("ps")
        .args(["axww", "-o", "pid=,command="])
        .output()
        .map_err(|error| format!("Failed to inspect the process table: {error}"))?;
    if !output.status.success() {
        return Err("Failed to inspect the process table.".into());
    }

    let mut pids = Vec::new();
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let mut parts = trimmed.splitn(2, char::is_whitespace);
        let Some(pid_text) = parts.next() else {
            continue;
        };
        let Some(command_line) = parts.next() else {
            continue;
        };
        let Ok(pid) = pid_text.trim().parse::<u32>() else {
            continue;
        };
        if command_matches_sidecar(command_line.trim(), runtime_root) {
            pids.push(pid);
        }
    }
    Ok(pids)
}

fn terminate_sidecar_pid(pid: u32) -> Result<(), String> {
    let pid_string = pid.to_string();
    let term_status = Command::new("kill")
        .args(["-TERM", pid_string.as_str()])
        .status()
        .map_err(|error| format!("Failed to stop the stale sidecar process {pid}: {error}"))?;
    if !term_status.success() {
        return Ok(());
    }

    for _ in 0..20 {
        thread::sleep(Duration::from_millis(100));
        let remaining = Command::new("ps")
            .args(["-p", pid_string.as_str(), "-o", "pid="])
            .output()
            .map_err(|error| format!("Failed to confirm the stale sidecar process status: {error}"))?;
        if String::from_utf8_lossy(&remaining.stdout).trim().is_empty() {
            return Ok(());
        }
    }

    let _ = Command::new("kill").args(["-KILL", pid_string.as_str()]).status();
    Ok(())
}

fn cleanup_stale_sidecars(app: &AppHandle) -> Result<(), String> {
    let app_data_dir = app
        .path()
        .app_data_dir()
        .map_err(|error| format!("Failed to resolve the app support directory: {error}"))?;
    let runtime_root = app_data_dir.join("runtime");
    for pid in sidecar_pids_for_runtime(&runtime_root)? {
        terminate_sidecar_pid(pid)?;
    }
    Ok(())
}

fn build_sidecar_env(support_root: &Path) -> Vec<(String, String)> {
    let cache_root = support_root.join("cache");
    let huggingface_root = cache_root.join("huggingface");
    let python_install_dir = support_root.join("python");

    vec![
        ("UV_CACHE_DIR".into(), cache_root.join("uv").to_string_lossy().into_owned()),
        (
            "UV_PYTHON_INSTALL_DIR".into(),
            python_install_dir.to_string_lossy().into_owned(),
        ),
        ("XDG_CACHE_HOME".into(), cache_root.to_string_lossy().into_owned()),
        ("HF_HOME".into(), huggingface_root.to_string_lossy().into_owned()),
        (
            "HUGGINGFACE_HUB_CACHE".into(),
            huggingface_root.join("hub").to_string_lossy().into_owned(),
        ),
        (
            "TRANSFORMERS_CACHE".into(),
            cache_root.join("transformers").to_string_lossy().into_owned(),
        ),
    ]
}

fn resolve_uninstaller_app(app: &AppHandle) -> Result<PathBuf, String> {
    let mut candidates = Vec::new();

    if let Ok(executable_dir) = app.path().executable_dir() {
        if let Some(bundle_dir) = executable_dir
            .parent()
            .and_then(|path| path.parent())
            .filter(|path| path.extension().and_then(|extension| extension.to_str()) == Some("app"))
        {
            if let Some(parent_dir) = bundle_dir.parent() {
                candidates.push(parent_dir.join(UNINSTALLER_APP_NAME));
            }
        }
    }

    if let Ok(resource_dir) = app.path().resource_dir() {
        candidates.push(resource_dir.join("uninstall").join(UNINSTALLER_APP_NAME));
        candidates.push(
            resource_dir
                .join("resources")
                .join("uninstall")
                .join(UNINSTALLER_APP_NAME),
        );
    }

    candidates
        .into_iter()
        .find(|path| path.is_dir())
        .ok_or_else(|| "The uninstaller app could not be found in this desktop build.".into())
}

fn resolve_app_bundle_dir(app: &AppHandle) -> Option<PathBuf> {
    let executable_dir = app.path().executable_dir().ok()?;
    executable_dir
        .parent()
        .and_then(|path| path.parent())
        .filter(|path| path.extension().and_then(|extension| extension.to_str()) == Some("app"))
        .map(Path::to_path_buf)
}

fn mark_setup_cancelling(state: &SidecarState) -> (BootstrapSnapshot, Option<Child>) {
    let mut runtime = state.shared.runtime.lock().expect("sidecar runtime lock poisoned");
    if !runtime.starting {
        return (runtime.bootstrap.clone(), None);
    }

    runtime.cancel_requested = true;
    runtime.bootstrap.status = "cancelling".into();
    runtime.bootstrap.message = "Stopping the local desktop runtime installation.".into();
    append_runtime_log(
        &mut runtime,
        "Stopping the local desktop runtime installation.",
    );
    (runtime.bootstrap.clone(), runtime.child.take())
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
    copy_tree_with_cancel(source, target, &|| false).map_err(|error| match error {
        StartError::Failed(message) => message,
        StartError::Cancelled => "The copy operation was cancelled.".into(),
    })
}

fn stage_uninstaller_for_self_removal(helper_path: &Path, staging_root: &Path) -> Result<PathBuf, String> {
    if !helper_path.is_dir() {
        return Err(format!(
            "The bundled uninstaller app is missing at {}.",
            helper_path.display()
        ));
    }

    fs::create_dir_all(staging_root)
        .map_err(|error| format!("Failed to create {}: {error}", staging_root.display()))?;
    let staged_helper = staging_root.join(UNINSTALLER_APP_NAME);
    if staged_helper.exists() {
        fs::remove_dir_all(&staged_helper)
            .map_err(|error| format!("Failed to remove {}: {error}", staged_helper.display()))?;
    }
    copy_tree(helper_path, &staged_helper)?;
    Ok(staged_helper)
}

#[cfg(test)]
fn sync_runtime_template(template_root: &Path, runtime_root: &Path) -> Result<(), String> {
    sync_runtime_template_with_cancel(template_root, runtime_root, &|| false).map_err(|error| match error {
        StartError::Failed(message) => message,
        StartError::Cancelled => "The desktop runtime installation was cancelled.".into(),
    })
}

fn copy_tree_with_cancel<F>(source: &Path, target: &Path, should_cancel: &F) -> Result<(), StartError>
where
    F: Fn() -> bool,
{
    if should_cancel() {
        return Err(StartError::Cancelled);
    }

    fs::create_dir_all(target)
        .map_err(|error| StartError::Failed(format!("Failed to create {}: {error}", target.display())))?;
    for entry in fs::read_dir(source)
        .map_err(|error| StartError::Failed(format!("Failed to read {}: {error}", source.display())))?
    {
        if should_cancel() {
            return Err(StartError::Cancelled);
        }

        let entry = entry
            .map_err(|error| StartError::Failed(format!("Failed to read {} entry: {error}", source.display())))?;
        let source_path = entry.path();
        let target_path = target.join(entry.file_name());
        if source_path.is_dir() {
            copy_tree_with_cancel(&source_path, &target_path, should_cancel)?;
        } else if source_path.is_file() {
            copy_file(&source_path, &target_path).map_err(StartError::Failed)?;
        }
    }
    Ok(())
}

fn sync_runtime_template_with_cancel<F>(
    template_root: &Path,
    runtime_root: &Path,
    should_cancel: &F,
) -> Result<(), StartError>
where
    F: Fn() -> bool,
{
    if should_cancel() {
        return Err(StartError::Cancelled);
    }

    fs::create_dir_all(runtime_root)
        .map_err(|error| StartError::Failed(format!("Failed to create {}: {error}", runtime_root.display())))?;
    for entry in TEMPLATE_ENTRIES {
        if should_cancel() {
            return Err(StartError::Cancelled);
        }

        let source = template_root.join(entry);
        let target = runtime_root.join(entry);
        if source.is_dir() {
            copy_tree_with_cancel(&source, &target, should_cancel)?;
        } else if source.is_file() {
            copy_file(&source, &target).map_err(StartError::Failed)?;
        } else {
            return Err(StartError::Failed(format!(
                "The desktop runtime template is missing {}.",
                source.display()
            )));
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
                emit_setup_progress(&stdout_shared, &stdout_app, payload);
                continue;
            }

            if line.starts_with("READY ") {
                let base_url = line["READY ".len()..].to_string();
                if parse_sidecar_port(&line).is_none() {
                    mark_start_failure(
                        &stdout_shared,
                        &stdout_app,
                        "The sidecar reported an invalid ready URL.".into(),
                    );
                    return;
                }
                mark_runtime_ready(&stdout_shared, &stdout_app, base_url);
                continue;
            }

            emit_setup_log(&stdout_shared, &stdout_app, &line);
        }

        let failure_message = {
            let runtime = stdout_shared.runtime.lock().expect("sidecar runtime lock poisoned");
            if runtime.starting && runtime.base_url.is_none() && !runtime.cancel_requested {
                Some(build_start_failure_message(
                    &runtime,
                    "The sidecar exited before it reported a ready URL.",
                ))
            } else {
                None
            }
        };
        if let Some(message) = failure_message {
            mark_start_failure(&stdout_shared, &stdout_app, message);
        }
    });

    let stderr_app = app;
    let stderr_shared = shared;
    thread::spawn(move || {
        let reader = BufReader::new(stderr);
        for line_result in reader.lines() {
            match line_result {
                Ok(line) => emit_setup_log(&stderr_shared, &stderr_app, &line),
                Err(error) => {
                    emit_setup_log(
                        &stderr_shared,
                        &stderr_app,
                        &format!("Failed to read sidecar stderr: {error}"),
                    );
                    return;
                }
            }
        }
    });

    Ok(())
}

fn start_sidecar(app: &AppHandle, shared: Arc<SidecarShared>) -> Result<(), StartError> {
    let template_root = resolve_runtime_template_dir(app).map_err(StartError::Failed)?;
    let support_root = app
        .path()
        .app_data_dir()
        .map_err(|error| StartError::Failed(format!("Failed to resolve the app support directory: {error}")))?;
    let runtime_root = support_root.join("runtime");
    let index_db_path = support_root.join("index.sqlite3");
    let bundled_resources_dir = resolve_bundled_resources_dir(app, &template_root);
    let uv_binary = resolve_uv_binary(app, &template_root).map_err(StartError::Failed)?;
    let port = reserve_sidecar_port().map_err(StartError::Failed)?;

    check_cancellation(&shared)?;

    emit_setup_progress(
        &shared,
        app,
        serde_json::json!({
            "task": "sync-runtime",
            "phase": "start",
            "message": "Copying the bundled runtime files into Application Support",
            "current": 0,
            "total": SETUP_STEPS.len(),
        }),
    );
    sync_runtime_template_with_cancel(&template_root, &runtime_root, &|| cancellation_requested(&shared))?;
    emit_setup_progress(
        &shared,
        app,
        serde_json::json!({
            "task": "sync-runtime",
            "phase": "finish",
            "message": "Bundled runtime files are ready",
            "current": 1,
            "total": SETUP_STEPS.len(),
        }),
    );
    check_cancellation(&shared)?;

    let args = build_bootstrap_args(
        &runtime_root,
        &index_db_path,
        port,
        bundled_resources_dir.as_deref(),
    );

    let mut command = Command::new(&uv_binary);
    command
        .args(&args)
        .current_dir(&runtime_root)
        .env("PYTHONUNBUFFERED", "1");
    for (key, value) in build_sidecar_env(&support_root) {
        command.env(key, value);
    }
    command.stdout(Stdio::piped()).stderr(Stdio::piped());

    let mut child = command
        .spawn()
        .map_err(|error| StartError::Failed(format!("Failed to start the sidecar bootstrap: {error}")))?;

    spawn_output_threads(app.clone(), shared.clone(), &mut child).map_err(StartError::Failed)?;

    let mut runtime = shared.runtime.lock().expect("sidecar runtime lock poisoned");
    runtime.child = Some(child);
    Ok(())
}

fn refresh_child_state(runtime: &mut SidecarRuntime) {
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
}

fn start_sidecar_in_background(app: &AppHandle, state: &SidecarState) -> Result<(), String> {
    let shared = state.shared.clone();
    let should_start = {
        let mut runtime = shared.runtime.lock().expect("sidecar runtime lock poisoned");
        refresh_child_state(&mut runtime);

        if runtime.base_url.is_some() || runtime.starting {
            return Ok(());
        }

        runtime.last_error = None;
        runtime.base_url = None;
        runtime.child = None;
        runtime.cancel_requested = false;
        runtime.starting = true;
        reset_bootstrap(&mut runtime);
        true
    };

    if should_start {
        let app_handle = app.clone();
        thread::spawn(move || {
            if let Err(error) = start_sidecar(&app_handle, shared.clone()) {
                match error {
                    StartError::Failed(message) => mark_start_failure(&shared, &app_handle, message),
                    StartError::Cancelled => {
                        mark_start_cancelled(&shared);
                    }
                }
            }
        });
    }

    Ok(())
}

fn ensure_sidecar(app: &AppHandle, state: &SidecarState) -> Result<String, String> {
    start_sidecar_in_background(app, state)?;

    let shared = state.shared.clone();
    let mut runtime = shared.runtime.lock().expect("sidecar runtime lock poisoned");
    if let Some(base_url) = runtime.base_url.clone() {
        return Ok(base_url);
    }

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
        runtime.cancel_requested = false;
        runtime.last_error = None;
        runtime.child.take()
    };
    if let Some(mut child) = child {
        let _ = child.kill();
        let _ = child.wait();
    }
}

#[tauri::command]
async fn pick_folder(app: AppHandle) -> Result<Option<String>, String> {
    let (tx, rx) = sync_channel(1);
    app.dialog()
        .file()
        .set_title("Choose a photo folder")
        .pick_folder(move |folder| {
            let _ = tx.send(folder.map(|path| path.to_string()));
        });
    rx.recv()
        .map_err(|error| format!("Failed to receive the selected folder: {error}"))
}

#[tauri::command]
fn start_runtime(app: AppHandle, state: State<'_, SidecarState>) -> Result<BootstrapSnapshot, String> {
    start_sidecar_in_background(&app, state.inner())?;
    Ok(snapshot_for_app(&app, bootstrap_snapshot(state.inner())))
}

#[tauri::command]
fn cancel_runtime_setup(app: AppHandle, state: State<'_, SidecarState>) -> BootstrapSnapshot {
    let (snapshot, child) = mark_setup_cancelling(state.inner());
    if let Some(mut child) = child {
        let _ = child.kill();
        let _ = child.wait();
        return snapshot_for_app(&app, mark_start_cancelled(&state.inner().shared));
    }
    snapshot_for_app(&app, snapshot)
}

#[tauri::command]
fn runtime_bootstrap_state(app: AppHandle, state: State<'_, SidecarState>) -> BootstrapSnapshot {
    snapshot_for_app(&app, bootstrap_snapshot(state.inner()))
}

#[tauri::command]
fn complete_onboarding(app: AppHandle, state: State<'_, SidecarState>) -> Result<BootstrapSnapshot, String> {
    let app_data_dir = app
        .path()
        .app_data_dir()
        .map_err(|error| format!("Failed to resolve the app support directory: {error}"))?;
    fs::create_dir_all(&app_data_dir)
        .map_err(|error| format!("Failed to create {}: {error}", app_data_dir.display()))?;
    let marker_path = onboarding_marker_path(&app_data_dir);
    fs::write(&marker_path, b"ready")
        .map_err(|error| format!("Failed to write {}: {error}", marker_path.display()))?;
    Ok(complete_onboarding_snapshot(bootstrap_snapshot(state.inner())))
}

#[tauri::command]
fn sidecar_base_url(app: AppHandle, state: State<'_, SidecarState>) -> Result<String, String> {
    ensure_sidecar(&app, state.inner())
}

#[tauri::command]
fn open_uninstaller(app: AppHandle) -> Result<(), String> {
    let resolved_helper_path = resolve_uninstaller_app(&app)?;
    let helper_path = if let Some(bundle_dir) = resolve_app_bundle_dir(&app) {
        if resolved_helper_path.starts_with(&bundle_dir) {
            stage_uninstaller_for_self_removal(
                &resolved_helper_path,
                &env::temp_dir().join("semanticgallery-uninstaller"),
            )?
        } else {
            resolved_helper_path
        }
    } else {
        resolved_helper_path
    };
    let status = Command::new("open")
        .arg(&helper_path)
        .status()
        .map_err(|error| format!("Failed to launch the uninstaller: {error}"))?;
    if !status.success() {
        return Err("The uninstaller app did not launch successfully.".into());
    }
    Ok(())
}

#[tauri::command]
fn desktop_automation_tools_enabled(app: AppHandle) -> Result<bool, String> {
    let app_data_dir = app
        .path()
        .app_data_dir()
        .map_err(|error| format!("Failed to resolve the app support directory: {error}"))?;
    Ok(automation_tools_enabled_for_dir(&app_data_dir))
}

pub fn run() {
    let app = tauri::Builder::default()
        .manage(desktop_core::DesktopCoreState::default())
        .manage(SidecarState::default())
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            desktop_core::desktop_delete_image,
            desktop_core::desktop_delete_images,
            desktop_core::desktop_metadata,
            desktop_core::desktop_refresh_folder,
            desktop_core::desktop_run_stage2,
            desktop_core::desktop_runtime_status,
            desktop_core::desktop_search_similar,
            desktop_core::desktop_search_text,
            desktop_core::desktop_search_uploaded_image,
            desktop_core::desktop_select_folder,
            desktop_automation_tools_enabled,
            pick_folder,
            cancel_runtime_setup,
            complete_onboarding,
            open_uninstaller,
            runtime_bootstrap_state,
            start_runtime,
            sidecar_base_url
        ])
        .setup(|app| {
            spawn_automation_request_watcher(app.handle())?;
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("failed to build SemanticGallery desktop shell");

    let _ = cleanup_stale_sidecars(&app.handle());

    app.run(|app_handle, event| {
        if matches!(event, tauri::RunEvent::Exit | tauri::RunEvent::ExitRequested { .. }) {
            stop_sidecar(&app_handle.state::<SidecarState>());
        }
    });
}

#[cfg(test)]
mod tests {
    use super::{
        apply_setup_progress, build_bootstrap_args, build_sidecar_env, build_start_failure_message,
        automation_tools_enabled_for_dir, automation_tools_enabled_from_args,
        automation_tools_env_value_enabled, automation_tools_marker_path, command_matches_sidecar,
        complete_onboarding_snapshot, copy_file,
        locate_runtime_template_root, normalize_setup_log_line, onboarding_marker_path,
        onboarding_required_for_dir, parse_setup_payload, parse_sidecar_port, runtime_files_look_ready,
        runtime_ready_for_dir, runtime_ready_marker_path, setup_steps,
        stage_uninstaller_for_self_removal, sync_runtime_template, SidecarRuntime,
    };
    use std::ffi::OsStr;
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

    fn write_fake_python(runtime_root: &PathBuf, body: &str) {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            let python_path = runtime_root.join(".venv").join("bin").join("python");
            fs::create_dir_all(python_path.parent().expect("python parent should exist"))
                .expect("failed to create python directory");
            fs::write(&python_path, format!("#!/bin/sh\n{body}\n")).expect("failed to write fake python");
            let mut permissions = fs::metadata(&python_path)
                .expect("failed to read fake python metadata")
                .permissions();
            permissions.set_mode(0o755);
            fs::set_permissions(&python_path, permissions).expect("failed to set fake python permissions");
        }
    }

    #[test]
    fn parse_sidecar_port_reads_ready_line() {
        let value = parse_sidecar_port("READY http://127.0.0.1:38291");
        assert_eq!(value, Some(38291));
    }

    #[test]
    fn command_matches_sidecar_accepts_matching_runtime_root() {
        let runtime_root =
            PathBuf::from("/Users/test/Library/Application Support/com.semanticgallery.desktop/runtime");
        let command_line = format!(
            "/Users/test/Library/Application Support/com.semanticgallery.desktop/runtime/.venv/bin/python -m desktop_runtime.sidecar_main --workspace-root {}",
            runtime_root.display()
        );
        assert!(command_matches_sidecar(&command_line, &runtime_root));
    }

    #[test]
    fn command_matches_sidecar_rejects_other_runtime_roots() {
        let runtime_root =
            PathBuf::from("/Users/test/Library/Application Support/com.semanticgallery.desktop/runtime");
        let other_root = PathBuf::from("/Users/test/Library/Application Support/other/runtime");
        let command_line = format!(
            "/Users/test/Library/Application Support/other/runtime/.venv/bin/python -m desktop_runtime.sidecar_main --workspace-root {}",
            other_root.display()
        );
        assert!(!command_matches_sidecar(&command_line, &runtime_root));
    }

    #[test]
    fn build_start_failure_message_includes_recent_runtime_logs() {
        let mut runtime = SidecarRuntime::default();
        runtime.bootstrap.logs = vec![
            "Starting SemanticGallery desktop runtime.".into(),
            "Preparing Python dependencies".into(),
            "No module named 'mlx'".into(),
        ];

        let message = build_start_failure_message(
            &runtime,
            "The sidecar exited before it reported a ready URL.",
        );

        assert!(message.contains("Recent runtime log:"));
        assert!(message.contains("Preparing Python dependencies"));
        assert!(message.contains("No module named 'mlx'"));
    }

    #[test]
    fn build_sidecar_env_uses_support_directory_paths() {
        let support_root = PathBuf::from("/Users/test/Library/Application Support/com.semanticgallery.desktop");
        let env_pairs = build_sidecar_env(&support_root);
        let env_map = env_pairs.into_iter().collect::<std::collections::HashMap<_, _>>();

        assert_eq!(
            env_map.get("UV_CACHE_DIR"),
            Some(&support_root.join("cache").join("uv").to_string_lossy().into_owned())
        );
        assert_eq!(
            env_map.get("UV_PYTHON_INSTALL_DIR"),
            Some(&support_root.join("python").to_string_lossy().into_owned())
        );
        assert_eq!(
            env_map.get("HF_HOME"),
            Some(&support_root.join("cache").join("huggingface").to_string_lossy().into_owned())
        );
    }

    #[test]
    fn parse_setup_payload_reads_json_progress_event() {
        let payload =
            parse_setup_payload("SETUP {\"task\":\"prepare-base-model\",\"phase\":\"finish\",\"current\":4,\"total\":6}")
                .expect("payload should parse");
        assert_eq!(payload["task"], "prepare-base-model");
        assert_eq!(payload["phase"], "finish");
        assert_eq!(payload["current"], 4);
    }

    #[test]
    fn normalize_setup_log_line_filters_uvicorn_noise() {
        assert_eq!(
            normalize_setup_log_line(
                "INFO: Uvicorn running on http://127.0.0.1:60538 (Press CTRL+C to quit)"
            ),
            None
        );
        assert_eq!(
            normalize_setup_log_line(
                "INFO: 127.0.0.1:60562 - \"OPTIONS /api/runtime/status HTTP/1.1\" 405 Method Not Allowed"
            ),
            None
        );
        assert_eq!(
            normalize_setup_log_line(
                "/Users/example/runtime/desktop_runtime/sidecar_main.py:62: DeprecationWarning:"
            ),
            None
        );
    }

    #[test]
    fn normalize_setup_log_line_keeps_real_error_lines() {
        assert_eq!(
            normalize_setup_log_line("ERROR: model weights are missing"),
            Some("ERROR: model weights are missing".into())
        );
    }

    #[test]
    fn onboarding_required_checks_for_marker_file() {
        let root = temp_root("onboarding");
        assert!(onboarding_required_for_dir(&root));

        let marker_path = onboarding_marker_path(&root);
        fs::write(&marker_path, "ready").expect("failed to write marker");

        assert!(onboarding_required_for_dir(&root));

        let runtime_ready_path = runtime_ready_marker_path(&root);
        fs::write(&runtime_ready_path, "ready").expect("failed to write runtime marker");

        let runtime_root = root.join("runtime");
        fs::create_dir_all(runtime_root.join(".venv").join("bin")).expect("failed to create python path");
        fs::create_dir_all(
            runtime_root
                .join(".cache")
                .join("mlx")
                .join("siglip2-base-patch16-224-f32"),
        )
        .expect("failed to create model path");
        fs::create_dir_all(
            runtime_root
                .join(".cache")
                .join("semanticgallery")
                .join("stage2_public_anchor")
                .join("extracted")
                .join("flickr30k"),
        )
        .expect("failed to create flickr path");
        fs::create_dir_all(
            runtime_root
                .join(".cache")
                .join("semanticgallery")
                .join("stage2_public_anchor")
                .join("extracted")
                .join("screen2words"),
        )
        .expect("failed to create screen2words path");
        write_fake_python(&runtime_root, "exit 0");
        fs::write(
            runtime_root
                .join(".cache")
                .join("mlx")
                .join("siglip2-base-patch16-224-f32")
                .join("config.json"),
            "{}",
        )
        .expect("failed to write model config");
        fs::write(
            runtime_root
                .join(".cache")
                .join("semanticgallery")
                .join("stage2_public_anchor")
                .join("extracted")
                .join("flickr30k")
                .join("captions.txt"),
            "",
        )
        .expect("failed to write captions");
        fs::write(
            runtime_root
                .join(".cache")
                .join("semanticgallery")
                .join("stage2_public_anchor")
                .join("extracted")
                .join("screen2words")
                .join("manifest.jsonl"),
            "",
        )
        .expect("failed to write manifest");

        assert!(runtime_files_look_ready(&root));
        assert!(runtime_ready_for_dir(&root));
        assert!(!onboarding_required_for_dir(&root));

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn onboarding_required_when_runtime_python_sanity_check_fails() {
        let root = temp_root("onboarding-sanity");
        fs::write(onboarding_marker_path(&root), "ready").expect("failed to write onboarding marker");
        fs::write(runtime_ready_marker_path(&root), "ready").expect("failed to write runtime marker");

        let runtime_root = root.join("runtime");
        fs::create_dir_all(
            runtime_root
                .join(".cache")
                .join("mlx")
                .join("siglip2-base-patch16-224-f32"),
        )
        .expect("failed to create model path");
        fs::create_dir_all(
            runtime_root
                .join(".cache")
                .join("semanticgallery")
                .join("stage2_public_anchor")
                .join("extracted")
                .join("flickr30k"),
        )
        .expect("failed to create flickr path");
        fs::create_dir_all(
            runtime_root
                .join(".cache")
                .join("semanticgallery")
                .join("stage2_public_anchor")
                .join("extracted")
                .join("screen2words"),
        )
        .expect("failed to create screen2words path");
        write_fake_python(&runtime_root, "exit 1");
        fs::write(
            runtime_root
                .join(".cache")
                .join("mlx")
                .join("siglip2-base-patch16-224-f32")
                .join("config.json"),
            "{}",
        )
        .expect("failed to write model config");
        fs::write(
            runtime_root
                .join(".cache")
                .join("semanticgallery")
                .join("stage2_public_anchor")
                .join("extracted")
                .join("flickr30k")
                .join("captions.txt"),
            "",
        )
        .expect("failed to write captions");
        fs::write(
            runtime_root
                .join(".cache")
                .join("semanticgallery")
                .join("stage2_public_anchor")
                .join("extracted")
                .join("screen2words")
                .join("manifest.jsonl"),
            "",
        )
        .expect("failed to write manifest");

        assert!(runtime_files_look_ready(&root));
        assert!(onboarding_required_for_dir(&root));

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn complete_onboarding_snapshot_clears_the_flag_without_rechecking_runtime() {
        let snapshot = complete_onboarding_snapshot(SidecarRuntime::default().bootstrap);

        assert!(!snapshot.onboarding_required);
        assert_eq!(snapshot.status, "idle");
    }

    #[test]
    fn automation_tools_flag_tracks_the_marker_file() {
        let root = temp_root("automation-tools");
        assert!(!automation_tools_enabled_for_dir(&root));
        fs::write(automation_tools_marker_path(&root), "ready").expect("failed to write automation marker");
        assert!(automation_tools_enabled_for_dir(&root));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn automation_tools_flag_accepts_the_test_mode_argument() {
        assert!(automation_tools_enabled_from_args([
            "/Applications/SemanticGallery.app/Contents/MacOS/semanticgallery_desktop",
            "--semanticgallery-test-mode",
        ]));
        assert!(!automation_tools_enabled_from_args([
            "/Applications/SemanticGallery.app/Contents/MacOS/semanticgallery_desktop",
            "--other-flag",
        ]));
    }

    #[test]
    fn automation_tools_flag_accepts_truthy_environment_values() {
        assert!(automation_tools_env_value_enabled(Some(OsStr::new("1"))));
        assert!(automation_tools_env_value_enabled(Some(OsStr::new("true"))));
        assert!(automation_tools_env_value_enabled(Some(OsStr::new("yes"))));
        assert!(!automation_tools_env_value_enabled(Some(OsStr::new("0"))));
        assert!(!automation_tools_env_value_enabled(Some(OsStr::new("false"))));
        assert!(!automation_tools_env_value_enabled(None::<&OsStr>));
    }

    #[test]
    fn setup_steps_cover_all_bootstrap_phases() {
        let steps = setup_steps();
        assert_eq!(steps.len(), 6);
        assert_eq!(steps[0].task, "sync-runtime");
        assert_eq!(steps[5].task, "finish-setup");
    }

    #[test]
    fn apply_setup_progress_marks_matching_step_and_counter() {
        let mut runtime = SidecarRuntime::default();
        apply_setup_progress(
            &mut runtime,
            &serde_json::json!({
                "task": "prepare-dependencies",
                "phase": "finish",
                "message": "Python dependencies are ready",
                "current": 3,
                "total": 6,
            }),
        );

        assert_eq!(runtime.bootstrap.current_step, 3);
        assert_eq!(runtime.bootstrap.total_steps, 6);
        assert_eq!(runtime.bootstrap.message, "Python dependencies are ready");
        let step = runtime
            .bootstrap
            .steps
            .iter()
            .find(|step| step.task == "prepare-dependencies")
            .expect("step should exist");
        assert_eq!(step.status, "done");
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
        fs::write(nested_root.join("requirements.txt"), "fastapi\n").expect("failed to write file");
        fs::write(nested_root.join("mlx_pipeline.py"), "VALUE = 1\n").expect("failed to write file");

        let resolved = locate_runtime_template_root(&resource_root).expect("template root should resolve");
        assert_eq!(resolved, nested_root);

        let _ = fs::remove_dir_all(resource_root);
    }

    #[test]
    fn stage_uninstaller_for_self_removal_copies_helper_outside_the_app_bundle() {
        let bundle_root = temp_root("bundle");
        let helper_root = bundle_root
            .join("SemanticGallery.app")
            .join("Contents")
            .join("Resources")
            .join("resources")
            .join("uninstall")
            .join("Uninstall SemanticGallery.app");
        let helper_info = helper_root.join("Contents").join("Info.plist");
        fs::create_dir_all(helper_info.parent().expect("helper parent should exist"))
            .expect("failed to create helper bundle");
        fs::write(&helper_info, "helper").expect("failed to write helper plist");

        let staging_root = temp_root("staged-uninstaller");
        let staged_helper =
            stage_uninstaller_for_self_removal(&helper_root, &staging_root).expect("staging should succeed");

        assert_ne!(staged_helper, helper_root);
        assert!(staged_helper.starts_with(&staging_root));
        assert!(staged_helper.join("Contents").join("Info.plist").is_file());

        let _ = fs::remove_dir_all(bundle_root);
        let _ = fs::remove_dir_all(staging_root);
    }
}
