use tauri::{AppHandle, State};
use tauri_plugin_dialog::DialogExt;

struct SidecarState {
    port: u16,
}

fn resolve_sidecar_port() -> u16 {
    if let Ok(line) = std::env::var("SEMANTICGALLERY_SIDECAR_READY_LINE") {
        if let Some(port) = parse_sidecar_port(&line) {
            return port;
        }
    }

    if let Ok(raw_port) = std::env::var("SEMANTICGALLERY_SIDECAR_PORT") {
        if let Ok(port) = raw_port.parse::<u16>() {
            return port;
        }
    }

    36168
}

fn parse_sidecar_port(line: &str) -> Option<u16> {
    let (_, url) = line.split_once(' ')?;
    let (_, port_text) = url.rsplit_once(':')?;
    port_text.parse::<u16>().ok()
}

#[tauri::command]
fn pick_folder(app: AppHandle) -> Result<Option<String>, String> {
    let folder = app.dialog().file().blocking_pick_folder();
    Ok(folder.map(|path| path.to_string()))
}

#[tauri::command]
fn sidecar_base_url(state: State<'_, SidecarState>) -> String {
    format!("http://127.0.0.1:{}", state.port)
}

pub fn run() {
    tauri::Builder::default()
        .manage(SidecarState {
            port: resolve_sidecar_port(),
        })
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![pick_folder, sidecar_base_url])
        .run(tauri::generate_context!())
        .expect("failed to run SemanticGallery desktop shell");
}

#[cfg(test)]
mod tests {
    use super::parse_sidecar_port;

    #[test]
    fn parse_sidecar_port_reads_ready_line() {
        let value = parse_sidecar_port("READY http://127.0.0.1:38291");
        assert_eq!(value, Some(38291));
    }
}
