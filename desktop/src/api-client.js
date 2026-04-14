import { convertFileSrc, invoke } from "@tauri-apps/api/core";

let cachedBaseUrl = null;

export function resetSidecarBaseUrl() {
  cachedBaseUrl = null;
}

export async function sidecarBaseUrl() {
  if (!cachedBaseUrl) {
    cachedBaseUrl = await invoke("sidecar_base_url");
  }
  return cachedBaseUrl;
}

export async function startRuntime() {
  return invoke("start_runtime");
}

export async function cancelRuntimeSetup() {
  return invoke("cancel_runtime_setup");
}

export async function completeOnboarding() {
  return invoke("complete_onboarding");
}

export async function runtimeBootstrapState() {
  return invoke("runtime_bootstrap_state");
}

export async function openUninstaller() {
  return invoke("open_uninstaller");
}

export async function desktopRuntimeStatus() {
  return invoke("desktop_runtime_status");
}

export async function desktopAutomationToolsEnabled() {
  return invoke("desktop_automation_tools_enabled");
}

export async function desktopSelectFolder(folderPath) {
  return invoke("desktop_select_folder", { args: { folderPath } });
}

export async function desktopRefreshFolder({ lightweight = false } = {}) {
  return invoke("desktop_refresh_folder", { args: { lightweight } });
}

export async function desktopSearchText(queryText, limit) {
  return invoke("desktop_search_text", { args: { queryText, limit } });
}

export async function desktopSearchUploadedImage(imageBase64, filename, limit) {
  return invoke("desktop_search_uploaded_image", {
    args: { imageBase64, filename, limit }
  });
}

export async function desktopSearchSimilar(relativePath, limit) {
  return invoke("desktop_search_similar", { args: { relativePath, limit } });
}

export async function desktopMetadata(relativePath) {
  return invoke("desktop_metadata", { args: { relativePath } });
}

export async function desktopDeleteImage(relativePath) {
  return invoke("desktop_delete_image", { args: { relativePath } });
}

export async function desktopDeleteImages(paths) {
  return invoke("desktop_delete_images", { args: { paths } });
}

export async function desktopRunStage2() {
  return invoke("desktop_run_stage2");
}

export function toAssetUrl(path) {
  if (!path) {
    return "";
  }
  if (isAbsoluteUrl(path)) {
    return path;
  }
  return convertFileSrc(path);
}

function isAbsoluteUrl(path) {
  return /^https?:\/\//i.test(path);
}

function shouldSetJsonContentType(body) {
  if (body == null) {
    return false;
  }
  if (typeof FormData !== "undefined" && body instanceof FormData) {
    return false;
  }
  if (typeof Blob !== "undefined" && body instanceof Blob) {
    return false;
  }
  if (typeof URLSearchParams !== "undefined" && body instanceof URLSearchParams) {
    return false;
  }
  return true;
}

export async function resolveSidecarUrl(path) {
  if (isAbsoluteUrl(path)) {
    return path;
  }
  const baseUrl = await sidecarBaseUrl();
  return new URL(path, `${baseUrl}/`).toString();
}

export async function fetchJson(path, init = {}) {
  const requestUrl = await resolveSidecarUrl(path);
  const headers = new Headers(init.headers ?? {});
  if (shouldSetJsonContentType(init.body) && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  const response = await fetch(requestUrl, {
    ...init,
    headers: Object.fromEntries(headers.entries())
  });

  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`.trim();
    try {
      const payload = await response.json();
      if (typeof payload.detail === "string" && payload.detail) {
        detail = payload.detail;
      }
    } catch {
      // Keep the fallback message when the body is not JSON.
    }
    throw new Error(detail);
  }

  return response.json();
}

export async function pickFolder() {
  return invoke("pick_folder");
}
