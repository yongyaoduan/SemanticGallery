import { invoke } from "@tauri-apps/api/core";

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
