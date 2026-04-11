import { invoke } from "@tauri-apps/api/core";

let cachedBaseUrl = null;

export async function sidecarBaseUrl() {
  if (!cachedBaseUrl) {
    cachedBaseUrl = await invoke("sidecar_base_url");
  }
  return cachedBaseUrl;
}

export async function fetchJson(path, init = {}) {
  const baseUrl = await sidecarBaseUrl();
  const response = await fetch(`${baseUrl}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(init.headers ?? {})
    },
    ...init
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
