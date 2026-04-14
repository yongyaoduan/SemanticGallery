import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({
  convertFileSrc: vi.fn((path) => `asset://localhost/${path}`),
  invoke: vi.fn()
}));

import { convertFileSrc, invoke } from "@tauri-apps/api/core";

import {
  desktopAutomationToolsEnabled,
  desktopDeleteImages,
  desktopSearchText,
  fetchJson,
  openUninstaller,
  resetSidecarBaseUrl,
  resolveSidecarUrl,
  toAssetUrl
} from "./api-client";

describe("api client", () => {
  beforeEach(() => {
    vi.resetAllMocks();
    resetSidecarBaseUrl();
    global.fetch = vi.fn();
    convertFileSrc.mockImplementation((path) => `asset://localhost/${path}`);
    invoke.mockResolvedValue("http://127.0.0.1:60538");
  });

  it("does not add a JSON content-type header to GET requests", async () => {
    global.fetch.mockResolvedValue({
      ok: true,
      json: async () => ({ setupStatus: "ready" })
    });

    await fetchJson("/api/runtime/status");

    expect(global.fetch).toHaveBeenCalledWith(
      "http://127.0.0.1:60538/api/runtime/status",
      expect.objectContaining({
        headers: {}
      })
    );
  });

  it("adds a JSON content-type header when sending a request body", async () => {
    global.fetch.mockResolvedValue({
      ok: true,
      json: async () => ({ ok: true })
    });

    await fetchJson("/api/folders/select", {
      method: "POST",
      body: JSON.stringify({ folderPath: "/tmp/gallery" })
    });

    expect(global.fetch).toHaveBeenCalledWith(
      "http://127.0.0.1:60538/api/folders/select",
      expect.objectContaining({
        method: "POST",
        headers: expect.objectContaining({
          "content-type": "application/json"
        })
      })
    );
  });

  it("leaves multipart requests untouched so the browser can set the boundary", async () => {
    global.fetch.mockResolvedValue({
      ok: true,
      json: async () => ({ ok: true })
    });
    const formData = new FormData();
    formData.append("image", new Blob(["test"], { type: "image/png" }), "query.png");

    await fetchJson("/api/search/image?limit=25", {
      method: "POST",
      body: formData
    });

    expect(global.fetch).toHaveBeenCalledWith(
      "http://127.0.0.1:60538/api/search/image?limit=25",
      expect.objectContaining({
        method: "POST",
        headers: {}
      })
    );
  });

  it("invokes the desktop shell when opening the bundled uninstaller", async () => {
    await openUninstaller();

    expect(invoke).toHaveBeenCalledWith("open_uninstaller");
  });

  it("resolves relative media paths against the sidecar base url", async () => {
    await expect(resolveSidecarUrl("/thumbs/cat.jpg")).resolves.toBe("http://127.0.0.1:60538/thumbs/cat.jpg");
  });

  it("routes text search through the Rust desktop command bridge", async () => {
    await desktopSearchText("cat", 25);

    expect(invoke).toHaveBeenCalledWith("desktop_search_text", {
      args: { queryText: "cat", limit: 25 }
    });
  });

  it("reads the desktop automation tools flag through the Rust command bridge", async () => {
    await desktopAutomationToolsEnabled();

    expect(invoke).toHaveBeenCalledWith("desktop_automation_tools_enabled");
  });

  it("routes batch delete through the Rust desktop command bridge", async () => {
    await desktopDeleteImages(["cat.jpg", "dog.jpg"]);

    expect(invoke).toHaveBeenCalledWith("desktop_delete_images", {
      args: { paths: ["cat.jpg", "dog.jpg"] }
    });
  });

  it("converts local file paths into Tauri asset urls", () => {
    expect(toAssetUrl("/Users/example/photo.jpg")).toBe("asset://localhost//Users/example/photo.jpg");
    expect(convertFileSrc).toHaveBeenCalledWith("/Users/example/photo.jpg");
  });
});
