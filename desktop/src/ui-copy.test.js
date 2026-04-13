import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

const here = path.dirname(fileURLToPath(import.meta.url));
const files = [
  path.join(here, "../index.html"),
  path.join(here, "main.js"),
  path.join(here, "state.js"),
  path.join(here, "../src-tauri/src/lib.rs")
];

const bannedPhrases = [
  "Install the local runtime to prepare Python, search dependencies, and the local encoder.",
  "SemanticGallery only downloads missing runtime pieces and keeps your private gallery local.",
  "SemanticGallery sets up a local Python runtime, search dependencies, the base encoder model, and the public adaptation set.",
  "It only downloads missing components and keeps your photos on this Mac.",
  "Embeddings are reused by image content hash, while each folder keeps its own path records.",
  "Bootstrap logs will appear here as each step starts.",
  "Choose a folder in Settings to build the first searchable album.",
  "Choose a folder in Settings to enable search",
  "Choose a folder from Settings",
  "Choose a folder from Finder.",
  "No background task has started yet.",
  "Your local search engine is ready.",
  "Continue to the main workspace to choose a folder and build the first index."
];

describe("ui copy", () => {
  it("does not keep the removed off-topic helper copy in the desktop sources", () => {
    const source = files.map((filePath) => fs.readFileSync(filePath, "utf-8")).join("\n");

    for (const phrase of bannedPhrases) {
      expect(source).not.toContain(phrase);
    }
  });

  it("removes the workspace status card from the search screen", () => {
    const html = fs.readFileSync(path.join(here, "../index.html"), "utf-8");

    expect(html).not.toContain("workspace-status-card");
    expect(html).not.toContain("Latest status");
  });
});
