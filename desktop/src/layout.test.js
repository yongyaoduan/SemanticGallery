import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

const styles = readFileSync(new URL("./styles.css", import.meta.url), "utf8");
const html = readFileSync(new URL("../index.html", import.meta.url), "utf8");

describe("desktop layout", () => {
  it("pins the search result grid to five columns on wide windows", () => {
    expect(styles).toContain("grid-template-columns: repeat(5, minmax(0, 1fr));");
  });

  it("keeps narrower windows responsive", () => {
    expect(styles).toMatch(/@media\s*\(max-width:\s*1400px\)\s*\{[\s\S]*?\.results\s*\{[\s\S]*?grid-template-columns:\s*repeat\(4,\s*minmax\(0,\s*1fr\)\);/);
    expect(styles).toMatch(/@media\s*\(max-width:\s*1120px\)\s*\{[\s\S]*?\.results\s*\{[\s\S]*?grid-template-columns:\s*repeat\(3,\s*minmax\(0,\s*1fr\)\);/);
    expect(styles).toMatch(/@media\s*\(max-width:\s*860px\)\s*\{[\s\S]*?\.results\s*\{[\s\S]*?grid-template-columns:\s*repeat\(2,\s*minmax\(0,\s*1fr\)\);/);
    expect(styles).toMatch(/@media\s*\(max-width:\s*640px\)\s*\{[\s\S]*?\.results\s*\{[\s\S]*?grid-template-columns:\s*1fr;/);
  });

  it("does not lazy-load the result thumbnails", () => {
    expect(html).toContain('<img class="result-thumb" alt="" />');
    expect(html).not.toContain('loading="lazy"');
  });

  it("wraps the Stage 2 progress area so idle runs can hide the whole shell", () => {
    expect(html).toContain('<div id="settings-stage2-progress-shell" class="settings-progress">');
  });

  it("wraps the index progress area so idle and completed indexing can hide the whole shell", () => {
    expect(html).toContain('<div id="settings-index-progress-shell" class="settings-progress">');
  });

  it("keeps a dedicated Stage 2 feedback line for validation failures", () => {
    expect(html).toContain('id="settings-stage2-feedback"');
  });

  it("keeps the hidden automation tools in both settings and workspace for regression runs", () => {
    expect(html).toContain('id="automation-folder-shell"');
    expect(html).toContain('id="automation-folder-input"');
    expect(html).toContain('id="automation-folder-submit"');
    expect(html).toContain('id="automation-image-shell"');
    expect(html).toContain('id="automation-image-input"');
    expect(html).toContain('id="automation-image-submit"');
  });
});
