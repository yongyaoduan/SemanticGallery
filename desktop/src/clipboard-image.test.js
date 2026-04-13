import { describe, expect, it } from "vitest";

import { extractImageFileFromClipboardItems } from "./clipboard-image";

describe("clipboard image helpers", () => {
  it("returns the first pasted image file", () => {
    const file = { name: "query.png" };
    const items = [
      { kind: "string", type: "text/plain", getAsFile: () => null },
      { kind: "file", type: "image/png", getAsFile: () => file }
    ];

    expect(extractImageFileFromClipboardItems(items)).toBe(file);
  });

  it("ignores non-image clipboard items", () => {
    const items = [
      { kind: "file", type: "application/pdf", getAsFile: () => ({ name: "doc.pdf" }) },
      { kind: "string", type: "text/plain", getAsFile: () => null }
    ];

    expect(extractImageFileFromClipboardItems(items)).toBeNull();
  });
});
