import { describe, expect, it } from "vitest";

import { attachImageFallback, buildImageSourceCandidates } from "./image-fallback";

describe("image fallback", () => {
  it("deduplicates empty and repeated image sources", () => {
    expect(buildImageSourceCandidates("", "/tmp/full.jpg", "/tmp/full.jpg", null)).toEqual(["/tmp/full.jpg"]);
  });

  it("falls back to the next candidate when the current source fails", () => {
    const image = {
      _src: "",
      onerror: null,
      set src(value) {
        this._src = value;
      },
      get src() {
        return this._src;
      },
      getAttribute(name) {
        return name === "src" ? this._src : null;
      },
      removeAttribute(name) {
        if (name === "src") {
          this._src = "";
        }
      },
      dispatchEvent(event) {
        if (event.type === "error" && typeof this.onerror === "function") {
          this.onerror(event);
        }
      }
    };

    attachImageFallback(image, ["/tmp/thumb.png", "/tmp/full.jpg", "/tmp/thumb.png"]);
    expect(image.getAttribute("src")).toBe("/tmp/thumb.png");

    image.dispatchEvent(new Event("error"));
    expect(image.getAttribute("src")).toBe("/tmp/full.jpg");

    image.dispatchEvent(new Event("error"));
    expect(image.getAttribute("src")).toBe("/tmp/full.jpg");
  });
});
