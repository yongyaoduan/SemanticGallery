import { describe, expect, it } from "vitest";

import { buildQueryPresentation } from "./query-presentation";

describe("query presentation", () => {
  it("keeps plain text searches in the input field", () => {
    expect(
      buildQueryPresentation({
        kind: "text",
        text: "food",
        previewUrl: "",
        fileName: "",
        relativePath: "",
        url: "",
        file: null
      })
    ).toEqual({
      inputValue: "food",
      chip: null
    });
  });

  it("renders an image chip for uploaded image search", () => {
    expect(
      buildQueryPresentation({
        kind: "upload",
        text: "",
        previewUrl: "blob:semanticgallery-test",
        fileName: "Clipboard image",
        relativePath: "",
        url: "",
        file: null
      })
    ).toEqual({
      inputValue: "",
      chip: {
        label: "Clipboard image",
        previewUrl: "blob:semanticgallery-test"
      }
    });
  });

  it("renders an image chip for similar-image search", () => {
    expect(
      buildQueryPresentation({
        kind: "similar",
        text: "",
        previewUrl: "http://127.0.0.1:38291/thumbs/cat.jpg",
        fileName: "cat.jpg",
        relativePath: "cat.jpg",
        url: "http://127.0.0.1:38291/api/similar/cat.jpg",
        file: null
      })
    ).toEqual({
      inputValue: "",
      chip: {
        label: "cat.jpg",
        previewUrl: "http://127.0.0.1:38291/thumbs/cat.jpg"
      }
    });
  });
});
