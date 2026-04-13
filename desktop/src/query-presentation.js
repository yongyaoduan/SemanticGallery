export function buildQueryPresentation(query) {
  if (!query || query.kind === "none") {
    return {
      inputValue: "",
      chip: null
    };
  }

  if (query.kind === "text") {
    return {
      inputValue: query.text ?? "",
      chip: null
    };
  }

  return {
    inputValue: "",
    chip: {
      label: query.fileName || "Image query",
      previewUrl: query.previewUrl || query.url || ""
    }
  };
}
