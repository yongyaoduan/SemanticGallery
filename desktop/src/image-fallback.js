export function buildImageSourceCandidates(...sources) {
  const unique = [];
  const seen = new Set();
  for (const source of sources) {
    if (typeof source !== "string") {
      continue;
    }
    const normalized = source.trim();
    if (!normalized || seen.has(normalized)) {
      continue;
    }
    seen.add(normalized);
    unique.push(normalized);
  }
  return unique;
}

export function attachImageFallback(image, sources) {
  const candidates = Array.isArray(sources)
    ? buildImageSourceCandidates(...sources)
    : buildImageSourceCandidates(sources);
  let activeIndex = 0;

  image.onerror = () => {
    if (activeIndex + 1 >= candidates.length) {
      return;
    }
    activeIndex += 1;
    image.src = candidates[activeIndex];
  };

  if (!candidates.length) {
    image.removeAttribute("src");
    return [];
  }

  image.src = candidates[0];
  return candidates;
}
