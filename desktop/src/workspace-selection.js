function visibleResultPaths(results) {
  return results.map((item) => item.relativePath).filter(Boolean);
}

export function pruneSelectionToVisibleResults(results, selectedPaths) {
  const visiblePaths = new Set(visibleResultPaths(results));
  return new Set([...selectedPaths].filter((path) => visiblePaths.has(path)));
}

export function countVisibleSelection(results, selectedPaths) {
  return pruneSelectionToVisibleResults(results, selectedPaths).size;
}

export function toggleResultSelection(selectedPaths, relativePath) {
  const nextSelection = new Set(selectedPaths);
  if (!relativePath) {
    return nextSelection;
  }
  if (nextSelection.has(relativePath)) {
    nextSelection.delete(relativePath);
  } else {
    nextSelection.add(relativePath);
  }
  return nextSelection;
}

export function toggleAllVisibleResults(results, selectedPaths) {
  const visiblePaths = visibleResultPaths(results);
  const nextSelection = new Set(selectedPaths);
  const allVisibleSelected = visiblePaths.length > 0 && visiblePaths.every((path) => nextSelection.has(path));

  for (const path of visiblePaths) {
    if (allVisibleSelected) {
      nextSelection.delete(path);
    } else {
      nextSelection.add(path);
    }
  }

  return nextSelection;
}
