function normalizeCount(value) {
  const count = Number(value);
  if (!Number.isFinite(count) || count < 0) {
    return 0;
  }
  return Math.floor(count);
}

function normalizeTimestamp(value) {
  const timestamp = Number(value);
  if (!Number.isFinite(timestamp) || timestamp <= 0) {
    return 0;
  }
  return Math.floor(timestamp);
}

function formatDuration(totalSeconds) {
  const safeSeconds = Math.max(0, Math.round(Number(totalSeconds) || 0));
  const hours = Math.floor(safeSeconds / 3600);
  const minutes = Math.floor((safeSeconds % 3600) / 60);
  const seconds = safeSeconds % 60;

  if (hours > 0) {
    return `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
  }
  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}

function formatFinishTime(timestampMs) {
  return new Intl.DateTimeFormat(undefined, {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).format(new Date(timestampMs));
}

export function buildIndexProgressTiming(indexing = {}) {
  const startedAtMs = normalizeTimestamp(indexing.startedAtMs);
  const elapsedSeconds = normalizeCount(indexing.elapsedSeconds);
  const remainingSeconds = normalizeCount(indexing.remainingSeconds);
  const elapsedLabel = elapsedSeconds > 0 ? `Elapsed ${formatDuration(elapsedSeconds)}` : "";
  const remainingLabel =
    indexing.status === "running" && remainingSeconds > 0 ? `Left ${formatDuration(remainingSeconds)}` : "";
  const etaLabel =
    indexing.status === "running" && startedAtMs > 0 && remainingSeconds > 0
      ? `Done by ${formatFinishTime(startedAtMs + (elapsedSeconds + remainingSeconds) * 1000)}`
      : "";

  return {
    elapsedLabel,
    remainingLabel,
    etaLabel
  };
}

export function buildIndexProgressSummary(indexing = {}) {
  const current = normalizeCount(indexing.current);
  const total = normalizeCount(indexing.total);
  const safeTotal = total > 0 ? total : 0;
  const cappedCurrent = safeTotal > 0 ? Math.min(current, safeTotal) : current;
  const percent = safeTotal > 0 ? Math.round((cappedCurrent / safeTotal) * 100) : 0;

  return {
    countLabel: `${cappedCurrent} / ${safeTotal}`,
    percent,
    ...buildIndexProgressTiming(indexing)
  };
}

export function buildStage2ProgressSummary(stage2 = {}) {
  const current = normalizeCount(stage2.current);
  const total = normalizeCount(stage2.total);
  const safeTotal = total > 0 ? total : 0;
  const cappedCurrent = safeTotal > 0 ? Math.min(current, safeTotal) : current;
  const percent = safeTotal > 0 ? Math.round((cappedCurrent / safeTotal) * 100) : 0;

  return {
    countLabel: `${percent}%`,
    percent,
    detailLabel: stage2.message ?? "",
    ...buildIndexProgressTiming(stage2)
  };
}
