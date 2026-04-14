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

const STAGE2_STEPS = [
  { key: "prepare", label: "Private data", weight: 15 },
  { key: "adapt", label: "Model training", weight: 55 },
  { key: "validate", label: "Validation", weight: 10 },
  { key: "reindex", label: "Rebuild index", weight: 15 },
  { key: "finalize", label: "Finalize", weight: 5 }
];

function clampPercent(value) {
  return Math.min(100, Math.max(0, Math.round(value)));
}

function normalizeRatio(current, total) {
  const safeCurrent = Number(current);
  const safeTotal = Number(total);
  if (!Number.isFinite(safeCurrent) || !Number.isFinite(safeTotal) || safeTotal <= 0) {
    return 0;
  }
  return Math.min(Math.max(safeCurrent / safeTotal, 0), 1);
}

function activeStage2Step(stage2 = {}) {
  const phase = STAGE2_STEPS.find((step) => step.key === stage2.phase);
  return phase?.key ?? null;
}

function stage2StepRatio(stage2 = {}) {
  const ratio = normalizeRatio(
    stage2.phaseCurrent ?? stage2.current,
    stage2.phaseTotal ?? stage2.total
  );
  if (stage2.status === "running" && stage2.phase === "finalize" && ratio <= 0) {
    return 0.64;
  }
  return ratio;
}

function buildStage2Steps(stage2 = {}) {
  const activeKey = activeStage2Step(stage2);
  const activeIndex = STAGE2_STEPS.findIndex((step) => step.key === activeKey);
  const activeRatio = stage2StepRatio(stage2);

  if (stage2.status === "ready") {
    return STAGE2_STEPS.map((step) => ({
      key: step.key,
      label: step.label,
      status: "done",
      percent: 100
    }));
  }

  if (activeIndex < 0) {
    return STAGE2_STEPS.map((step) => ({
      key: step.key,
      label: step.label,
      status: "waiting",
      percent: 0
    }));
  }

  return STAGE2_STEPS.map((step, index) => {
    if (index < activeIndex || (index === activeIndex && activeRatio >= 1)) {
      return { key: step.key, label: step.label, status: "done", percent: 100 };
    }
    if (index > activeIndex) {
      return { key: step.key, label: step.label, status: "waiting", percent: 0 };
    }
    return {
      key: step.key,
      label: step.label,
      status: stage2.status === "failed" ? "failed" : "running",
      percent: clampPercent(activeRatio * 100)
    };
  });
}

function buildStage2OverallPercent(stage2 = {}) {
  if (stage2.status === "ready") {
    return 100;
  }
  const activeKey = activeStage2Step(stage2);
  if (!activeKey) {
    return 0;
  }
  const activeRatio = stage2StepRatio(stage2);
  let total = 0;
  for (const step of STAGE2_STEPS) {
    if (step.key === activeKey) {
      total += step.weight * activeRatio;
      break;
    }
    total += step.weight;
  }
  return clampPercent(total);
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
  const showProgress = indexing.status === "running";
  const indeterminate = showProgress && indexing.phase === "scan" && safeTotal === 0;
  const countLabel = indeterminate
    ? current > 0
      ? `${current} files checked`
      : "Scanning..."
    : `${cappedCurrent} / ${safeTotal}`;

  return {
    countLabel,
    percent,
    showProgress,
    indeterminate,
    ...buildIndexProgressTiming(indexing)
  };
}

export function buildStage2ProgressSummary(stage2 = {}) {
  const percent = buildStage2OverallPercent(stage2);
  const showProgress = stage2.status === "running";
  const showSteps = showProgress;

  return {
    countLabel: `${percent}%`,
    percent,
    detailLabel: stage2.message ?? "",
    showProgress,
    showSteps,
    steps: buildStage2Steps(stage2),
    ...buildIndexProgressTiming(stage2)
  };
}
