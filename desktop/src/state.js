const SETUP_STEP_LABELS = {
  "sync-runtime": "App files",
  "check-runtime": "Python runtime",
  "prepare-dependencies": "Dependencies",
  "prepare-base-model": "Base model",
  "prepare-public-anchor": "Public anchor",
  "finish-setup": "Finalize"
};

const DEFAULT_TOTAL_STEPS = Object.keys(SETUP_STEP_LABELS).length;

function createSetupSteps() {
  return Object.entries(SETUP_STEP_LABELS).map(([task, label]) => ({
    task,
    label,
    status: "idle"
  }));
}

function hydrateSetupSteps(inputSteps) {
  const known = new Map((inputSteps ?? []).map((step) => [step.task, step]));
  return Object.entries(SETUP_STEP_LABELS).map(([task, label]) => {
    const snapshot = known.get(task);
    return {
      task,
      label,
      status: snapshot?.status ?? "idle"
    };
  });
}

function updateSetupSteps(steps, payload) {
  return steps.map((step) => {
    if (step.task !== payload.task) {
      return step;
    }

    if (payload.phase === "finish") {
      return { ...step, status: "done" };
    }

    return { ...step, status: "running" };
  });
}

function appendSetupLog(logs, payload) {
  const line = typeof payload === "string" ? payload : payload?.line;
  if (!line) {
    return logs;
  }
  if (logs.at(-1) === line) {
    return logs;
  }
  const nextLogs = [...logs, line];
  return nextLogs.slice(-16);
}

function mergeSetupState(currentSetup, payload) {
  return {
    currentStep: payload.currentStep ?? currentSetup.currentStep,
    totalSteps: payload.totalSteps ?? currentSetup.totalSteps,
    status: payload.status ?? currentSetup.status,
    message: payload.message ?? currentSetup.message,
    onboardingRequired:
      payload.onboardingRequired ?? payload.onboarding_required ?? currentSetup.onboardingRequired,
    steps: hydrateSetupSteps(payload.steps ?? currentSetup.steps),
    logs: payload.logs ?? currentSetup.logs
  };
}

export function createInitialState() {
  return {
    setup: {
      currentStep: 0,
      totalSteps: DEFAULT_TOTAL_STEPS,
      status: "idle",
      message: "Install the local runtime to continue.",
      onboardingRequired: true,
      steps: createSetupSteps(),
      logs: []
    },
    indexing: {
      status: "idle",
      phase: "idle",
      current: 0,
      total: 0,
      startedAtMs: null,
      elapsedSeconds: 0,
      remainingSeconds: null,
      embeddedCount: 0,
      reusedCount: 0,
      message: "No indexing task is running."
    },
    refresh: {
      isRunning: false,
      label: "Idle"
    },
    settingsOpen: false,
    activeFolder: null,
    activeEncoderSignature: "stage1",
    lastTaskMessage: "",
    results: []
  };
}

export function reduceAction(state, action) {
  if (action.type === "bootstrap-snapshot") {
    return {
      ...state,
      setup: mergeSetupState(state.setup, action.payload ?? {})
    };
  }

  if (action.type === "setup-progress") {
    return {
      ...state,
      setup: {
        ...state.setup,
        currentStep: action.payload.current ?? action.payload.currentStep ?? state.setup.currentStep,
        totalSteps: action.payload.total ?? action.payload.totalSteps ?? state.setup.totalSteps,
        message: action.payload.message ?? state.setup.message,
        status: "running",
        steps: action.payload.task ? updateSetupSteps(state.setup.steps, action.payload) : state.setup.steps
      }
    };
  }

  if (action.type === "setup-log") {
    return {
      ...state,
      setup: {
        ...state.setup,
        logs: appendSetupLog(state.setup.logs, action.payload)
      }
    };
  }

  if (action.type === "setup-finished") {
    return {
      ...state,
      setup: {
        ...state.setup,
        status: "ready",
        currentStep: state.setup.totalSteps,
        message: "Desktop runtime is ready.",
        steps: state.setup.steps.map((step) => ({ ...step, status: "done" }))
      }
    };
  }

  if (action.type === "setup-failed") {
    return {
      ...state,
      setup: {
        ...state.setup,
        status: "failed",
        message: action.payload ?? state.setup.message
      },
      lastTaskMessage: action.payload ?? state.lastTaskMessage
    };
  }

  if (action.type === "refresh-started") {
    return {
      ...state,
      refresh: {
        isRunning: true,
        label: "Refreshing"
      }
    };
  }

  if (action.type === "refresh-finished") {
    return {
      ...state,
      refresh: {
        isRunning: false,
        label: action.payload?.label ?? "Idle"
      }
    };
  }

  if (action.type === "settings-opened") {
    return { ...state, settingsOpen: true };
  }

  if (action.type === "settings-closed") {
    return { ...state, settingsOpen: false };
  }

  if (action.type === "runtime-status") {
    return {
      ...state,
      activeFolder: action.payload.activeFolder,
      activeEncoderSignature: action.payload.activeEncoderSignature,
      lastTaskMessage: action.payload.lastTaskMessage ?? state.lastTaskMessage,
      indexing: action.payload.indexing ?? state.indexing
    };
  }

  if (action.type === "folder-selected") {
    return {
      ...state,
      activeFolder: action.payload.activeFolder,
      activeEncoderSignature: action.payload.activeEncoderSignature ?? state.activeEncoderSignature,
      lastTaskMessage: action.payload.lastTaskMessage ?? state.lastTaskMessage,
      indexing: action.payload.indexing ?? state.indexing,
      results: []
    };
  }

  if (action.type === "index-progress") {
    return {
      ...state,
      indexing: {
        ...state.indexing,
        ...(action.payload ?? {})
      },
      lastTaskMessage: action.payload?.message ?? state.lastTaskMessage
    };
  }

  if (action.type === "results-received") {
    return {
      ...state,
      results: action.payload.results ?? []
    };
  }

  if (action.type === "task-message") {
    return {
      ...state,
      lastTaskMessage: action.payload
    };
  }

  return state;
}
