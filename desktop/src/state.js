const SETUP_STEP_LABELS = {
  "check-runtime": "Python runtime",
  "prepare-dependencies": "Dependencies",
  "prepare-base-model": "Base model",
  "prepare-public-anchor": "Public anchor",
  "finish-setup": "Finalize"
};

function createSetupSteps() {
  return Object.entries(SETUP_STEP_LABELS).map(([task, label]) => ({
    task,
    label,
    status: "idle"
  }));
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
  const nextLogs = [...logs, line];
  return nextLogs.slice(-10);
}

export function createInitialState() {
  return {
    setup: {
      currentStep: 0,
      totalSteps: 5,
      status: "idle",
      message: "Checking local resources and preparing the model runtime.",
      steps: createSetupSteps(),
      logs: []
    },
    refresh: {
      isRunning: false,
      label: "Idle"
    },
    settingsOpen: false,
    activeFolder: null,
    activeEncoderSignature: "stage1",
    lastTaskMessage: "No background task has started yet.",
    results: []
  };
}

export function reduceAction(state, action) {
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
      lastTaskMessage: action.payload.lastTaskMessage ?? state.lastTaskMessage
    };
  }

  if (action.type === "folder-selected") {
    return {
      ...state,
      activeFolder: action.payload.activeFolder,
      activeEncoderSignature: action.payload.activeEncoderSignature ?? state.activeEncoderSignature,
      lastTaskMessage: action.payload.lastTaskMessage ?? state.lastTaskMessage
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
