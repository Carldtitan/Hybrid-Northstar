import type { BrowserAction, BrowserObservation, HarnessMode, TaskScenario } from "../types.js";

export class KernelSessionManager {
  constructor(private readonly mode: HarnessMode) {}

  async startSession(task: TaskScenario): Promise<BrowserObservation> {
    if (this.mode === "dry-run") {
      return {
        url: task.startUrl,
        title: `Dry run: ${task.title}`,
        visibleTextSummary: "No live browser session was started."
      };
    }

    if (!process.env.KERNEL_API_KEY) {
      throw new Error("KERNEL_API_KEY must be configured for live Kernel runs.");
    }

    throw new Error("Live Kernel session startup is not implemented yet.");
  }

  async performAction(action: BrowserAction): Promise<BrowserObservation> {
    if (this.mode === "dry-run") {
      return {
        url: action.target ?? "about:blank",
        title: "Dry run observation",
        visibleTextSummary: `Dry-run action accepted: ${action.type}`
      };
    }

    throw new Error("Live Kernel action execution is not implemented yet.");
  }

  async stopSession(): Promise<void> {
    return;
  }
}
