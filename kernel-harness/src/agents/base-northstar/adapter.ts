import type { AgentRunContext, BrowserAction, ComputerUseAgent } from "../../types.js";

export class BaseNorthstarAgent implements ComputerUseAgent {
  id = "base_northstar" as const;

  describe(): string {
    return "Base Tzafon Northstar CUA Fast adapter placeholder.";
  }

  async nextAction(context: AgentRunContext): Promise<BrowserAction> {
    if (context.mode === "dry-run") {
      return {
        type: context.stepIndex === 0 ? "navigate" : "stop",
        target: context.stepIndex === 0 ? context.task.startUrl : undefined,
        reason:
          context.stepIndex === 0
            ? "Dry-run baseline navigation to the task start URL."
            : "Dry-run baseline stops before live browser actions."
      };
    }

    throw new Error("Live BaseNorthstarAgent integration is not implemented yet.");
  }
}
