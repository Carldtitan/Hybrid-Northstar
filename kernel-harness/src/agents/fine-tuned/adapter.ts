import type { AgentRunContext, BrowserAction, ComputerUseAgent } from "../../types.js";

export class FineTunedNorthstarAgent implements ComputerUseAgent {
  id = "fine_tuned_northstar" as const;

  describe(): string {
    return "Future fine-tuned model adapter placeholder.";
  }

  async nextAction(context: AgentRunContext): Promise<BrowserAction> {
    if (context.mode === "dry-run") {
      return {
        type: context.stepIndex === 0 ? "navigate" : "stop",
        target: context.stepIndex === 0 ? context.task.startUrl : undefined,
        reason:
          context.stepIndex === 0
            ? "Dry-run fine-tuned navigation using task context."
            : "Dry-run fine-tuned adapter stops before live browser actions."
      };
    }

    if (!process.env.FINE_TUNED_MODEL_ENDPOINT) {
      throw new Error("FINE_TUNED_MODEL_ENDPOINT must be configured for live fine-tuned runs.");
    }

    throw new Error("Live FineTunedNorthstarAgent integration is not implemented yet.");
  }
}
