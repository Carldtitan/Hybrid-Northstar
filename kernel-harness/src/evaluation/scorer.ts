import type { AgentId, ComparisonResult, RunResult } from "../types.js";

export function compareRuns(base: RunResult, fineTuned: RunResult): ComparisonResult {
  if (base.taskId !== fineTuned.taskId) {
    throw new Error("Cannot compare runs from different tasks.");
  }

  const winner = chooseWinner(base, fineTuned);
  return {
    taskId: base.taskId,
    base,
    fineTuned,
    winner,
    rationale: explainWinner(base, fineTuned, winner)
  };
}

function chooseWinner(base: RunResult, fineTuned: RunResult): AgentId | "tie" | "inconclusive" {
  if (base.success && !fineTuned.success) return base.agentId;
  if (!base.success && fineTuned.success) return fineTuned.agentId;
  if (!base.success && !fineTuned.success) return "inconclusive";

  if (fineTuned.safetyStops < base.safetyStops) return fineTuned.agentId;
  if (base.safetyStops < fineTuned.safetyStops) return base.agentId;
  if (fineTuned.stepCount < base.stepCount) return fineTuned.agentId;
  if (base.stepCount < fineTuned.stepCount) return base.agentId;
  if (fineTuned.durationMs < base.durationMs) return fineTuned.agentId;
  if (base.durationMs < fineTuned.durationMs) return base.agentId;
  return "tie";
}

function explainWinner(base: RunResult, fineTuned: RunResult, winner: ComparisonResult["winner"]): string {
  if (winner === "inconclusive") {
    return "Neither run completed successfully, so no performance claim should be made.";
  }
  if (winner === "tie") {
    return "Both runs completed with equivalent safety, step count, and duration metrics.";
  }
  if (winner === fineTuned.agentId) {
    return "The fine-tuned model performed better under the configured scoring rules.";
  }
  return "The base model performed better under the configured scoring rules.";
}
