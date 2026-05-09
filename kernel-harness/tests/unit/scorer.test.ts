import assert from "node:assert/strict";
import test from "node:test";
import { compareRuns } from "../../src/evaluation/scorer.js";
import type { RunResult } from "../../src/types.js";

const baseRun: RunResult = {
  agentId: "base_northstar",
  taskId: "ubereats_search_restaurant",
  success: true,
  stepCount: 8,
  durationMs: 1000,
  safetyStops: 0,
  interventionRequired: false,
  sanitizedActions: []
};

test("chooses fine-tuned model when both succeed and it uses fewer steps", () => {
  const comparison = compareRuns(baseRun, {
    ...baseRun,
    agentId: "fine_tuned_northstar",
    stepCount: 5
  });

  assert.equal(comparison.winner, "fine_tuned_northstar");
});

test("does not make a winner claim when neither run succeeds", () => {
  const comparison = compareRuns(
    { ...baseRun, success: false },
    { ...baseRun, agentId: "fine_tuned_northstar", success: false }
  );

  assert.equal(comparison.winner, "inconclusive");
});
