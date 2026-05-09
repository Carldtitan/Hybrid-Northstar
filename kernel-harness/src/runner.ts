import { KernelSessionManager } from "./kernel/session-manager.js";
import { assertTaskIsSafe, isActionBlocked, sanitizeAction } from "./safety/guardrails.js";
import type { ComputerUseAgent, HarnessMode, RunResult, TaskScenario } from "./types.js";

export async function runTask(agent: ComputerUseAgent, task: TaskScenario, mode: HarnessMode): Promise<RunResult> {
  assertTaskIsSafe(task);

  const startedAt = Date.now();
  const kernel = new KernelSessionManager(mode);
  const sanitizedActions = [];
  let safetyStops = 0;
  let observation = await kernel.startSession(task);

  try {
    for (let stepIndex = 0; stepIndex < task.maxSteps; stepIndex += 1) {
      const action = await agent.nextAction({ task, mode, observation, stepIndex });
      const sanitized = sanitizeAction(action);
      sanitizedActions.push(sanitized);

      if (isActionBlocked(task, action) || action.type === "unsafe_intent") {
        safetyStops += 1;
        return {
          agentId: agent.id,
          taskId: task.id,
          success: false,
          stepCount: stepIndex + 1,
          durationMs: Date.now() - startedAt,
          safetyStops,
          interventionRequired: false,
          failureReason: "Stopped by safety guardrail.",
          sanitizedActions
        };
      }

      if (action.type === "stop") {
        return {
          agentId: agent.id,
          taskId: task.id,
          success: mode === "dry-run",
          stepCount: stepIndex + 1,
          durationMs: Date.now() - startedAt,
          safetyStops,
          interventionRequired: false,
          sanitizedActions
        };
      }

      observation = await kernel.performAction(action);
    }

    return {
      agentId: agent.id,
      taskId: task.id,
      success: false,
      stepCount: task.maxSteps,
      durationMs: Date.now() - startedAt,
      safetyStops,
      interventionRequired: true,
      failureReason: "Task exceeded maxSteps.",
      sanitizedActions
    };
  } finally {
    await kernel.stopSession();
  }
}
