import type { BrowserAction, TaskScenario } from "../types.js";

const sensitivePatterns = [
  /api[_-]?key/i,
  /authorization:\s*bearer/i,
  /password/i,
  /session/i,
  /cookie/i,
  /oauth/i,
  /token/i,
  /card number/i
];

export function isSensitiveText(value: string): boolean {
  return sensitivePatterns.some((pattern) => pattern.test(value));
}

export function assertTaskIsSafe(task: TaskScenario): void {
  if (task.stopBefore.length === 0) {
    throw new Error(`Task ${task.id} must declare stopBefore safety boundaries.`);
  }
  if (task.maxSteps < 1 || task.maxSteps > 100) {
    throw new Error(`Task ${task.id} must use a bounded maxSteps value between 1 and 100.`);
  }
}

export function isActionBlocked(task: TaskScenario, action: BrowserAction): boolean {
  const text = [action.target, action.value, action.reason].filter(Boolean).join(" ").toLowerCase();
  return task.stopBefore.some((stopPhrase) => text.includes(stopPhrase.toLowerCase()));
}

export function sanitizeAction(action: BrowserAction): BrowserAction {
  const sanitize = (value: string | undefined): string | undefined => {
    if (!value) return value;
    return isSensitiveText(value) ? "<redacted>" : value;
  };

  return {
    ...action,
    target: sanitize(action.target),
    value: sanitize(action.value),
    reason: sanitize(action.reason) ?? ""
  };
}
