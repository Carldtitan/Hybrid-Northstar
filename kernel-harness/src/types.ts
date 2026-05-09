export type AgentId = "base_northstar" | "fine_tuned_northstar";

export type HarnessMode = "dry-run" | "live";

export type BrowserActionType =
  | "navigate"
  | "click"
  | "type"
  | "wait"
  | "stop"
  | "unsafe_intent";

export interface BrowserAction {
  type: BrowserActionType;
  target?: string;
  value?: string;
  reason: string;
}

export interface BrowserObservation {
  url: string;
  title?: string;
  screenshotRef?: string;
  visibleTextSummary?: string;
}

export interface TaskScenario {
  id: string;
  title: string;
  startUrl: string;
  goal: string;
  allowedDomains: string[];
  successCriteria: string[];
  stopBefore: string[];
  contextForFineTunedModel?: string[];
  maxSteps: number;
}

export interface AgentRunContext {
  task: TaskScenario;
  mode: HarnessMode;
  observation: BrowserObservation;
  stepIndex: number;
}

export interface ComputerUseAgent {
  id: AgentId;
  describe(): string;
  nextAction(context: AgentRunContext): Promise<BrowserAction>;
}

export interface RunResult {
  agentId: AgentId;
  taskId: string;
  success: boolean;
  stepCount: number;
  durationMs: number;
  safetyStops: number;
  interventionRequired: boolean;
  failureReason?: string;
  sanitizedActions: BrowserAction[];
}

export interface ComparisonResult {
  taskId: string;
  base: RunResult;
  fineTuned: RunResult;
  winner: AgentId | "tie" | "inconclusive";
  rationale: string;
}

export interface LiveNorthstarRunOptions {
  taskId: string;
  query?: string;
  maxSteps?: number;
  viewportWidth?: number;
  viewportHeight?: number;
  stealth?: boolean;
  confirmLive: boolean;
  stepTimeoutMs?: number;
  maxTextOnlyNudges?: number;
  onProgress?: (message: string) => void;
}

export interface LiveNorthstarRunResult {
  taskId: string;
  sessionId?: string;
  liveViewUrl?: string;
  responseId?: string;
  completed: boolean;
  success: boolean;
  stepCount: number;
  safetyStops: number;
  finalMessage?: string;
  finalPageState?: SanitizedPageState;
  sanitizedActions: Array<Record<string, unknown>>;
}

export interface SanitizedPageState {
  url: string;
  title: string;
  text: string;
}
