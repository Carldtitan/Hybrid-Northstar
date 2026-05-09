import type { AgentId, ComputerUseAgent } from "../types.js";
import { BaseNorthstarAgent } from "./base-northstar/adapter.js";
import { FineTunedNorthstarAgent } from "./fine-tuned/adapter.js";

export function createAgent(agentId: AgentId): ComputerUseAgent {
  if (agentId === "base_northstar") return new BaseNorthstarAgent();
  if (agentId === "fine_tuned_northstar") return new FineTunedNorthstarAgent();
  throw new Error(`Unknown agent: ${agentId satisfies never}`);
}
