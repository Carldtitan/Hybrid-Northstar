import { createAgent } from "../src/agents/index.js";
import { runTask } from "../src/runner.js";
import { getTask } from "../src/tasks/registry.js";
import type { AgentId, HarnessMode } from "../src/types.js";

const args = new Map<string, string>();
for (let i = 2; i < process.argv.length; i += 2) {
  args.set(process.argv[i], process.argv[i + 1]);
}

const agentId = (args.get("--agent") ?? "base_northstar") as AgentId;
const taskId = args.get("--task") ?? "ubereats_search_restaurant";
const mode = (process.env.HARNESS_MODE ?? "dry-run") as HarnessMode;

const result = await runTask(createAgent(agentId), getTask(taskId), mode);
console.log(JSON.stringify(result, null, 2));
