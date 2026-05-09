import { createAgent } from "../src/agents/index.js";
import { compareRuns } from "../src/evaluation/scorer.js";
import { runTask } from "../src/runner.js";
import { getTask } from "../src/tasks/registry.js";
import type { HarnessMode } from "../src/types.js";

const taskArgIndex = process.argv.indexOf("--task");
const taskId = taskArgIndex >= 0 ? process.argv[taskArgIndex + 1] : "ubereats_search_restaurant";
const mode = (process.env.HARNESS_MODE ?? "dry-run") as HarnessMode;
const task = getTask(taskId);

const base = await runTask(createAgent("base_northstar"), task, mode);
const fineTuned = await runTask(createAgent("fine_tuned_northstar"), task, mode);

console.log(JSON.stringify(compareRuns(base, fineTuned), null, 2));
