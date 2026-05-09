import "dotenv/config";
import { runLiveNorthstarOnKernel } from "../src/live/northstar-kernel-runner.js";

const args = new Map<string, string | boolean>();
for (let i = 2; i < process.argv.length; i += 1) {
  const arg = process.argv[i];
  if (!arg.startsWith("--")) continue;
  const next = process.argv[i + 1];
  if (!next || next.startsWith("--")) {
    args.set(arg, true);
  } else {
    args.set(arg, next);
    i += 1;
  }
}

const result = await runLiveNorthstarOnKernel({
  taskId: String(args.get("--task") ?? "ubereats_search_restaurant"),
  query: typeof args.get("--query") === "string" ? String(args.get("--query")) : undefined,
  maxSteps: typeof args.get("--max-steps") === "string" ? Number(args.get("--max-steps")) : undefined,
  confirmLive: args.get("--confirm-live") === true,
  stepTimeoutMs: typeof args.get("--step-timeout-ms") === "string" ? Number(args.get("--step-timeout-ms")) : undefined,
  maxTextOnlyNudges:
    typeof args.get("--max-text-only-nudges") === "string" ? Number(args.get("--max-text-only-nudges")) : undefined,
  onProgress: (message) => {
    console.error(`[live] ${message}`);
  }
});

console.log(JSON.stringify(result, null, 2));
