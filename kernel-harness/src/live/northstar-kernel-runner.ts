import Kernel from "@onkernel/sdk";
import Lightcone from "@tzafon/lightcone";
import { getTask } from "../tasks/registry.js";
import type { LiveNorthstarRunOptions, LiveNorthstarRunResult, SanitizedPageState, TaskScenario } from "../types.js";

type NorthstarAction = {
  type?: string;
  x?: number;
  y?: number;
  end_x?: number;
  end_y?: number;
  path?: Array<{ x?: number; y?: number }>;
  button?: string;
  text?: string;
  keys?: string[];
  scroll_x?: number;
  scroll_y?: number;
  url?: string;
  result?: string;
};

type ComputerCall = {
  type?: string;
  call_id?: string;
  action?: NorthstarAction;
};

type NorthstarResponse = {
  id?: string;
  output?: Array<ComputerCall | Record<string, unknown>>;
};

const DEFAULT_VIEWPORT = { width: 1280, height: 800 };
const PNG_DIMENSIONS_OFFSET = 16;

export async function runLiveNorthstarOnKernel(
  options: LiveNorthstarRunOptions
): Promise<LiveNorthstarRunResult> {
  if (!options.confirmLive) {
    throw new Error("Live Kernel/Northstar runs require --confirm-live.");
  }

  normalizeTzafonEnv();
  assertRequiredEnv();

  const task = getTask(options.taskId);
  const viewportWidth = options.viewportWidth ?? DEFAULT_VIEWPORT.width;
  const viewportHeight = options.viewportHeight ?? DEFAULT_VIEWPORT.height;
  const maxSteps = Math.min(options.maxSteps ?? task.maxSteps, task.maxSteps);
  const stepTimeoutMs = options.stepTimeoutMs ?? 45_000;
  const maxTextOnlyNudges = options.maxTextOnlyNudges ?? 3;
  const progress = options.onProgress ?? (() => undefined);
  const kernel = new Kernel({ apiKey: process.env.KERNEL_API_KEY });
  const tzafon = new Lightcone({ apiKey: process.env.TZAFON_API_KEY });
  const sanitizedActions: Array<Record<string, unknown>> = [];
  let sessionId: string | undefined;
  let liveViewUrl: string | undefined;
  let response: NorthstarResponse | undefined;
  let pageState: SanitizedPageState | undefined;
  let safetyStops = 0;
  let screenshotWidth = viewportWidth;
  let screenshotHeight = viewportHeight;
  let textOnlyNudges = 0;

  try {
    progress("creating Kernel browser session");
    const session = await withTimeout(
      kernel.browsers.create({
        stealthMode: options.stealth ?? true,
        viewport: { width: viewportWidth, height: viewportHeight }
      } as never),
      stepTimeoutMs,
      "Kernel browser creation timed out"
    );

    sessionId = pickSessionId(session);
    liveViewUrl = pickLiveViewUrl(session);
    progress(`created Kernel browser session ${sessionId}`);
    if (liveViewUrl) progress(`live view: ${liveViewUrl}`);

    progress(`navigating to ${task.startUrl}`);
    await withTimeout(navigate(kernel, sessionId, task.startUrl), stepTimeoutMs, "Kernel navigation timed out");
    pageState = await capturePageState(kernel, sessionId, progress);

    progress("capturing initial screenshot");
    const firstScreenshot = await captureScreenshot(kernel, sessionId);
    screenshotWidth = firstScreenshot.width;
    screenshotHeight = firstScreenshot.height;
    progress(`screenshot dimensions: ${screenshotWidth}x${screenshotHeight}`);
    progress("requesting first Northstar action");
    response = (await withTimeout(
      tzafon.responses.create({
        model: "tzafon.northstar-cua-fast",
        input: [
          {
            role: "user",
            content: [
              { type: "input_text", text: buildInstruction(task, options.query) },
              {
                type: "input_image",
                image_url: `data:image/png;base64,${firstScreenshot.base64}`,
                detail: "auto"
              }
            ]
          }
        ],
        tools: [
          {
            type: "computer_use_preview",
            display_width: screenshotWidth,
            display_height: screenshotHeight,
            environment: "browser"
          }
        ]
      } as never),
      stepTimeoutMs,
      "Northstar initial response timed out"
    )) as NorthstarResponse;

    for (let step = 0; step < maxSteps; step += 1) {
      const computerCall = findComputerCall(response);
      if (!computerCall?.action) {
        const success = evaluateTaskSuccess(task, pageState);
        const modelText = extractModelText(response);
        const textAnswerSuccess = evaluateTextAnswerSuccess(task, modelText);
        if (textAnswerSuccess) {
          return {
            taskId: task.id,
            sessionId,
            liveViewUrl,
            responseId: response?.id,
            completed: true,
            success: true,
            stepCount: step,
            safetyStops,
            finalMessage: `Task answer returned by model. ${modelText}`,
            finalPageState: pageState,
            sanitizedActions
          };
        }
        if (!success && modelText && textOnlyNudges < maxTextOnlyNudges) {
          textOnlyNudges += 1;
          progress(`text-only model response; nudging computer tool use (${textOnlyNudges}/${maxTextOnlyNudges})`);
          try {
            response = (await withTimeout(
              tzafon.responses.create({
                model: "tzafon.northstar-cua-fast",
                previous_response_id: response?.id,
                input: [
                  {
                    role: "user",
                    content: [
                      {
                        type: "input_text",
                        text:
                          "Continue by performing the next computer action with the computer tool. Do not describe the action in text unless the task is complete or blocked."
                      }
                    ]
                  }
                ],
                tools: [
                  {
                    type: "computer_use_preview",
                    display_width: screenshotWidth,
                    display_height: screenshotHeight,
                    environment: "browser"
                  }
                ]
              } as never),
              stepTimeoutMs,
              "Northstar text-only nudge response timed out"
            )) as NorthstarResponse;
            step -= 1;
            continue;
          } catch (error) {
            progress(`text-only nudge failed: ${error instanceof Error ? error.message : "unknown error"}`);
          }
        }
        return {
          taskId: task.id,
          sessionId,
          liveViewUrl,
          responseId: response?.id,
          completed: true,
          success,
          stepCount: step,
          safetyStops,
          finalMessage: success
            ? "Task success criteria matched after Northstar stopped."
            : `Northstar returned no further computer action before success criteria matched.${modelText ? ` Model text: ${modelText}` : ""}`,
          finalPageState: pageState,
          sanitizedActions
        };
      }

      const action = computerCall.action;
      progress(`step ${step + 1}: ${action.type ?? "unknown"} action`);
      sanitizedActions.push(sanitizeNorthstarAction(action));

      if (hasRepeatedActionLoop(sanitizedActions)) {
        return {
          taskId: task.id,
          sessionId,
          liveViewUrl,
          responseId: response?.id,
          completed: false,
          success: false,
          stepCount: step + 1,
          safetyStops,
          finalMessage: "Stopped after detecting a repeated-action loop.",
          finalPageState: pageState,
          sanitizedActions
        };
      }

      if (isTerminalAction(action)) {
        const success = evaluateTaskSuccess(task, pageState);
        return {
          taskId: task.id,
          sessionId,
          liveViewUrl,
          responseId: response?.id,
          completed: true,
          success,
          stepCount: step + 1,
          safetyStops,
          finalMessage: success
            ? (action.result ?? "Task success criteria matched.")
            : (action.result ?? "Northstar completed without matching task success criteria."),
          finalPageState: pageState,
          sanitizedActions
        };
      }

      if (isUnsafeAction(task, action)) {
        safetyStops += 1;
        return {
          taskId: task.id,
          sessionId,
          liveViewUrl,
          responseId: response?.id,
          completed: false,
          success: false,
          stepCount: step + 1,
          safetyStops,
          finalMessage: "Stopped before an unsafe commerce or irreversible action.",
          finalPageState: pageState,
          sanitizedActions
        };
      }

      const executionDescription = describeActionExecution(action, screenshotWidth, screenshotHeight);
      if (executionDescription) progress(executionDescription);
      await withTimeout(
        executeNorthstarAction(kernel, sessionId, action, screenshotWidth, screenshotHeight),
        stepTimeoutMs,
        `Kernel action timed out: ${action.type ?? "unknown"}`
      );
      await sleep(1000);
      pageState = await capturePageState(kernel, sessionId, progress);
      if (evaluateTaskSuccess(task, pageState)) {
        return {
          taskId: task.id,
          sessionId,
          liveViewUrl,
          responseId: response?.id,
          completed: true,
          success: true,
          stepCount: step + 1,
          safetyStops,
          finalMessage: "Task success criteria matched.",
          finalPageState: pageState,
          sanitizedActions
        };
      }

      progress(`step ${step + 1}: capturing screenshot`);
      const screenshot = await captureScreenshot(kernel, sessionId);
      screenshotWidth = screenshot.width;
      screenshotHeight = screenshot.height;
      progress(`screenshot dimensions: ${screenshotWidth}x${screenshotHeight}`);
      progress(`step ${step + 1}: requesting next Northstar action`);
      response = (await withTimeout(
        tzafon.responses.create({
          model: "tzafon.northstar-cua-fast",
          previous_response_id: response?.id,
          input: [
            {
              type: "computer_call_output",
              call_id: computerCall.call_id,
              output: {
                type: "computer_screenshot",
                image_url: `data:image/png;base64,${screenshot.base64}`,
                detail: "auto"
              }
            }
          ],
          tools: [
            {
              type: "computer_use_preview",
              display_width: screenshotWidth,
              display_height: screenshotHeight,
              environment: "browser"
            }
          ]
        } as never),
        stepTimeoutMs,
        "Northstar follow-up response timed out"
      )) as NorthstarResponse;
    }

    return {
      taskId: task.id,
      sessionId,
      liveViewUrl,
      responseId: response?.id,
      completed: false,
      success: false,
      stepCount: maxSteps,
      safetyStops,
      finalMessage: "Stopped after reaching maxSteps.",
      finalPageState: pageState,
      sanitizedActions
    };
  } finally {
    if (sessionId) {
      progress(`deleting Kernel browser session ${sessionId}`);
      await kernel.browsers.deleteByID(sessionId).catch(() => undefined);
    }
  }
}

function normalizeTzafonEnv(): void {
  if (!process.env.TZAFON_API_KEY && process.env.LIGHTCONE_API_KEY) {
    process.env.TZAFON_API_KEY = process.env.LIGHTCONE_API_KEY;
  }
}

function assertRequiredEnv(): void {
  if (!process.env.KERNEL_API_KEY) {
    throw new Error("KERNEL_API_KEY is required for live runs.");
  }
  if (!process.env.TZAFON_API_KEY) {
    throw new Error("TZAFON_API_KEY or LIGHTCONE_API_KEY is required for live runs.");
  }
}

function buildInstruction(task: TaskScenario, query?: string): string {
  const requestedTask = query ?? task.goal;
  return [
    requestedTask,
    "",
    "Safety boundaries:",
    "- Do not sign in, enter credentials, or expose private account information.",
    "- Do not click checkout, place order, submit order, payment, subscribe, send, or confirm purchase.",
    "- If a location, login, payment, or account wall appears, stop and report the blocker.",
    "- Stop after reaching the visible success criteria.",
    "",
    `Allowed domains: ${task.allowedDomains.join(", ")}`,
    `Success criteria: ${task.successCriteria.join(" ")}`,
    `Must stop before: ${task.stopBefore.join(", ")}`
  ].join("\n");
}

function findComputerCall(response: NorthstarResponse | undefined): ComputerCall | undefined {
  return response?.output?.find((item): item is ComputerCall => item.type === "computer_call");
}

function isTerminalAction(action: NorthstarAction): boolean {
  return ["terminate", "done", "answer"].includes(action.type ?? "");
}

function isUnsafeAction(task: TaskScenario, action: NorthstarAction): boolean {
  const text = JSON.stringify(action).toLowerCase();
  return task.stopBefore.some((phrase) => text.includes(phrase.toLowerCase()));
}

function sanitizeNorthstarAction(action: NorthstarAction): Record<string, unknown> {
  const sanitized: Record<string, unknown> = { type: action.type };
  for (const key of ["x", "y", "end_x", "end_y", "button", "scroll_x", "scroll_y", "url", "result"] as const) {
    if (action[key] !== undefined) sanitized[key] = action[key];
  }
  if (action.text !== undefined) sanitized.text = "<redacted typed text>";
  if (action.keys !== undefined) sanitized.keys = action.keys;
  return sanitized;
}

function hasRepeatedActionLoop(actions: Array<Record<string, unknown>>, limit = 5): boolean {
  if (actions.length < limit) return false;
  const recent = actions.slice(-limit).map((action) => JSON.stringify(action));
  return recent.every((action) => action === recent[0]);
}

async function executeNorthstarAction(
  kernel: Kernel,
  sessionId: string,
  action: NorthstarAction,
  viewportWidth: number,
  viewportHeight: number
): Promise<void> {
  switch (action.type) {
    case "click":
      await moveMouse(kernel, sessionId, action.x ?? 0, action.y ?? 0, viewportWidth, viewportHeight);
      await kernel.browsers.computer.clickMouse(sessionId, {
        x: scaleCoordinate(action.x ?? 0, viewportWidth),
        y: scaleCoordinate(action.y ?? 0, viewportHeight),
        button: action.button ?? "left"
      } as never);
      return;
    case "double_click":
      await moveMouse(kernel, sessionId, action.x ?? 0, action.y ?? 0, viewportWidth, viewportHeight);
      await kernel.browsers.computer.clickMouse(sessionId, {
        x: scaleCoordinate(action.x ?? 0, viewportWidth),
        y: scaleCoordinate(action.y ?? 0, viewportHeight),
        num_clicks: 2
      } as never);
      return;
    case "point_and_type":
      await moveMouse(kernel, sessionId, action.x ?? 0, action.y ?? 0, viewportWidth, viewportHeight);
      await kernel.browsers.computer.clickMouse(sessionId, {
        x: scaleCoordinate(action.x ?? 0, viewportWidth),
        y: scaleCoordinate(action.y ?? 0, viewportHeight),
        button: "left"
      } as never);
      await kernel.browsers.computer.typeText(sessionId, { text: action.text ?? "" } as never);
      return;
    case "type":
      await kernel.browsers.computer.typeText(sessionId, { text: action.text ?? "" } as never);
      return;
    case "keypress":
    case "key":
      await kernel.browsers.computer.pressKey(sessionId, { keys: action.keys ?? [] } as never);
      return;
    case "scroll":
      await moveMouse(
        kernel,
        sessionId,
        action.x ?? viewportWidth / 2,
        action.y ?? viewportHeight / 2,
        viewportWidth,
        viewportHeight
      );
      await kernel.browsers.computer.scroll(sessionId, {
        x: scaleCoordinate(action.x ?? viewportWidth / 2, viewportWidth),
        y: scaleCoordinate(action.y ?? viewportHeight / 2, viewportHeight),
        delta_x: action.scroll_x ?? 0,
        delta_y: action.scroll_y ?? 0
      } as never);
      return;
    case "move":
      await moveMouse(kernel, sessionId, action.x ?? 0, action.y ?? 0, viewportWidth, viewportHeight);
      return;
    case "mouse_down":
      await moveMouse(kernel, sessionId, action.x ?? 0, action.y ?? 0, viewportWidth, viewportHeight);
      await kernel.browsers.computer.clickMouse(sessionId, {
        x: scaleCoordinate(action.x ?? 0, viewportWidth),
        y: scaleCoordinate(action.y ?? 0, viewportHeight),
        click_type: "down"
      } as never);
      return;
    case "mouse_up":
      await moveMouse(kernel, sessionId, action.x ?? 0, action.y ?? 0, viewportWidth, viewportHeight);
      await kernel.browsers.computer.clickMouse(sessionId, {
        x: scaleCoordinate(action.x ?? 0, viewportWidth),
        y: scaleCoordinate(action.y ?? 0, viewportHeight),
        click_type: "up"
      } as never);
      return;
    case "drag":
      await kernel.browsers.computer.dragMouse(sessionId, {
        path: buildDragPath(action, viewportWidth, viewportHeight)
      } as never);
      return;
    case "navigate":
      await navigate(kernel, sessionId, action.url ?? "about:blank");
      return;
    case "wait":
    case "screenshot":
      await sleep(1000);
      return;
    default:
      throw new Error(`Unsupported Northstar action: ${action.type ?? "unknown"}`);
  }
}

async function navigate(kernel: Kernel, sessionId: string, url: string): Promise<void> {
  const escapedUrl = JSON.stringify(url);
  await kernel.browsers.playwright.execute(sessionId, {
    code: `await page.goto(${escapedUrl}, { waitUntil: "domcontentloaded" });`
  } as never);
}

async function capturePageState(
  kernel: Kernel,
  sessionId: string,
  progress: (message: string) => void
): Promise<SanitizedPageState | undefined> {
  const state = await kernel.browsers.playwright.execute(sessionId, {
    code: `return JSON.stringify({
      url: page.url(),
      title: await page.title(),
      text: (await page.locator("body").innerText({ timeout: 2000 }).catch(() => "")).slice(0, 500)
    });`,
    timeout_sec: 5
  } as never);
  const parsed = parsePlaywrightResult(state);
  if (!parsed) return undefined;
  const sanitized = {
    url: parsed.url,
    title: sanitizeVisibleText(parsed.title),
    text: sanitizeVisibleText(parsed.text)
  };
  progress(`page url: ${parsed.url}`);
  progress(`page title: ${sanitized.title}`);
  if (sanitized.text) progress(`page text: ${sanitized.text}`);
  return sanitized;
}

async function captureScreenshot(
  kernel: Kernel,
  sessionId: string
): Promise<{ base64: string; width: number; height: number }> {
  const screenshot = await kernel.browsers.computer.captureScreenshot(sessionId);
  const buffer = await toBuffer(screenshot);
  const dimensions = readPngDimensions(buffer);
  return {
    base64: buffer.toString("base64"),
    width: dimensions?.width ?? DEFAULT_VIEWPORT.width,
    height: dimensions?.height ?? DEFAULT_VIEWPORT.height
  };
}

async function toBuffer(value: unknown): Promise<Buffer> {
  if (Buffer.isBuffer(value)) return value;
  if (value instanceof ArrayBuffer) return Buffer.from(value);
  if (isBlobLike(value)) return Buffer.from(await value.arrayBuffer());
  if (isResponseLike(value)) {
    const blob = await value.blob();
    return Buffer.from(await blob.arrayBuffer());
  }
  throw new Error("Unsupported screenshot response type from Kernel SDK.");
}

function isBlobLike(value: unknown): value is { arrayBuffer(): Promise<ArrayBuffer> } {
  return typeof value === "object" && value !== null && "arrayBuffer" in value;
}

function isResponseLike(value: unknown): value is { blob(): Promise<{ arrayBuffer(): Promise<ArrayBuffer> }> } {
  return typeof value === "object" && value !== null && "blob" in value;
}

function pickSessionId(session: unknown): string {
  const record = session as Record<string, unknown>;
  const id = record.session_id ?? record.sessionId ?? record.id;
  if (typeof id !== "string") {
    throw new Error("Kernel session response did not include a session ID.");
  }
  return id;
}

function pickLiveViewUrl(session: unknown): string | undefined {
  const record = session as Record<string, unknown>;
  const url = record.browser_live_view_url ?? record.liveViewUrl;
  return typeof url === "string" ? url : undefined;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function moveMouse(
  kernel: Kernel,
  sessionId: string,
  x: number,
  y: number,
  viewportWidth: number,
  viewportHeight: number
): Promise<void> {
  await kernel.browsers.computer.moveMouse(sessionId, {
    x: scaleCoordinate(x, viewportWidth),
    y: scaleCoordinate(y, viewportHeight),
    smooth: true
  } as never);
}

function scaleCoordinate(value: number, dimension: number): number {
  return clampCoordinate(value, dimension);
}

function clampCoordinate(value: number, dimension: number): number {
  return Math.max(0, Math.min(Math.round(value), dimension - 1));
}

function buildDragPath(action: NorthstarAction, viewportWidth: number, viewportHeight: number): number[][] {
  const sourcePath =
    action.path && action.path.length >= 2
      ? action.path
      : [
          { x: action.x ?? 0, y: action.y ?? 0 },
          { x: action.end_x ?? action.x ?? 0, y: action.end_y ?? action.y ?? 0 }
        ];

  return sourcePath.map((point) => [
    scaleCoordinate(point.x ?? 0, viewportWidth),
    scaleCoordinate(point.y ?? 0, viewportHeight)
  ]);
}

function parsePlaywrightResult(value: unknown): { url: string; title: string; text: string } | undefined {
  const candidate = typeof value === "string" ? value : extractResultValue(value);
  if (typeof candidate !== "string") return undefined;
  try {
    const parsed = JSON.parse(candidate) as { url?: unknown; title?: unknown; text?: unknown };
    return {
      url: typeof parsed.url === "string" ? parsed.url : "",
      title: typeof parsed.title === "string" ? parsed.title : "",
      text: typeof parsed.text === "string" ? parsed.text : ""
    };
  } catch {
    return undefined;
  }
}

function extractResultValue(value: unknown): unknown {
  if (typeof value !== "object" || value === null) return value;
  const record = value as Record<string, unknown>;
  return record.result ?? record.value ?? record.data ?? value;
}

function sanitizeVisibleText(text: string): string {
  return text
    .replace(/[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}/gi, "<email>")
    .replace(/\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b/g, "<phone>")
    .replace(/\s+/g, " ")
    .slice(0, 500);
}

function readPngDimensions(buffer: Buffer): { width: number; height: number } | undefined {
  if (buffer.length < 24) return undefined;
  const isPng =
    buffer[0] === 0x89 &&
    buffer[1] === 0x50 &&
    buffer[2] === 0x4e &&
    buffer[3] === 0x47 &&
    buffer[4] === 0x0d &&
    buffer[5] === 0x0a &&
    buffer[6] === 0x1a &&
    buffer[7] === 0x0a;
  if (!isPng) return undefined;
  return {
    width: buffer.readUInt32BE(PNG_DIMENSIONS_OFFSET),
    height: buffer.readUInt32BE(PNG_DIMENSIONS_OFFSET + 4)
  };
}

function describeActionExecution(action: NorthstarAction, viewportWidth: number, viewportHeight: number): string | undefined {
  if (action.x === undefined && action.y === undefined) return undefined;
  const rawX = action.x ?? 0;
  const rawY = action.y ?? 0;
  return `coordinate map: ${action.type ?? "unknown"} raw=(${rawX},${rawY}) executed=(${scaleCoordinate(
    rawX,
    viewportWidth
  )},${scaleCoordinate(rawY, viewportHeight)}) screenshot=${viewportWidth}x${viewportHeight}`;
}

function extractModelText(response: NorthstarResponse | undefined): string | undefined {
  const chunks = response?.output
    ?.map((item) => {
      const record = item as Record<string, unknown>;
      if (typeof record.content === "string") return record.content;
      if (typeof record.text === "string") return record.text;
      if (Array.isArray(record.content)) {
        return record.content
          .map((contentItem) => {
            const contentRecord = contentItem as Record<string, unknown>;
            return contentRecord.text;
          })
          .filter((text): text is string => typeof text === "string")
          .join(" ");
      }
      return undefined;
    })
    .filter((text): text is string => typeof text === "string" && text.length > 0);

  if (!chunks || chunks.length === 0) return undefined;
  return sanitizeVisibleText(chunks.join(" ")).slice(0, 300);
}

function evaluateTaskSuccess(task: TaskScenario, pageState: SanitizedPageState | undefined): boolean {
  if (!pageState) return false;
  if (!isAllowedDomain(task, pageState.url)) return false;

  if (task.id === "ubereats_search_restaurant") {
    return isUberEatsRestaurantPage(pageState);
  }

  if (task.id === "ubereats_find_menu_item") {
    return isUberEatsRestaurantPage(pageState) && /menu|popular|recommended|add|item/i.test(pageState.text);
  }

  if (task.id === "doordash_search_restaurant") {
    return isDoorDashRestaurantPage(pageState);
  }

  if (task.id === "airbnb_red_exterior_san_francisco") {
    return isAirbnbSearchOrListingPage(pageState);
  }

  if (task.id === "craigslist_sf_rentals_under_3000") {
    return false;
  }

  return task.successCriteria.every((criterion) => {
    const significantWords = criterion
      .toLowerCase()
      .split(/[^a-z0-9]+/)
      .filter((word) => word.length > 3);
    const haystack = `${pageState.url} ${pageState.title} ${pageState.text}`.toLowerCase();
    return significantWords.some((word) => haystack.includes(word));
  });
}

function isAllowedDomain(task: TaskScenario, url: string): boolean {
  try {
    const hostname = new URL(url).hostname;
    return task.allowedDomains.some((domain) => hostname === domain || hostname.endsWith(`.${domain}`));
  } catch {
    return false;
  }
}

function isUberEatsRestaurantPage(pageState: SanitizedPageState): boolean {
  const haystack = `${pageState.url} ${pageState.title} ${pageState.text}`;
  const looksLikeRestaurantUrl = /ubereats\.com\/store\//i.test(pageState.url);
  const hasMenuSignals = /\b(menu|popular|featured|recommended|reviews|delivery fee|add to order|restaurant info)\b/i.test(
    haystack
  );
  const hasSearchOnlySignals = /\b(search uber eats|enter delivery address|delivery address|sign in)\b/i.test(
    pageState.text
  );
  return looksLikeRestaurantUrl && hasMenuSignals && !hasSearchOnlySignals;
}

function isDoorDashRestaurantPage(pageState: SanitizedPageState): boolean {
  const haystack = `${pageState.url} ${pageState.title} ${pageState.text}`;
  const looksLikeStoreUrl = /doordash\.com\/store\//i.test(pageState.url);
  const hasMenuSignals = /\b(menu|popular items|most ordered|ratings|reviews|delivery fee|add to cart|group order)\b/i.test(
    haystack
  );
  const hasSearchOnlySignals = /\b(delivery address|enter address|sign in|sign up|restaurants near me)\b/i.test(
    pageState.text
  );
  return looksLikeStoreUrl && hasMenuSignals && !hasSearchOnlySignals;
}

function isAirbnbSearchOrListingPage(pageState: SanitizedPageState): boolean {
  const haystack = `${pageState.url} ${pageState.title} ${pageState.text}`;
  const onAirbnb = /airbnb\.com/i.test(pageState.url);
  const isHomepage = /^https:\/\/www\.airbnb\.com\/?(\?.*)?$/i.test(pageState.url);
  const hasSanFranciscoSignals = /\b(san francisco|sf bay area)\b/i.test(haystack);
  const hasSearchOrListingSignals =
    /\/s\/|\/rooms\//i.test(pageState.url) ||
    /\b(stays in san francisco|san francisco homes|night|reviews|hosted by)\b/i.test(haystack);
  const inBlockedFlow = /\b(reserve|confirm and pay|request to book|log in|sign up|payment)\b/i.test(pageState.text);
  return onAirbnb && !isHomepage && hasSanFranciscoSignals && hasSearchOrListingSignals && !inBlockedFlow;
}

function evaluateTextAnswerSuccess(task: TaskScenario, modelText: string | undefined): boolean {
  if (!modelText) return false;
  if (task.id === "airbnb_red_exterior_san_francisco") {
    return /\bred\b/i.test(modelText) && /\b(airbnb|listing|home|url|https?:\/\/)\b/i.test(modelText);
  }
  if (task.id === "craigslist_sf_rentals_under_3000") {
    const priceMentions = modelText.match(/\$\s?\d{3,4}/g) ?? [];
    return priceMentions.length >= 3 && /\b(craigslist|reply|contact|location|address|url|https?:\/\/)\b/i.test(modelText);
  }
  return false;
}

async function withTimeout<T>(promise: Promise<T>, timeoutMs: number, message: string): Promise<T> {
  let timeout: NodeJS.Timeout | undefined;
  try {
    return await Promise.race([
      promise,
      new Promise<never>((_, reject) => {
        timeout = setTimeout(() => reject(new Error(message)), timeoutMs);
      })
    ]);
  } finally {
    if (timeout) clearTimeout(timeout);
  }
}
