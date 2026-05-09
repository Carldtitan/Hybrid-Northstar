import assert from "node:assert/strict";
import test from "node:test";
import { isActionBlocked, isSensitiveText, sanitizeAction } from "../../src/safety/guardrails.js";
import { ubereatsSearchRestaurant } from "../../src/tasks/scenarios/ubereats.js";

test("blocks task stop phrases", () => {
  assert.equal(
    isActionBlocked(ubereatsSearchRestaurant, {
      type: "click",
      target: "Place order",
      reason: "Proceed to place order"
    }),
    true
  );
});

test("redacts sensitive action text", () => {
  const action = sanitizeAction({
    type: "type",
    target: "password",
    value: "password: secret",
    reason: "enter password"
  });

  assert.equal(action.target, "<redacted>");
  assert.equal(action.value, "<redacted>");
  assert.equal(action.reason, "<redacted>");
});

test("detects sensitive text", () => {
  assert.equal(isSensitiveText("Authorization: Bearer abc"), true);
  assert.equal(isSensitiveText("open restaurant page"), false);
});
