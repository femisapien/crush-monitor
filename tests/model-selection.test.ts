import { test } from "node:test";
import assert from "node:assert/strict";
import { preferredModel, modelLabel } from "../shared/model-selection";
test("discovery repairs invalid manual IDs, preserves available IDs, never invents a model", () => {
  const ids = ["deepseek-pro", "deepseek-flash"];
  assert.equal(
    preferredModel(ids, "wrong model", "deepseek"),
    "deepseek-flash",
  );
  assert.equal(preferredModel(ids, "deepseek-pro", "deepseek"), "deepseek-pro");
  assert.equal(
    preferredModel(["other-model"], "deepseek-flash", "deepseek"),
    "other-model",
  );
  assert.equal(preferredModel([], "", "custom"), "");
  assert.match(modelLabel("deepseek-flash", "deepseek"), /V4.1 Flash/);
});
