import { test } from "node:test";
import assert from "node:assert/strict";
import { readLlmResponse } from "../server/llm-stream";
import { fastInferenceOptions } from "../server/llm-profile";
import { settingsConfig, type LlmConfig } from "../server/model-config";
import { evaluateLlm } from "../server/llm";
import { isTextAnalysisModel } from "../shared/model-selection";
const settings = {
  preset: "qwen",
  protocol: "openai" as const,
  baseUrl: "https://dashscope.aliyuncs.com/compatible-mode/v1",
  apiKey: "test-only-key",
  model: "qwen3.8-flash",
  jsonMode: true,
};
function config(model = settings.model) {
  return settingsConfig({ ...settings, model }) as LlmConfig;
}
function sse(parts: unknown[]) {
  return new Response(
    new ReadableStream({
      start(c) {
        const bytes = new TextEncoder().encode(
          parts
            .map(
              (p) =>
                "data: " +
                (typeof p === "string" ? p : JSON.stringify(p)) +
                "\r\n\r\n",
            )
            .join(""),
        );
        for (let i = 0; i < bytes.length; i += 3)
          c.enqueue(bytes.slice(i, i + 3));
        c.close();
      },
    }),
    { headers: { "content-type": "text/event-stream" } },
  );
}
test("Bailian model-family controls follow serving endpoint, including forced thinking", () => {
  for (const id of [
    "qwen3.8-flash",
    "qwen3.8-max",
    "qwen3.6-max-preview",
    "qwen3-32b",
    "deepseek-v4.1-flash",
    "kimi-k2.6",
    "glm-5.2",
  ])
    assert.deepEqual(
      fastInferenceOptions(config(id)),
      { enable_thinking: false },
      id,
    );
  assert.deepEqual(fastInferenceOptions(config("glm-5.3")), {
    reasoning_effort: "low",
  });
  assert.deepEqual(fastInferenceOptions(config("qwen3.7-max-2026-05-17")), {
    thinking_budget: 1024,
  });
  assert.deepEqual(fastInferenceOptions(config("MiniMax/MiniMax-M3")), {
    thinking: { type: "disabled" },
  });
  assert.deepEqual(fastInferenceOptions(config("deepseek-r1")), {});
  assert.deepEqual(
    fastInferenceOptions({
      ...config(),
      endpoint: "https://unknown.example/v1/chat/completions",
    }),
    {},
  );
});
test("SSE handles split UTF8/CRLF and usage chunks while discarding reasoning", async () => {
  const r = await readLlmResponse(
    sse([
      {
        choices: [
          { index: 0, delta: { reasoning_content: "PRIVATE_REASONING" } },
        ],
      },
      { choices: [{ index: 0, delta: { content: '{"中文":' } }] },
      {
        choices: [
          { index: 0, delta: { content: "true}" }, finish_reason: "stop" },
        ],
      },
      { usage: { prompt_tokens: 3, completion_tokens: 4 }, choices: [] },
      "[DONE]",
    ]),
  );
  assert.equal(r.choices[0].message.content, '{"中文":true}');
  assert.equal(r.usage.completion_tokens, 4);
  assert.ok(!JSON.stringify(r).includes("PRIVATE_REASONING"));
  await assert.rejects(
    readLlmResponse(
      sse([{ choices: [{ index: 0, delta: { content: "{}" } }] }, "[DONE]"]),
    ),
    /连接中断/,
  );
});
test("SSE errors preserve activation errors and never echo upstream secrets", async () => {
  await assert.rejects(
    readLlmResponse(
      sse([
        {
          error: {
            message:
              "Aliyun market app does not exist, the user may not have activated the service. secret-test",
          },
        },
      ]),
    ),
    (e) =>
      (e as any).status === 403 &&
      /尚未开通/.test(String(e)) &&
      !String(e).includes("secret-test"),
  );
});
test("JSON-mode fallback only retries a specific unsupported parameter and still validates output", async () => {
  let calls = 0;
  const q = { ok: { type: "noul" as const, instructions: "yes?" } };
  const r = await evaluateLlm(
    { state: "synthetic", questions: q },
    config(),
    undefined,
    async (_u, o) => {
      const b = JSON.parse(String(o?.body));
      calls++;
      assert.equal(b.stream, true);
      assert.equal(b.enable_thinking, false);
      if (calls === 1) {
        assert.ok(b.response_format);
        return Response.json(
          {
            error: { message: "response_format json_object is not supported" },
          },
          { status: 400 },
        );
      }
      assert.equal(b.response_format, undefined);
      return sse([
        {
          choices: [
            {
              index: 0,
              delta: { content: '{"answers":{"q0":0.8}}' },
              finish_reason: "stop",
            },
          ],
        },
        "[DONE]",
      ]);
    },
  );
  assert.equal(calls, 2);
  assert.equal((r.answers.ok as any).noul, 0.8);
  calls = 0;
  await assert.rejects(
    evaluateLlm({ state: "x", questions: q }, config(), undefined, async () => {
      calls++;
      return Response.json(
        { error: { message: "The product is not activated secret-test" } },
        { status: 400 },
      );
    }),
    /尚未开通/,
  );
  assert.equal(calls, 1);
});
test("catalog hides dedicated non-chat APIs without hiding text-capable vision models", () => {
  for (const m of [
    "qwen-image-3.0",
    "qwen3-tts-flash",
    "qwen-mt-plus",
    "qwen3.7-text-embedding",
    "qwen-vl-ocr",
    "MiniMax/speech-02-hd",
    "qwen3.8-omni-flash-realtime",
  ])
    assert.equal(isTextAnalysisModel(m), false, m);
  for (const m of [
    "qwen3.8-flash",
    "qwen3-vl-plus",
    "qwen3-coder-plus",
    "kimi/kimi-k2.6",
    "deepseek-v4.1-flash",
  ])
    assert.equal(isTextAnalysisModel(m), true, m);
});
