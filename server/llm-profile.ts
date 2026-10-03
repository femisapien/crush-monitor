import type { LlmConfig } from "./model-config";
export function isBailian(endpoint: string) {
  const host = new URL(endpoint).hostname;
  return (
    [
      "dashscope.aliyuncs.com",
      "dashscope-intl.aliyuncs.com",
      "dashscope-us.aliyuncs.com",
    ].includes(host) || /\.maas\.aliyuncs\.com$/.test(host)
  );
}
// Policies belong to the serving platform, not just the upstream model name.
// Reference: Alibaba Model Studio OpenAI Chat API, 2026-10-04.
export function fastInferenceOptions(
  config: LlmConfig,
): Record<string, unknown> {
  const host = new URL(config.endpoint).hostname,
    m = config.model.toLowerCase();
  if (isBailian(config.endpoint)) {
    if (m === "minimax/minimax-m3") return { thinking: { type: "disabled" } };
    if (/^(?:zhipu\/)?glm-5\.3/.test(m)) return { reasoning_effort: "low" };
    if (m === "qwen3.7-max-2026-05-17") return { thinking_budget: 1024 };
    if (/^qwen3\.8-2\.4t/.test(m)) return { reasoning_effort: "low" };
    if (/thinking|^qwq|^deepseek-r1|\/deepseek-r1/.test(m)) return {};
    if (
      /^qwen3(?:\.[5-8])?-/.test(m) &&
      !/instruct|coder/.test(m) &&
      m !== "qwen3.7-max-preview"
    )
      return { enable_thinking: false };
    if (/^qwen-(plus|flash|turbo)(?:-|$)/.test(m))
      return { enable_thinking: false };
    if (/^(?:(?:siliconflow|vanchin)\/)?deepseek-v(?:3\.[12]|4)/.test(m))
      return { enable_thinking: false };
    if (/^(?:kimi\/)?kimi-k2\.[56](?:-|$)/.test(m) || m === "kimi-k3")
      return { enable_thinking: false };
    if (/^(?:zhipu\/)?glm-(4\.[567]|5(?:[.-]|$))/.test(m))
      return { enable_thinking: false };
    return {};
  }
  if (
    host === "api.deepseek.com" ||
    (host === "open.bigmodel.cn" &&
      /^glm-(4\.[567]|5(?:[.-]|$))/.test(m) &&
      !/^glm-5\.3/.test(m)) ||
    (["api.moonshot.cn", "api.moonshot.ai"].includes(host) &&
      /^kimi-k2\.[56](?:-|$)/.test(m))
  )
    return { thinking: { type: "disabled" } };
  return {};
}
