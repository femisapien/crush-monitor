import { upstreamError } from "./upstream-error";
import { ProviderError } from "./provider";
// Normalize SSE to the same envelope as non-streaming Chat Completions.
// Reasoning is deliberately discarded; only final content can become judgments.
export async function readLlmResponse(response: Response, name = "模型服务") {
  if (!response.headers.get("content-type")?.includes("text/event-stream")) {
    const data = await response.json();
    if (data.error) throw upstreamError(502, name, data);
    return data;
  }
  if (!response.body)
    throw new ProviderError(502, "模型没有返回内容，请重试。");
  const reader = response.body.getReader(),
    decoder = new TextDecoder();
  let buffer = "",
    content = "",
    finish: string | null = null,
    usage: unknown = {},
    ended = false,
    bytes = 0;
  function event(block: string) {
    const text = block
      .split("\n")
      .filter((l) => l.startsWith("data:"))
      .map((l) => l.slice(5).trimStart())
      .join("\n");
    if (!text) return;
    if (text.trim() === "[DONE]") {
      ended = true;
      return;
    }
    const data = JSON.parse(text);
    if (data.error) throw upstreamError(502, name, data);
    if (data.usage) usage = data.usage;
    const c = data.choices?.find(
      (c: { index?: number }) => c.index === 0 || c.index === undefined,
    );
    if (typeof c?.delta?.content === "string") content += c.delta.content;
    if (c?.finish_reason) finish = c.finish_reason;
  }
  try {
    while (!ended) {
      const { value, done } = await reader.read();
      if (done) break;
      bytes += value.byteLength;
      if (bytes > 8 * 1024 * 1024)
        throw new ProviderError(502, "模型输出过长，请换用快速文本模型。");
      buffer += decoder.decode(value, { stream: true });
      buffer = buffer.replace(/\r\n/g, "\n");
      let at: number;
      while ((at = buffer.indexOf("\n\n")) >= 0) {
        event(buffer.slice(0, at));
        buffer = buffer.slice(at + 2);
      }
    }
    buffer += decoder.decode();
    if (buffer.trim()) event(buffer);
    if (!finish)
      throw new ProviderError(502, "模型连接中断，未收到完整结果，请重试。");
    return {
      choices: [{ message: { content }, finish_reason: finish }],
      usage,
    };
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}
