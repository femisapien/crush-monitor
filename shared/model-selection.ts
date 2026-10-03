// Select only an ID actually returned by this account's model-list endpoint.
export function preferredModel(
  ids: string[],
  current: string,
  preset: string,
): string {
  if (ids.includes(current)) return current;
  const preferred: Record<string, string[]> = {
    deepseek: ["deepseek-flash", "deepseek-v4-flash", "deepseek-chat"],
    qwen: [
      "qwen3.8-flash",
      "qwen3.7-flash",
      "qwen3.5-flash",
      "qwen-plus",
      "qwen-flash",
      "qwen-turbo",
    ],
    zhipu: ["glm-4.7-flash", "glm-4.5-flash"],
  };
  return (
    preferred[preset]?.find((id) => ids.includes(id)) ??
    ids.find(
      (id) =>
        /flash|mini|haiku|turbo/i.test(id) &&
        !/embed|rerank|image|audio|tts|vision/i.test(id),
    ) ??
    ids.find((id) => !/embed|rerank|image|audio|tts/i.test(id)) ??
    ids[0] ??
    ""
  );
}
export function modelLabel(id: string, preset: string): string {
  return preset === "deepseek" && id === "deepseek-flash"
    ? "DeepSeek V4.1 Flash · deepseek-flash"
    : id;
}

/** Exclude dedicated non-chat APIs; text-capable vision/coding models remain selectable. */
export function isTextAnalysisModel(id: string) {
  return (
    !/(?:embedding|rerank|speech|(?:^|[-/])tts(?:-|$)|(?:^|[-/])asr(?:-|$)|audio|realtime|livetranslate|(?:^|[-/])ocr(?:-|$)|s2s)/i.test(
      id,
    ) &&
    !/^(?:qwen-image|wan\d|z-image|qwen-mt-|qwen-deep-(?:research|search)|gui-|(?:test-)?sre-)/i.test(
      id,
    )
  );
}
