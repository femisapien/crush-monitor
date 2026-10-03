import { ProviderError, httpError } from "./provider";
export function errorDetail(body: unknown): string {
  if (!body || typeof body !== "object") return "";
  const b = body as {
    error?: { message?: unknown; code?: unknown };
    message?: unknown;
    code?: unknown;
  };
  return [b.error?.code, b.error?.message, b.code, b.message]
    .filter((v) => typeof v === "string")
    .join(" ")
    .slice(0, 6000);
}
// Return our own messages; upstream responses can contain request data or secrets.
export function upstreamError(status: number, name: string, body: unknown) {
  const detail = errorDetail(body);
  if (
    /not activated|not have activated|product.*not.*activat|market app does not exist/i.test(
      detail,
    )
  )
    return new ProviderError(
      403,
      `${name}：这个型号的服务尚未开通，请到平台控制台开通对应模型，或选择已开通的型号。`,
    );
  if (
    /free.?tier|free quota|AllocationQuota|Arrearage|insufficient.*(?:balance|credit)|quota.*exhaust/i.test(
      detail,
    )
  )
    return new ProviderError(
      402,
      `${name}：这个型号的额度不足或只允许使用免费额度，请检查控制台的额度与计费设置。`,
    );
  if (/enable_thinking.*(?:restricted to True|must be true)/i.test(detail))
    return new ProviderError(
      400,
      `${name}：这个型号强制使用思考模式，请改用支持快速模式的型号。`,
    );
  return httpError(status, name);
}
export function jsonModeUnsupported(body: unknown) {
  const detail = errorDetail(body);
  return (
    /response_format|json_object|json mode/i.test(detail) &&
    /not support|unsupported|not compatible|不支持/i.test(detail)
  );
}
