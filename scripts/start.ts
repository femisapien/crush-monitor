export {};
// Keep headless use possible with CRUSH_OPEN_BROWSER=0.
process.env.CRUSH_OPEN_BROWSER ??= "1";
await import("../server/index");
