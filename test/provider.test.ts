import { describe, it, expect } from "vitest";
import { createBuilder, mapFinishReason, redact } from "../src/index";

const ndjson = (...e: object[]) => e.map((x) => JSON.stringify(x)).join("\n") + "\n";
function mk(body: string, status = 200, calls: any[] = []) {
  const f = (async (url: any, init: any) => {
    calls.push({ url: String(url), body: JSON.parse(init.body), headers: init.headers });
    return new Response(body, { status });
  }) as typeof fetch;
  return createBuilder({ apiKey: "PUBKEY123", userId: "u1", privateKey: "bpk-secret999", fetch: f })("m");
}
const user = (t: string) => ({ role: "user" as const, content: [{ type: "text" as const, text: t }] });
const base = { prompt: [user("hi")] };

describe("finish reasons", () => {
  it("maps", () => {
    expect(mapFinishReason("end_turn").unified).toBe("stop");
    expect(mapFinishReason("tool_use").unified).toBe("tool-calls");
    expect(mapFinishReason("max_tokens").unified).toBe("length");
    expect(mapFinishReason("error").unified).toBe("error");
    expect(mapFinishReason(null).unified).toBe("other");
  });
});

describe("doGenerate", () => {
  it("returns tool calls with stringified input", async () => {
    const m = mk(ndjson({ type: "text", content: "x" }, { type: "tool", id: "t1", name: "get_w", content: '{"a":1}' }, { type: "done", stopReason: "tool_use" }));
    const r = await m.doGenerate({ ...base, tools: [{ type: "function", name: "get_w", inputSchema: { type: "object" } }] });
    const tc: any = r.content.find((c) => c.type === "tool-call");
    expect(tc).toMatchObject({ toolCallId: "t1", toolName: "get_w", input: '{"a":1}' });
    expect(r.finishReason.unified).toBe("tool-calls");
  });
  it("sends tool results and keeps session", async () => {
    const calls: any[] = [];
    const m = mk(ndjson({ type: "text", content: "ok" }, { type: "done", stopReason: "end_turn" }), 200, calls);
    await m.doGenerate(base);
    await m.doGenerate({ prompt: [user("hi"), { role: "assistant", content: [{ type: "tool-call", toolCallId: "t1", toolName: "get_w", input: {} }] }, { role: "tool", content: [{ type: "tool-result", toolCallId: "t1", toolName: "get_w", output: { type: "json", value: { ok: 1 } } }] }] });
    expect(calls[1].body.sessionId).toBe(calls[0].body.sessionId);
    expect(calls[1].body.toolResults[0]).toMatchObject({ tool_use_id: "t1", content: '{"ok":1}' });
    expect(calls[1].body.userPrompt).toBeUndefined();
  });
  it("replays earlier turns for unknown session", async () => {
    const calls: any[] = [];
    const m = mk(ndjson({ type: "text", content: "ok" }, { type: "done", stopReason: "end_turn" }), 200, calls);
    await m.doGenerate({ prompt: [user("my name is Bo"), { role: "assistant", content: [{ type: "text", text: "ok" }] }, user("what is my name")] });
    expect(calls[0].body.userPrompt).toContain("my name is Bo");
    expect(calls[0].body.userPrompt).toContain("what is my name");
  });
  it("does not leak secrets in errors", async () => {
    const m = mk('{"type":"error","code":"x","message":"bad bpk-secret999 apiKey=PUBKEY123","id":"1","stopReason":"error"}', 403);
    const e: any = await m.doGenerate(base).catch((x) => x);
    const dump = JSON.stringify({ m: e.message, u: e.url, b: e.responseBody, r: e.requestBodyValues });
    expect(dump).not.toContain("bpk-secret999");
    expect(dump).not.toContain("PUBKEY123");
    expect(e.statusCode).toBe(403);
  });
  it("throws on stream error without done", async () => {
    const m = mk(ndjson({ type: "error", code: "ask-to-continue", message: "limit" }));
    await expect(m.doGenerate(base)).rejects.toThrow(/ask-to-continue/);
  });
});

describe("doStream", () => {
  const collect = async (m: any, o: any) => {
    const { stream } = await m.doStream(o);
    const out: any[] = [];
    const rd = stream.getReader();
    for (;;) { const { done, value } = await rd.read(); if (done) break; out.push(value); }
    return out;
  };
  it("emits well formed parts", async () => {
    const parts = await collect(mk(ndjson({ type: "thinking", content: "t" }, { type: "delta", content: "a" }, { type: "delta", content: "b" }, { type: "done", stopReason: "end_turn" })), base);
    expect(parts.map((p) => p.type)).toEqual(["stream-start", "reasoning-start", "reasoning-delta", "reasoning-end", "text-start", "text-delta", "text-delta", "text-end", "finish"]);
    expect(parts.at(-1).finishReason.unified).toBe("stop");
  });
  it("tool call finish", async () => {
    const parts = await collect(mk(ndjson({ type: "tool", id: "t", name: "f", content: "{}" }, { type: "done", stopReason: "end_turn" })), { ...base, tools: [{ type: "function", name: "f", inputSchema: {} }] });
    expect(parts.find((p) => p.type === "tool-call").input).toBe("{}");
    expect(parts.at(-1).finishReason.unified).toBe("tool-calls");
  });
  it("truncated stream errors", async () => {
    const parts = await collect(mk(ndjson({ type: "delta", content: "a" })), base);
    expect(parts.some((p) => p.type === "error")).toBe(true);
    expect(parts.at(-1).finishReason.unified).toBe("error");
  });
});

describe("redact", () => {
  it("scrubs", () => expect(redact("Bearer abc.def apiKey=zzz", [])).toBe("Bearer [redacted] apiKey=[redacted]"));
});
