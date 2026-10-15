import { describe, it, expect } from "vitest";
import { generateText, streamText, generateObject, jsonSchema } from "ai";
import { createBuilder } from "../src";
import type { LanguageModelV3CallOptions, LanguageModelV3StreamPart } from "@ai-sdk/provider";

const prompt: LanguageModelV3CallOptions["prompt"] = [{ role: "user", content: [{ type: "text", text: "hi" }] }];
function model(events: object[], calls: any[] = []) {
  return createBuilder({
    apiKey: "test", userId: "test", privateKey: "test",
    fetch: (async (_url, init) => {
      calls.push(JSON.parse(init!.body as string));
      // Split NDJSON at arbitrary byte boundaries too.
      const bytes = new TextEncoder().encode(events.map(e => JSON.stringify(e)).join("\n"));
      return new Response(new ReadableStream({ start(c) {
        for (let i = 0; i < bytes.length; i += 7) c.enqueue(bytes.slice(i, i + 7));
        c.close();
      }}));
    }) as typeof fetch,
  })("test");
}
const done = (usage?: unknown) => ({ type: "done", stopReason: "end_turn", usage });
async function parts(m: ReturnType<typeof model>, options: LanguageModelV3CallOptions) {
  const r = await m.doStream(options);
  const out: LanguageModelV3StreamPart[] = [];
  for await (const p of r.stream) out.push(p);
  return out;
}

describe("token usage", () => {
  it.each([
    { inputTokens: 12, outputTokens: 5 },
    { input_tokens: 12, output_tokens: 5 },
    { prompt_tokens: 12, completion_tokens: 5 },
  ])("maps explicit token counts for generate and stream: %j", async usage => {
    const r = await model([done(usage)]).doGenerate({ prompt });
    expect(r.usage.inputTokens.total).toBe(12);
    expect(r.usage.outputTokens.total).toBe(5);
    const ps = await parts(model([done(usage)]), { prompt });
    expect(ps.at(-1)).toMatchObject({ type: "finish", usage: r.usage });
  });
  it("AI SDK computes input/output/total for both APIs", async () => {
    const events = [{ type: "text", content: "ok" }, done({ inputTokens: 12, outputTokens: 5 })];
    const r = await generateText({ model: model(events), prompt: "hi" });
    expect(r.usage).toMatchObject({ inputTokens: 12, outputTokens: 5, totalTokens: 17 });
    const stream = streamText({ model: model(events), prompt: "hi" });
    await stream.consumeStream();
    expect(await stream.usage).toMatchObject({ inputTokens: 12, outputTokens: 5, totalTokens: 17 });
  });
  it.each([undefined, null, {}, { total: 100, fast: 10, quality: 90 }, { inputTokens: -1, outputTokens: "5" }])("does not invent tokens from %j", async usage => {
    const r = await model([done(usage)]).doGenerate({ prompt });
    expect(r.usage.inputTokens.total).toBeUndefined();
    expect(r.usage.outputTokens.total).toBeUndefined();
  });
  it("preserves zero and partial counts and ignores internal compaction usage", async () => {
    const r = await model([{ ...done({ inputTokens: 99, outputTokens: 99 }), compact: true }, done({ inputTokens: 0 })]).doGenerate({ prompt });
    expect(r.usage.inputTokens.total).toBe(0);
    expect(r.usage.outputTokens.total).toBeUndefined();
  });
});

describe("JSON response emulation", () => {
  const schema = { type: "object" as const, properties: { ok: { type: "boolean" as const } }, required: ["ok"], additionalProperties: false };
  it("adds schema reminder and strips fences split across events", async () => {
    const calls: any[] = [];
    const events = ["``", "`json\n", '{"ok":true}', "\n`", "``"].map(content => ({ type: "text", content }));
    const r = await model([...events, done()], calls).doGenerate({ prompt, responseFormat: { type: "json", schema } });
    expect(r.content).toEqual([{ type: "text", text: '{"ok":true}' }]);
    expect(calls[0].systemReminders.join("\n")).toContain(JSON.stringify(schema));
    expect(calls[0].systemReminders.join("\n")).toContain("only valid JSON, no prose or code fences");
    expect(r.warnings).toEqual([]);
  });
  it("works through generateObject and streamed text", async () => {
    const events = [{ type: "text", content: '```json\n{"ok":true}\n```' }, done()];
    const r = await generateObject({ model: model(events), schema: jsonSchema(schema), prompt: "hi" });
    expect(r.object).toEqual({ ok: true });
    const ps = await parts(model(events), { prompt, responseFormat: { type: "json" } });
    expect(ps.filter(p => p.type === "text-delta").map(p => p.delta).join("")).toBe('{"ok":true}');
  });
  it("leaves plain JSON, invalid prose, and normal text mode untouched", async () => {
    for (const content of ['{"ok":true}', "not JSON"]) {
      const r = await model([{ type: "text", content }, done()]).doGenerate({ prompt, responseFormat: { type: "json" } });
      expect(r.content).toEqual([{ type: "text", text: content }]);
    }
    const content = '```json\n{}\n```';
    const r = await model([{ type: "text", content }, done()]).doGenerate({ prompt });
    expect(r.content).toEqual([{ type: "text", text: content }]);
  });
});

describe("image warnings", () => {
  it("warns for SDK v3 image files including historical turns", async () => {
    const imagePrompt: LanguageModelV3CallOptions["prompt"] = [
      { role: "user", content: [{ type: "text", text: "look" }, { type: "file", mediaType: "image/png", data: new Uint8Array([1]) }] },
      { role: "assistant", content: [{ type: "text", text: "ok" }] },
      ...prompt,
    ];
    const r = await model([done()]).doGenerate({ prompt: imagePrompt });
    expect(r.warnings).toContainEqual(expect.objectContaining({ type: "unsupported", feature: "image input" }));
    const ps = await parts(model([done()]), { prompt: imagePrompt });
    expect(ps[0]).toMatchObject({ type: "stream-start", warnings: r.warnings });
  });
  it("rejects image-only requests with a clear error", async () => {
    await expect(model([done()]).doGenerate({ prompt: [{ role: "user", content: [
      { type: "file", mediaType: "image/png", data: new Uint8Array([1]) },
    ] }] })).rejects.toThrow("does not support image input");
  });
  it("does not warn for text-only prompts", async () => {
    expect((await model([done()]).doGenerate({ prompt })).warnings).toEqual([]);
  });
});
