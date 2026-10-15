import { readFileSync } from "node:fs";
import { expect, it } from "vitest";
import { generateText, streamText } from "ai";
import { BuilderProviderError, createBuilder, type BuilderChatSettings } from "../src";
import type { LanguageModelV3StreamPart } from "@ai-sdk/provider";

const prompt = [{ role: "user" as const, content: [{ type: "text" as const, text: "hi" }] }];
const wire = (...events: object[]) => events.map(e => JSON.stringify(e)).join("\n") + "\n";
const done = { type: "done", content: "", stopReason: "end_turn" };
const answer = wire({ type: "text", content: "ok" }, done);
const report = readFileSync(new URL("./fixtures/compaction-only.ndjson", import.meta.url), "utf8");
function setup(body = answer, settings: BuilderChatSettings = {}) {
  const calls: any[] = [];
  const model = createBuilder({
    apiKey: "test", privateKey: "test", userId: "test",
    fetch: (async (_url, init) => {
      calls.push(JSON.parse(init!.body as string));
      const bytes = new TextEncoder().encode(body);
      return new Response(new ReadableStream({ start(c) {
        for (let i = 0; i < bytes.length; i += 7) c.enqueue(bytes.slice(i, i + 7));
        c.close();
      }}));
    }) as typeof fetch,
  })("test", settings);
  return { model, calls };
}

it.each([1024, 65536])("omits the wire cap and warns for call maxOutputTokens %i", async maxOutputTokens => {
  const { model, calls } = setup();
  const r = await model.doGenerate({ prompt, maxOutputTokens });
  expect(calls[0]).not.toHaveProperty("maxTokens");
  expect(r.warnings).toEqual([expect.objectContaining({ type: "unsupported", feature: "maxOutputTokens" })]);
  expect(r.content).toEqual([{ type: "text", text: "ok" }]);
});
it("ignores the model-level cap too", async () => {
  const { model, calls } = setup(answer, { maxOutputTokens: 1024 });
  const r = await model.doGenerate({ prompt });
  expect(calls[0]).not.toHaveProperty("maxTokens");
  expect(r.warnings[0]).toMatchObject({ feature: "maxOutputTokens" });
});
it("uncapped requests have no cap warning", async () => {
  expect((await setup().model.doGenerate({ prompt })).warnings).toEqual([]);
});
it("surfaces the exact reported five-compaction + continue sequence in doGenerate", async () => {
  const { model, calls } = setup(report);
  await expect(model.doGenerate({ prompt })).rejects.toMatchObject({
    name: "BuilderProviderError", code: "compaction-only",
    message: expect.stringContaining("without an answer"),
  });
  expect(calls).toHaveLength(1);
});
it("streams a clear provider error and one error finish for the report", async () => {
  const { model, calls } = setup(report);
  const { stream } = await model.doStream({ prompt });
  const parts: LanguageModelV3StreamPart[] = [];
  for await (const part of stream) parts.push(part);
  const errors = parts.filter(p => p.type === "error");
  expect(errors).toHaveLength(1);
  expect(errors[0].error).toBeInstanceOf(BuilderProviderError);
  expect(parts.filter(p => p.type === "finish")).toEqual([
    expect.objectContaining({ finishReason: expect.objectContaining({ unified: "error" }) }),
  ]);
  expect(calls).toHaveLength(1);
});
it.each([
  wire(done),
  wire({ type: "text", content: " \n" }, done),
  wire({ type: "thinking", content: "thinking only" }, done),
])("rejects empty/whitespace/reasoning-only completed answers", async body => {
  await expect(setup(body).model.doGenerate({ prompt })).rejects.toMatchObject({ code: "empty-response" });
});
it("accepts compaction followed by a real answer", async () => {
  const r = await setup(report + answer).model.doGenerate({ prompt });
  expect(r.content).toEqual([{ type: "text", text: "ok" }]);
  expect(r.finishReason.unified).toBe("stop");
});
it("accepts compaction followed by a tool-only response", async () => {
  const { model } = setup(report + wire({ type: "tool", id: "t", name: "f", content: "{}" }, done));
  const r = await model.doGenerate({ prompt, tools: [{ type: "function", name: "f", inputSchema: {} }] });
  expect(r.content).toEqual([expect.objectContaining({ type: "tool-call", toolName: "f" })]);
  expect(r.finishReason.unified).toBe("tool-calls");
});
it("generateText rejects and streamText exposes the same provider error", async () => {
  await expect(generateText({ model: setup(report).model, prompt: "hi", maxRetries: 0 })).rejects.toBeInstanceOf(BuilderProviderError);
  const errors: unknown[] = [];
  const r = streamText({ model: setup(report).model, prompt: "hi", maxRetries: 0, onError: ({ error }) => { errors.push(error); } });
  await r.consumeStream();
  expect(errors).toHaveLength(1);
  expect(errors[0]).toBeInstanceOf(BuilderProviderError);
});
