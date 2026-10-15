import { expect, it } from "vitest";
import { generateText } from "ai";
import { createBuilder } from "../src";
import type { LanguageModelV3Prompt, LanguageModelV3FilePart } from "@ai-sdk/provider";

const image = (data: LanguageModelV3FilePart["data"] = new Uint8Array([1, 2, 3]), mediaType = "image/png"): LanguageModelV3FilePart =>
  ({ type: "file", mediaType, data });
const prompt = (...parts: LanguageModelV3FilePart[]): LanguageModelV3Prompt =>
  [{ role: "user", content: parts }];
function setup(options: { status?: number; native?: boolean; echo?: boolean; network?: boolean } = {}) {
  const calls: any[] = [];
  const m = createBuilder({
    apiKey: "test-key", privateKey: "test-private", userId: "test",
    fetch: (async (_url, init) => {
      const body = JSON.parse(init!.body as string); calls.push(body);
      const echoed = body.attachments?.[0]?.dataUrl;
      if (options.network) throw new Error(echoed);
      const event = options.echo
        ? { type: "error", message: echoed, usage: { detail: echoed } }
        : { type: "done", stopReason: "end_turn", nextUrl: options.native ? "cgen://next" : undefined };
      return new Response((options.echo ? "" : JSON.stringify({ type: "text", content: "ok" }) + "\n") + JSON.stringify(event) + "\n", { status: options.status ?? 200 });
    }) as typeof fetch,
  })("test");
  return { m, calls };
}

it("maps image bytes to the CLI upload shape and permits image-only prompts", async () => {
  const { m, calls } = setup();
  const r = await m.doGenerate({ prompt: prompt({ ...image(), filename: "cat.png" }) });
  expect(calls[0].attachments).toEqual([{
    type: "upload", contentType: "image/png", name: "cat.png",
    dataUrl: "data:image/png;base64,AQID", size: 3, id: expect.any(String),
  }]);
  expect(calls[0].userPrompt).toBeUndefined();
  expect(r.warnings).toEqual([]);
  expect(JSON.stringify(r.request)).not.toContain("AQID");
  expect(calls[0].attachments[0].dataUrl).toContain("AQID");
});

it.each(["image/png", "image/jpeg", "image/gif", "image/webp"])("accepts base64 %s", async mediaType => {
  const { m, calls } = setup();
  await m.doGenerate({ prompt: prompt(image("AQID", mediaType)) });
  expect(calls[0].attachments[0]).toMatchObject({ contentType: mediaType, size: 3, dataUrl: "data:" + mediaType + ";base64,AQID" });
});

it("sends multiple ordered images alongside text with unique ids", async () => {
  const { m, calls } = setup();
  const p = prompt(image(), image("BA=="));
  p[0].role === "user" && p[0].content.unshift({ type: "text", text: "compare" });
  await m.doGenerate({ prompt: p });
  expect(calls[0].userPrompt).toBe("compare");
  expect(calls[0].attachments).toHaveLength(2);
  expect(calls[0].attachments[1].dataUrl).toContain("BA==");
  expect(calls[0].attachments[0].id).not.toBe(calls[0].attachments[1].id);
});

it("maps URL images to URL attachments", async () => {
  const { m, calls } = setup();
  await m.doGenerate({ prompt: prompt(image(new URL("https://example.com/cat.png"))) });
  expect(calls[0].attachments).toEqual([{ type: "url", value: "https://example.com/cat.png" }]);
});

it("passes image URLs through the actual AI SDK without downloading", async () => {
  const { m, calls } = setup();
  await generateText({ model: m, messages: [{ role: "user", content: [
    { type: "image", image: new URL("https://example.invalid/cat.png"), mediaType: "image/png" },
    { type: "image", image: new Uint8Array([1, 2, 3]), mediaType: "image/png" },
  ] }] });
  expect(calls[0].attachments).toHaveLength(2);
  expect(calls[0].attachments[0]).toEqual({ type: "url", value: "https://example.invalid/cat.png" });
});

it.each([
  image("AQID", "image/svg+xml"), image("AQID", "image/*"),
  image(new URL("file:///secret.png")), image("!invalid!"), image("A"),
  image(""), image(new Uint8Array()),
])("rejects invalid image MIME, URL or data before fetching (%j)", async part => {
  const { m, calls } = setup();
  await expect(m.doGenerate({ prompt: prompt(part) })).rejects.toThrow(/Builder image/);
  expect(calls).toHaveLength(0);
});

it.each(["bytes", "base64"])("enforces encoded size limit for %s", async kind => {
  const { m, calls } = setup();
  const bytes = new Uint8Array(3.75 * 1024 * 1024 + 1);
  await expect(m.doGenerate({ prompt: prompt(image(kind === "bytes" ? bytes : Buffer.from(bytes).toString("base64"))) }))
    .rejects.toThrow("5 MiB base64-encoded limit");
  expect(calls).toHaveLength(0);
});

it.each(["bytes", "base64"])("accepts exactly the encoded limit for %s", async kind => {
  const { m, calls } = setup();
  const bytes = new Uint8Array(3.75 * 1024 * 1024);
  await m.doGenerate({ prompt: prompt(image(kind === "bytes" ? bytes : Buffer.from(bytes).toString("base64"))) });
  expect(calls[0].attachments[0].size).toBe(3.75 * 1024 * 1024);
});

it.each([false, true])("does not resend old images on continuation (native %s)", async native => {
  const { m, calls } = setup({ native });
  const first = prompt(image());
  await m.doGenerate({ prompt: first });
  await m.doGenerate({ prompt: [...first, { role: "assistant", content: [{ type: "text", text: "seen" }] }, ...prompt(image("BA=="))] });
  expect(calls[1].attachments).toHaveLength(1);
  expect(calls[1].attachments[0].dataUrl).toContain("BA==");
  expect(JSON.stringify(calls[1])).not.toContain("AQID");
  expect(JSON.stringify(calls[1].systemReminders ?? [])).not.toContain('"data"');
});

it("omits historical images in a new-instance transcript and sends only fresh images", async () => {
  const { m, calls } = setup();
  await m.doGenerate({ prompt: [...prompt(image()), { role: "assistant", content: [{ type: "text", text: "seen" }] }, ...prompt(image("BA=="))] });
  expect(calls[0].attachments).toHaveLength(1);
  expect(JSON.stringify(calls[0])).not.toContain("AQID");
});

it.each([{ status: 400, echo: true }, { echo: true }, { network: true }])("redacts image data from errors %j", async options => {
  const { m } = setup(options);
  try {
    await m.doGenerate({ prompt: prompt(image()) });
    throw new Error("expected failure");
  } catch (error) {
    expect(String(error)).not.toContain("AQID");
    expect(JSON.stringify(error)).not.toContain("AQID");
    expect(String(error)).toContain("[redacted image data]");
  }
});

it("does not attach images or warn for text-only prompts", async () => {
  const { m, calls } = setup();
  const r = await m.doGenerate({ prompt: [{ role: "user", content: [{ type: "text", text: "hi" }] }] });
  expect(calls[0].attachments).toBeUndefined();
  expect(r.warnings).toEqual([]);
});
