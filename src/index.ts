import {
  LanguageModelV3,
  LanguageModelV3CallOptions,
  LanguageModelV3Content,
  LanguageModelV3StreamPart,
  LanguageModelV3FinishReason,
  LanguageModelV3GenerateResult,
  LanguageModelV3StreamResult,
  LanguageModelV3Usage,
  APICallError,
} from "@ai-sdk/provider";
import { loadApiKey, withoutTrailingSlash } from "@ai-sdk/provider-utils";
import os from "node:os";
import { randomUUID } from "node:crypto";

export interface BuilderProviderSettings {
  baseURL?: string;
  apiKey: string;
  userId: string;
  privateKey?: string;
  fetch?: typeof fetch;
}

export interface BuilderChatSettings {
  temperature?: number;
  maxOutputTokens?: number;
  topP?: number;
}

export interface BuilderProvider {
  (modelId: string, settings?: BuilderChatSettings): BuilderChatLanguageModel;
  languageModel(modelId: string, settings?: BuilderChatSettings): BuilderChatLanguageModel;
}

type FR = LanguageModelV3FinishReason;

export function mapFinishReason(reason?: string | null): FR {
  const raw = reason ?? undefined;
  switch (reason) {
    case "end_turn":
    case "stop":
    case "stop_sequence": return { unified: "stop", raw };
    case "max_tokens": return { unified: "length", raw };
    case "tool_use": return { unified: "tool-calls", raw };
    case "content_filter": return { unified: "content-filter", raw };
    case "error": return { unified: "error", raw };
    default: return { unified: "other", raw };
  }
}

/** Remove credentials from any string that may reach logs or errors. */
export function redact(text: string, secrets: string[]): string {
  let out = text;
  for (const s of secrets) if (s) out = out.split(s).join("[redacted]");
  return out
    .replace(/(Bearer\s+)[^\s"']+/gi, "$1[redacted]")
    .replace(/(apiKey|api_key|privateKey|token)(=|":\s*")[^&\s"']+/gi, "$1$2[redacted]");
}

const usage = (outChars: number, reasoning?: number) => ({
  inputTokens: { total: undefined, noCache: undefined, cacheRead: undefined, cacheWrite: undefined },
  outputTokens: { total: outChars, text: outChars, reasoning },
});

const stringify = (v: unknown) => (typeof v === "string" ? v : JSON.stringify(v ?? null));

function outputToString(o: any): string {
  if (o && typeof o === "object" && "type" in o) {
    if (o.type === "text" || o.type === "error-text") return o.value;
    if (o.type === "json" || o.type === "error-json") return JSON.stringify(o.value);
    if (o.type === "content") return JSON.stringify(o.value);
  }
  return stringify(o);
}

export class BuilderChatLanguageModel implements LanguageModelV3 {
  readonly specificationVersion = "v3";
  readonly provider = "builder";
  readonly modelId: string;

  private readonly settings: BuilderChatSettings;
  private readonly config: BuilderConfig;
  private sessionId = "session-" + randomUUID();
  private sentCalls = 0;

  constructor(modelId: string, settings: BuilderChatSettings, config: BuilderConfig) {
    this.modelId = modelId;
    this.settings = settings;
    this.config = config;
  }

  get supportedUrls() {
    return {};
  }

  private secrets() {
    return [this.config.privateKey, this.config.apiKey];
  }

  private buildBody(options: LanguageModelV3CallOptions) {
    const { userPrompt, toolResults } = this.convertMessages(options.prompt);
    const body: any = {
      position: "cli",
      sessionId: (options.providerOptions?.builder?.sessionId as string) || this.sessionId,
      userPrompt: toolResults.length ? undefined : userPrompt,
      toolResults,
      codeGenMode: "quality-v4",
      userContext: this.getUserContext(),
      maxTokens: options.maxOutputTokens ?? this.settings.maxOutputTokens,
    };
    if (options.tools?.length) {
      body.enabledTools = options.tools.filter((t) => t.type === "function").map((t) => t.name);
    }
    return body;
  }

  private async send(options: LanguageModelV3CallOptions) {
    const body = this.buildBody(options);
    const url = this.buildUrl();
    const safeUrl = url.replace(/([?&](?:apiKey|userId)=)[^&]*/g, "$1[redacted]");
    let response: Response;
    try {
      response = await this.config.fetch(url, {
        method: "POST",
        headers: this.config.headers(),
        body: JSON.stringify(body),
        signal: options.abortSignal,
      });
    } catch (e: any) {
      throw new APICallError({
        message: redact(String(e?.message ?? e), this.secrets()),
        url: safeUrl,
        requestBodyValues: {},
        isRetryable: e?.name !== "AbortError",
      });
    }
    if (!response.ok) {
      const text = redact(await response.text().catch(() => ""), this.secrets()).slice(0, 2000);
      throw new APICallError({
        statusCode: response.status,
        message: "Builder request failed with status " + response.status + (text ? ": " + text : ""),
        url: safeUrl,
        requestBodyValues: {},
        responseBody: text,
        isRetryable: response.status === 429 || response.status >= 500,
      });
    }
    this.sentCalls++;
    return { response, body };
  }

  async doGenerate(options: LanguageModelV3CallOptions): Promise<LanguageModelV3GenerateResult> {
    const { response, body } = await this.send(options);
    const text = await response.text();
    let fullText = "";
    let reasoning = "";
    const content: LanguageModelV3Content[] = [];
    let finish: FR | undefined;
    let error: any;
    for (const line of text.split("\n")) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      let ev: any;
      try { ev = JSON.parse(trimmed); } catch { continue; }
      if ((ev.type === "delta" || ev.type === "text") && ev.content) fullText += ev.content;
      else if (ev.type === "thinking" && ev.content) reasoning += ev.content;
      else if (ev.type === "tool") {
        content.push({
          type: "tool-call",
          toolCallId: ev.id || "tool-" + randomUUID(),
          toolName: ev.name,
          input: ev.content ? stringify(ev.content) : "{}",
        });
      } else if (ev.type === "error") error = ev;
      else if (ev.type === "done") finish = mapFinishReason(ev.stopReason);
    }
    if (!finish) {
      throw new Error(redact(error ? "Builder error " + error.code + ": " + error.message : "Builder stream ended without a done event", this.secrets()));
    }
    const out: LanguageModelV3Content[] = [];
    if (reasoning) out.push({ type: "reasoning", text: reasoning });
    if (fullText) out.push({ type: "text", text: fullText });
    out.push(...content);
    if (content.length && finish.unified !== "tool-calls") finish = { unified: "tool-calls", raw: finish.raw };
    return {
      content: out,
      finishReason: finish,
      usage: usage(fullText.length, reasoning.length || undefined),
      request: { body: { ...body, userContext: undefined } },
      response: { body: redact(text, this.secrets()) },
      warnings: [],
    };
  }

  async doStream(options: LanguageModelV3CallOptions): Promise<LanguageModelV3StreamResult> {
    const { response } = await this.send(options);
    const stream = response.body!
      .pipeThrough(new TextDecoderStream())
      .pipeThrough(this.createLineParser())
      .pipeThrough(this.createStreamTransformer());
    return { stream };
  }

  private buildUrl(): string {
    const baseURL = withoutTrailingSlash(this.config.baseURL);
    return baseURL + "/codegen/completion?apiKey=" + encodeURIComponent(this.config.apiKey) + "&userId=" + encodeURIComponent(this.config.userId);
  }

  private convertMessages(prompt: LanguageModelV3CallOptions["prompt"]): { userPrompt: string; toolResults: any[] } {
    const textOf = (c: any) => (typeof c === "string" ? c : c.filter((p: any) => p.type === "text").map((p: any) => p.text).join(""));
    const toolResults: any[] = [];
    const turns: { role: string; text: string }[] = [];
    let lastUserIdx = -1;
    for (const msg of prompt) {
      if (msg.role === "system") continue;
      if (msg.role === "user") { lastUserIdx = turns.length; turns.push({ role: "user", text: textOf(msg.content) }); }
      else if (msg.role === "assistant") {
        const t = textOf(msg.content);
        if (t) turns.push({ role: "assistant", text: t });
      } else if (msg.role === "tool") {
        toolResults.length = 0; // only the latest tool round is pending
        for (const part of msg.content) {
          if (part.type === "tool-result") {
            const o: any = part.output;
            toolResults.push({
              type: "tool_result",
              tool_use_id: part.toolCallId,
              tool_name: part.toolName,
              content: outputToString(o),
              is_error: o?.type === "error-text" || o?.type === "error-json",
            });
          }
        }
      }
    }
    const system = prompt.filter((m) => m.role === "system").map((m: any) => m.content).join("\n");
    let userPrompt: string;
    if (this.sentCalls > 0 || turns.length <= 1) {
      // server already holds earlier turns for this session
      userPrompt = lastUserIdx >= 0 ? turns[lastUserIdx].text : "";
    } else {
      // fresh session but earlier history supplied: replay it in the prompt
      const prior = turns.slice(0, lastUserIdx).map((t) => t.role + ": " + t.text).join("\n");
      userPrompt = "Conversation so far:\n" + prior + "\n\nuser: " + turns[lastUserIdx].text;
    }
    if (system) userPrompt = system + "\n\n" + userPrompt;
    return { userPrompt, toolResults };
  }

  private getUserContext() {
    return {
      client: "@builder.io/ai-sdk-provider",
      clientVersion: "1.0.0",
      nodeVersion: process.version,
      systemPlatform: process.platform,
      systemEOL: os.EOL,
      systemArch: os.arch(),
    };
  }

  private createLineParser(): TransformStream<string, any> {
    let buffer = "";
    const parse = (line: string, c: TransformStreamDefaultController<any>) => {
      const t = line.trim();
      if (!t) return;
      try { c.enqueue(JSON.parse(t)); } catch { /* skip malformed line */ }
    };
    return new TransformStream({
      transform(chunk, c) {
        buffer += chunk;
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";
        for (const l of lines) parse(l, c);
      },
      flush(c) { parse(buffer, c); },
    });
  }

  private createStreamTransformer(): TransformStream<any, LanguageModelV3StreamPart> {
    let textId: string | undefined;
    let reasoningId: string | undefined;
    let textLen = 0;
    let reasoningLen = 0;
    let sawTool = false;
    let finished = false;
    const secrets = this.secrets();

    const closeBlocks = (c: TransformStreamDefaultController<LanguageModelV3StreamPart>) => {
      if (reasoningId) { c.enqueue({ type: "reasoning-end", id: reasoningId }); reasoningId = undefined; }
      if (textId) { c.enqueue({ type: "text-end", id: textId }); textId = undefined; }
    };
    const finish = (c: TransformStreamDefaultController<LanguageModelV3StreamPart>, fr: FR) => {
      if (finished) return;
      finished = true;
      closeBlocks(c);
      c.enqueue({ type: "finish", finishReason: fr, usage: usage(textLen, reasoningLen || undefined) });
    };

    return new TransformStream({
      start(c) { c.enqueue({ type: "stream-start", warnings: [] }); },
      transform(chunk, c) {
        if (finished) return;
        switch (chunk.type) {
          case "thinking":
            if (chunk.content) {
              if (!reasoningId) { reasoningId = "reasoning-0"; c.enqueue({ type: "reasoning-start", id: reasoningId }); }
              reasoningLen += chunk.content.length;
              c.enqueue({ type: "reasoning-delta", id: reasoningId, delta: chunk.content });
            }
            break;
          case "delta":
          case "text":
            if (chunk.content) {
              if (reasoningId) { c.enqueue({ type: "reasoning-end", id: reasoningId }); reasoningId = undefined; }
              if (!textId) { textId = "text-0"; c.enqueue({ type: "text-start", id: textId }); }
              textLen += chunk.content.length;
              c.enqueue({ type: "text-delta", id: textId, delta: chunk.content });
            }
            break;
          case "tool":
            closeBlocks(c);
            sawTool = true;
            c.enqueue({
              type: "tool-call",
              toolCallId: chunk.id || "tool-" + randomUUID(),
              toolName: chunk.name,
              input: chunk.content ? stringify(chunk.content) : "{}",
            });
            break;
          case "error":
            c.enqueue({ type: "error", error: new Error(redact("Builder error " + (chunk.code ?? "unknown") + ": " + (chunk.message ?? ""), secrets)) });
            finish(c, { unified: "error", raw: chunk.code });
            break;
          case "done": {
            let fr = mapFinishReason(chunk.stopReason);
            if (sawTool && fr.unified === "stop") fr = { unified: "tool-calls", raw: fr.raw };
            finish(c, fr);
            break;
          }
        }
      },
      flush(c) {
        if (!finished) {
          c.enqueue({ type: "error", error: new Error("Builder stream ended without a done event") });
          finish(c, { unified: "error", raw: undefined });
        }
      },
    });
  }
}

interface BuilderConfig {
  baseURL: string;
  apiKey: string;
  userId: string;
  headers: () => Record<string, string>;
  fetch: typeof fetch;
  privateKey: string;
}

export function createBuilder(options: BuilderProviderSettings): BuilderProvider {
  const baseURL = withoutTrailingSlash(options.baseURL) ?? "https://api.builder.io";
  const privateKey = options.privateKey ?? loadApiKey({
    apiKey: options.privateKey,
    environmentVariableName: "BUILDER_PRIVATE_KEY",
    description: "Builder.io private key",
  });

  const createModel = (modelId: string, settings: BuilderChatSettings = {}) =>
    new BuilderChatLanguageModel(modelId, settings, {
      baseURL,
      apiKey: options.apiKey,
      userId: options.userId,
      fetch: options.fetch ?? ((...a: Parameters<typeof fetch>) => fetch(...a)),
      privateKey,
      headers: () => ({
        Authorization: `Bearer ${privateKey}`,
        "Content-Type": "application/json",
      }),
    });

  const provider = function (modelId: string, settings?: BuilderChatSettings) {
    return createModel(modelId, settings);
  };
  provider.languageModel = createModel;
  return provider as BuilderProvider;
}
