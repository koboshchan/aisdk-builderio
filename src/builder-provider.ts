import {
  APICallError,
  InvalidResponseDataError,
  type JSONValue,
  type LanguageModelV3,
  type LanguageModelV3CallOptions,
  type LanguageModelV3Content,
  type LanguageModelV3FinishReason,
  type LanguageModelV3GenerateResult,
  type LanguageModelV3Prompt,
  type LanguageModelV3StreamPart,
  type LanguageModelV3StreamResult,
  type LanguageModelV3ToolResultOutput,
  type LanguageModelV3Usage,
  type SharedV3Warning,
} from "@ai-sdk/provider";
import { loadApiKey, withoutTrailingSlash } from "@ai-sdk/provider-utils";
import { randomUUID } from "node:crypto";
import os from "node:os";

export interface BuilderProviderSettings {
  baseURL?: string;
  apiKey: string;
  userId: string;
  privateKey?: string;
  /** Custom fetch, mainly for tests. */
  fetch?: typeof fetch;
}

export interface BuilderChatSettings {
  temperature?: number;
  maxOutputTokens?: number;
  topP?: number;
  /** Builder session to continue. Defaults to one session per model instance. */
  sessionId?: string;
}

export interface BuilderProvider {
  (modelId: string, settings?: BuilderChatSettings): BuilderChatLanguageModel;
  languageModel(modelId: string, settings?: BuilderChatSettings): BuilderChatLanguageModel;
}

interface BuilderConfig {
  baseURL: string;
  apiKey: string;
  userId: string;
  headers: () => Record<string, string>;
  fetch: typeof fetch;
}

const TOOL_SERVER = "ai-sdk";
const TOOL_PREFIX = "mcp__" + TOOL_SERVER + "__";

type BuilderEvent = Record<string, any> & { type?: string };

interface PreparedRequest {
  body: Record<string, any>;
  warnings: SharedV3Warning[];
  toolNames: Map<string, string>;
}

const EMPTY_USAGE: LanguageModelV3Usage = {
  inputTokens: { total: undefined, noCache: undefined, cacheRead: undefined, cacheWrite: undefined },
  outputTokens: { total: undefined, text: undefined, reasoning: undefined },
};

export function mapFinishReason(reason?: string | null): LanguageModelV3FinishReason {
  const raw = reason ?? undefined;
  switch (reason) {
    case "end_turn":
    case "stop_sequence":
      return { unified: "stop", raw };
    case "max_tokens":
    case "model_context_window_exceeded":
      return { unified: "length", raw };
    case "tool_use":
      return { unified: "tool-calls", raw };
    case "content_filter":
    case "refusal":
      return { unified: "content-filter", raw };
    case "error":
      return { unified: "error", raw };
    default:
      return { unified: "other", raw };
  }
}

function outputToText(output: LanguageModelV3ToolResultOutput): { content: string; isError: boolean } {
  switch (output.type) {
    case "text":
      return { content: output.value, isError: false };
    case "error-text":
      return { content: output.value, isError: true };
    case "json":
      return { content: JSON.stringify(output.value), isError: false };
    case "error-json":
      return { content: JSON.stringify(output.value), isError: true };
    case "execution-denied":
      return { content: output.reason ?? "Tool execution was denied.", isError: true };
    case "content":
      return {
        content: output.value
          .map((p) => (p.type === "text" ? p.text : "[" + p.type + " omitted]"))
          .join("\n"),
        isError: false,
      };
    default:
      return { content: "", isError: false };
  }
}

function textOf(parts: unknown): string {
  if (typeof parts === "string") return parts;
  if (!Array.isArray(parts)) return "";
  return parts
    .filter((p: any) => p?.type === "text")
    .map((p: any) => p.text as string)
    .join("");
}

/** Removes anything credential-shaped before text reaches an error message or log. */
export function redact(text: string, secrets: string[]): string {
  let out = text;
  for (const s of secrets) if (s) out = out.split(s).join("[redacted]");
  return out
    .replace(/(apiKey|api_key|token|authorization)=([^&\s"']+)/gi, "$1=[redacted]")
    .replace(/Bearer\s+[A-Za-z0-9._~+/=-]+/g, "Bearer [redacted]")
    .replace(/\bbpk-[A-Za-z0-9-]+/g, "[redacted]");
}

export class BuilderChatLanguageModel implements LanguageModelV3 {
  readonly specificationVersion = "v3";
  readonly provider = "builder";
  readonly modelId: string;

  private readonly settings: BuilderChatSettings;
  private readonly config: BuilderConfig;
  private readonly defaultSessionId = "ai-sdk-" + randomUUID();
  /** Sessions that already hold history on the Builder side. */
  private readonly knownSessions = new Set<string>();

  constructor(modelId: string, settings: BuilderChatSettings, config: BuilderConfig) {
    this.modelId = modelId;
    this.settings = settings;
    this.config = config;
  }

  get supportedUrls() {
    return {};
  }

  private secrets(): string[] {
    const auth = this.config.headers().Authorization ?? "";
    return [this.config.apiKey, auth.replace(/^Bearer\s+/, "")];
  }

  async doGenerate(options: LanguageModelV3CallOptions): Promise<LanguageModelV3GenerateResult> {
    const { body, warnings, toolNames } = await this.prepare(options);
    const response = await this.post(body, options.abortSignal, options.headers);
    const text = await response.text();

    const content: LanguageModelV3Content[] = [];
    let textBuf = "";
    let reasoningBuf = "";
    let finish: LanguageModelV3FinishReason | undefined;
    let sawToolCall = false;
    let streamError: BuilderEvent | undefined;

    const flush = () => {
      if (reasoningBuf) content.push({ type: "reasoning", text: reasoningBuf });
      if (textBuf) content.push({ type: "text", text: textBuf });
      reasoningBuf = "";
      textBuf = "";
    };

    for (const line of text.split("\n")) {
      const event = parseLine(line);
      if (!event) continue;
      switch (event.type) {
        case "thinking":
          if (event.content) reasoningBuf += event.content;
          break;
        case "delta":
        case "text":
          if (event.content) textBuf += event.content;
          break;
        case "tool":
          flush();
          content.push(this.toolCall(event, toolNames));
          sawToolCall = true;
          break;
        case "error":
          streamError = event;
          break;
        case "done":
          finish = mapFinishReason(event.stopReason);
          break;
      }
    }
    flush();

    if (streamError && !finish) throw this.eventError(streamError);
    if (sawToolCall && finish?.unified !== "tool-calls") {
      finish = { unified: "tool-calls", raw: finish?.raw };
    }
    this.knownSessions.add(body.sessionId);

    return {
      content,
      finishReason: finish ?? { unified: "other", raw: undefined },
      usage: EMPTY_USAGE,
      request: { body: this.safeBody(body) },
      response: { body: redact(text, this.secrets()) },
      warnings,
    };
  }

  async doStream(options: LanguageModelV3CallOptions): Promise<LanguageModelV3StreamResult> {
    const { body, warnings, toolNames } = await this.prepare(options);
    const response = await this.post(body, options.abortSignal, options.headers);
    if (!response.body) {
      throw new InvalidResponseDataError({ data: undefined, message: "Builder returned an empty response body" });
    }
    this.knownSessions.add(body.sessionId);

    const stream = response.body
      .pipeThrough(new TextDecoderStream())
      .pipeThrough(this.createJsonLineParser())
      .pipeThrough(this.createStreamTransformer(warnings, toolNames));

    return { stream, request: { body: this.safeBody(body) } };
  }

  private async prepare(options: LanguageModelV3CallOptions): Promise<PreparedRequest> {
    const warnings: SharedV3Warning[] = [];
    const providerOpts = (options.providerOptions?.builder ?? {}) as Record<string, JSONValue>;
    const sessionId =
      (typeof providerOpts.sessionId === "string" && providerOpts.sessionId) ||
      this.settings.sessionId ||
      this.defaultSessionId;

    const { userPrompt, toolResults, systemReminders } = this.convertPrompt(
      options.prompt,
      this.knownSessions.has(sessionId),
    );

    if (!userPrompt && toolResults.length === 0) {
      throw new InvalidResponseDataError({
        data: undefined,
        message: "Builder request needs a user message or tool results",
      });
    }

    for (const feature of ["temperature", "topP", "topK", "stopSequences", "seed", "presencePenalty", "frequencyPenalty"] as const) {
      if (options[feature] !== undefined) {
        warnings.push({ type: "unsupported", feature, details: "Builder's completion API ignores this setting." });
      }
    }
    if (options.responseFormat && options.responseFormat.type !== "text") {
      warnings.push({ type: "unsupported", feature: "responseFormat" });
    }

    const toolNames = new Map<string, string>();
    const localMCPTools: Record<string, unknown>[] = [];
    for (const tool of options.tools ?? []) {
      if (tool.type !== "function") {
        warnings.push({ type: "unsupported", feature: "tool " + tool.name, details: "Only function tools are supported." });
        continue;
      }
      const wire = TOOL_PREFIX + tool.name.replace(/[^a-zA-Z0-9_-]/g, "_");
      toolNames.set(wire, tool.name);
      localMCPTools.push({
        name: wire,
        description: tool.description,
        inputSchema: tool.inputSchema,
        serverName: TOOL_SERVER,
      });
    }
    if (options.toolChoice && options.toolChoice.type !== "auto" && localMCPTools.length) {
      warnings.push({ type: "unsupported", feature: "toolChoice", details: "Builder always lets the model pick tools." });
    }

    const body: Record<string, any> = {
      position: "cli",
      eventName: "cli code",
      sessionId,
      userPrompt: userPrompt || undefined,
      toolResults: toolResults.length ? toolResults : undefined,
      systemReminders: systemReminders.length ? systemReminders : undefined,
      codeGenMode: "quality-v4",
      userContext: this.getUserContext(),
      maxTokens: options.maxOutputTokens ?? this.settings.maxOutputTokens,
      // Builder-side tools run on the client in the CLI, so only caller-provided tools are exposed.
      enabledTools: [],
      localMCPTools: localMCPTools.length ? localMCPTools : undefined,
      isLocal: false,
    };
    if (typeof providerOpts.modelOverride === "string") body.modelOverride = providerOpts.modelOverride;

    return { body, warnings, toolNames };
  }

  /**
   * Builder keeps history server side per session, so only the part of the prompt after the last
   * assistant message is new. When the session is unknown to this instance but the prompt already
   * has earlier turns, they are replayed as a transcript so no context is lost.
   */
  private convertPrompt(prompt: LanguageModelV3Prompt, sessionKnown: boolean) {
    const systemReminders: string[] = [];
    const turns: LanguageModelV3Prompt = [];
    for (const m of prompt) {
      if (m.role === "system") systemReminders.push(m.content);
      else turns.push(m);
    }

    let lastAssistant = -1;
    turns.forEach((m, i) => {
      if (m.role === "assistant") lastAssistant = i;
    });

    const fresh = turns.slice(lastAssistant + 1);
    const earlier = turns.slice(0, lastAssistant + 1);

    const userTexts: string[] = [];
    const toolResults: Record<string, unknown>[] = [];
    for (const m of fresh) {
      if (m.role === "user") {
        const t = textOf(m.content);
        if (t) userTexts.push(t);
      } else if (m.role === "tool") {
        for (const part of m.content) {
          if (part.type !== "tool-result") continue;
          const { content, isError } = outputToText(part.output);
          toolResults.push({
            type: "tool_result",
            tool_use_id: part.toolCallId,
            tool_name: TOOL_PREFIX + part.toolName.replace(/[^a-zA-Z0-9_-]/g, "_"),
            content,
            is_error: isError || undefined,
          });
        }
      }
    }

    let userPrompt = userTexts.join("\n\n");
    if (!sessionKnown && earlier.length > 0) {
      const transcript = earlier
        .map((m) => {
          if (m.role === "user") return "User: " + textOf(m.content);
          if (m.role === "assistant") {
            const parts = m.content.map((p) => {
              if (p.type === "text") return p.text;
              if (p.type === "tool-call") return "[called " + p.toolName + " " + JSON.stringify(p.input) + "]";
              return "";
            });
            return "Assistant: " + parts.filter(Boolean).join(" ");
          }
          if (m.role === "tool") {
            return m.content
              .filter((p) => p.type === "tool-result")
              .map((p: any) => "Tool " + p.toolName + " result: " + outputToText(p.output).content)
              .join("\n");
          }
          return "";
        })
        .filter(Boolean)
        .join("\n");
      userPrompt = "Earlier conversation:\n" + transcript + (userPrompt ? "\n\nCurrent message:\n" + userPrompt : "");
      // Tool results from the replayed part have no live tool_use on the server; send only the new ones.
    }
    return { userPrompt, toolResults, systemReminders };
  }

  private async post(body: Record<string, any>, signal?: AbortSignal, extraHeaders?: Record<string, string | undefined>) {
    const url = this.buildUrl();
    const headers: Record<string, string> = { ...this.config.headers() };
    for (const [k, v] of Object.entries(extraHeaders ?? {})) if (v !== undefined) headers[k] = v;

    let response: Response;
    try {
      response = await this.config.fetch(url, { method: "POST", headers, body: JSON.stringify(body), signal });
    } catch (error) {
      if ((error as any)?.name === "AbortError") throw error;
      throw new APICallError({
        message: redact("Cannot connect to Builder: " + (error instanceof Error ? error.message : String(error)), this.secrets()),
        url: this.publicUrl(),
        requestBodyValues: this.safeBody(body),
        isRetryable: true,
      });
    }

    if (!response.ok) {
      const text = redact(await response.text().catch(() => ""), this.secrets());
      let message = "Builder request failed with status " + response.status;
      let code: string | undefined;
      const first = parseLine(text.split("\n")[0] ?? "");
      if (first?.type === "error") {
        code = typeof first.code === "string" ? first.code : undefined;
        if (first.message) message += ": " + first.message;
      }
      throw new APICallError({
        message,
        url: this.publicUrl(),
        requestBodyValues: this.safeBody(body),
        statusCode: response.status,
        responseHeaders: Object.fromEntries(response.headers.entries()),
        responseBody: text,
        isRetryable: response.status === 408 || response.status === 409 || response.status === 429 || response.status >= 500,
        data: code ? { code } : undefined,
      });
    }
    return response;
  }

  private buildUrl(): string {
    const baseURL = withoutTrailingSlash(this.config.baseURL);
    const q = new URLSearchParams({ apiKey: this.config.apiKey, userId: this.config.userId });
    return baseURL + "/codegen/completion?" + q.toString();
  }

  /** Same URL without the key, for errors and logs. */
  private publicUrl(): string {
    return withoutTrailingSlash(this.config.baseURL) + "/codegen/completion";
  }

  private safeBody(body: Record<string, any>) {
    return { ...body, userContext: undefined };
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

  private toolCall(event: BuilderEvent, toolNames: Map<string, string>) {
    const wire = String(event.name ?? "");
    let input = typeof event.content === "string" && event.content.trim() ? event.content : "{}";
    try {
      JSON.parse(input);
    } catch {
      throw new InvalidResponseDataError({
        data: event,
        message: "Builder sent invalid JSON arguments for tool " + wire,
      });
    }
    return {
      type: "tool-call" as const,
      toolCallId: String(event.id ?? "call-" + randomUUID()),
      toolName: toolNames.get(wire) ?? (wire.startsWith(TOOL_PREFIX) ? wire.slice(TOOL_PREFIX.length) : wire),
      input,
    };
  }

  private eventError(event: BuilderEvent): Error {
    return new APICallError({
      message: redact(
        "Builder error" + (event.code ? " (" + event.code + ")" : "") + ": " + (event.message ?? "unknown error"),
        this.secrets(),
      ),
      url: this.publicUrl(),
      requestBodyValues: {},
      isRetryable: false,
      data: { code: event.code, stopReason: event.stopReason },
    });
  }

  private createJsonLineParser(): TransformStream<string, BuilderEvent> {
    let buffer = "";
    return new TransformStream({
      transform(chunk, controller) {
        buffer += chunk;
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";
        for (const line of lines) {
          const event = parseLine(line);
          if (event) controller.enqueue(event);
        }
      },
      flush(controller) {
        const event = parseLine(buffer);
        if (event) controller.enqueue(event);
      },
    });
  }

  private createStreamTransformer(
    warnings: SharedV3Warning[],
    toolNames: Map<string, string>,
  ): TransformStream<BuilderEvent, LanguageModelV3StreamPart> {
    let started = false;
    let textId: string | undefined;
    let reasoningId: string | undefined;
    let sawToolCall = false;
    let finished = false;
    const self = this;

    const closeBlocks = (controller: TransformStreamDefaultController<LanguageModelV3StreamPart>) => {
      if (reasoningId) controller.enqueue({ type: "reasoning-end", id: reasoningId });
      if (textId) controller.enqueue({ type: "text-end", id: textId });
      reasoningId = undefined;
      textId = undefined;
    };

    return new TransformStream({
      start(controller) {
        controller.enqueue({ type: "stream-start", warnings });
        started = true;
      },
      transform(event, controller) {
        switch (event.type) {
          case "thinking":
            if (event.content) {
              if (!reasoningId) {
                reasoningId = "reasoning-" + randomUUID();
                controller.enqueue({ type: "reasoning-start", id: reasoningId });
              }
              controller.enqueue({ type: "reasoning-delta", id: reasoningId, delta: event.content });
            }
            break;
          case "delta":
          case "text":
            if (event.content) {
              if (reasoningId) {
                controller.enqueue({ type: "reasoning-end", id: reasoningId });
                reasoningId = undefined;
              }
              if (!textId) {
                textId = "text-" + randomUUID();
                controller.enqueue({ type: "text-start", id: textId });
              }
              controller.enqueue({ type: "text-delta", id: textId, delta: event.content });
            }
            break;
          case "tool": {
            closeBlocks(controller);
            try {
              controller.enqueue(self.toolCall(event, toolNames));
              sawToolCall = true;
            } catch (error) {
              controller.enqueue({ type: "error", error });
            }
            break;
          }
          case "error":
            closeBlocks(controller);
            controller.enqueue({ type: "error", error: self.eventError(event) });
            if (!finished) {
              finished = true;
              controller.enqueue({ type: "finish", finishReason: { unified: "error", raw: event.code }, usage: EMPTY_USAGE });
            }
            break;
          case "done": {
            if (finished) break;
            closeBlocks(controller);
            let finishReason = mapFinishReason(event.stopReason);
            if (sawToolCall && finishReason.unified !== "tool-calls") {
              finishReason = { unified: "tool-calls", raw: finishReason.raw };
            }
            finished = true;
            controller.enqueue({
              type: "finish",
              finishReason,
              usage: EMPTY_USAGE,
              providerMetadata: {
                builder: {
                  creditsUsed: typeof event.creditsUsed === "number" ? event.creditsUsed : null,
                  model: typeof event.model === "string" ? event.model : null,
                },
              },
            });
            break;
          }
        }
      },
      flush(controller) {
        if (!started) controller.enqueue({ type: "stream-start", warnings });
        closeBlocks(controller);
        if (!finished) {
          controller.enqueue({
            type: "error",
            error: new InvalidResponseDataError({ data: undefined, message: "Builder stream ended without a done event" }),
          });
          controller.enqueue({ type: "finish", finishReason: { unified: "error", raw: "truncated" }, usage: EMPTY_USAGE });
        }
      },
    });
  }
}

function parseLine(line: string): BuilderEvent | undefined {
  const trimmed = line.trim();
  if (!trimmed) return undefined;
  try {
    const value = JSON.parse(trimmed);
    return value && typeof value === "object" ? value : undefined;
  } catch {
    return undefined;
  }
}

export function createBuilder(options: BuilderProviderSettings): BuilderProvider {
  const baseURL = withoutTrailingSlash(options.baseURL) ?? "https://api.builder.io";
  const privateKey = loadApiKey({
    apiKey: options.privateKey,
    environmentVariableName: "BUILDER_PRIVATE_KEY",
    description: "Builder.io private key",
  });

  const createModel = (modelId: string, settings: BuilderChatSettings = {}) =>
    new BuilderChatLanguageModel(modelId, settings, {
      baseURL,
      apiKey: options.apiKey,
      userId: options.userId,
      fetch: options.fetch ?? globalThis.fetch,
      headers: () => ({
        Authorization: "Bearer " + privateKey,
        "Content-Type": "application/json",
      }),
    });

  const provider = function (modelId: string, settings?: BuilderChatSettings) {
    return createModel(modelId, settings);
  };
  provider.languageModel = createModel;
  return provider as BuilderProvider;
}
