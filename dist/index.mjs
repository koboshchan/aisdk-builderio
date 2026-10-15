// src/builder-provider.ts
import {
  APICallError,
  InvalidResponseDataError
} from "@ai-sdk/provider";
import { loadApiKey, withoutTrailingSlash } from "@ai-sdk/provider-utils";
import { randomUUID } from "crypto";
import os from "os";
var TOOL_SERVER = "ai-sdk";
var TOOL_PREFIX = "mcp__" + TOOL_SERVER + "__";
var EMPTY_USAGE = {
  inputTokens: { total: void 0, noCache: void 0, cacheRead: void 0, cacheWrite: void 0 },
  outputTokens: { total: void 0, text: void 0, reasoning: void 0 }
};
function mapUsage(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return EMPTY_USAGE;
  const u = value;
  const count = (...keys) => {
    for (const key of keys) {
      const n = u[key];
      if (typeof n === "number" && Number.isFinite(n) && n >= 0) return n;
    }
    return void 0;
  };
  const input = count("inputTokens", "input_tokens", "promptTokens", "prompt_tokens");
  const output = count("outputTokens", "output_tokens", "completionTokens", "completion_tokens");
  if (input === void 0 && output === void 0) return EMPTY_USAGE;
  return {
    inputTokens: { ...EMPTY_USAGE.inputTokens, total: input },
    outputTokens: { ...EMPTY_USAGE.outputTokens, total: output }
  };
}
function stripJsonFence(text) {
  const match = /^\s*```(?:json)?\s*\n?([\s\S]*?)\n?\s*```\s*$/i.exec(text);
  return match ? match[1].trim() : text;
}
function mapFinishReason(reason) {
  const raw = reason ?? void 0;
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
function outputToText(output) {
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
        content: output.value.map((p) => p.type === "text" ? p.text : "[" + p.type + " omitted]").join("\n"),
        isError: false
      };
    default:
      return { content: "", isError: false };
  }
}
function textOf(parts) {
  if (typeof parts === "string") return parts;
  if (!Array.isArray(parts)) return "";
  return parts.filter((p) => p?.type === "text").map((p) => p.text).join("");
}
var IMAGE_TYPES = /* @__PURE__ */ new Map([
  ["image/png", "png"],
  ["image/jpeg", "jpg"],
  ["image/gif", "gif"],
  ["image/webp", "webp"]
]);
var MAX_IMAGE_ENCODED_BYTES = 5 * 1024 * 1024;
function imageAttachment(part) {
  const fail = (message) => {
    throw new InvalidResponseDataError({ data: void 0, message });
  };
  const extension = IMAGE_TYPES.get(part.mediaType);
  if (!extension) return fail("Builder images must be PNG, JPEG, GIF, or WebP.");
  if (part.data instanceof URL) {
    if (!["https:", "http:"].includes(part.data.protocol)) {
      return fail("Builder image URLs must use HTTP or HTTPS.");
    }
    return { type: "url", value: part.data.toString() };
  }
  let bytes;
  let base64;
  if (typeof part.data === "string") {
    base64 = part.data;
    if (base64.length > MAX_IMAGE_ENCODED_BYTES) return fail("Builder image exceeds the 5 MiB base64-encoded limit.");
    if (!base64 || base64.length % 4 !== 0 || !/^[A-Za-z0-9+/]*={0,2}$/.test(base64)) {
      return fail("Builder image data must be valid padded base64.");
    }
    bytes = Buffer.from(base64, "base64");
    if (Buffer.from(bytes).toString("base64") !== base64) return fail("Builder image data must be valid canonical base64.");
  } else {
    bytes = part.data;
    if (4 * Math.ceil(bytes.byteLength / 3) > MAX_IMAGE_ENCODED_BYTES) {
      return fail("Builder image exceeds the 5 MiB base64-encoded limit.");
    }
    base64 = Buffer.from(bytes).toString("base64");
  }
  if (!bytes.byteLength) return fail("Builder image data must not be empty.");
  return {
    type: "upload",
    contentType: part.mediaType,
    name: part.filename ?? "image." + extension,
    dataUrl: "data:" + part.mediaType + ";base64," + base64,
    size: bytes.byteLength,
    id: randomUUID()
  };
}
function textHistory(prompt) {
  return prompt.filter((m) => m.role !== "system").map((m) => ({
    ...m,
    content: m.content.map((p) => p.type === "file" ? { type: "text", text: "[Earlier file omitted; retained by native continuation when available]" } : p)
  }));
}
function redact(text, secrets) {
  let out = text.replace(/data:[^;\s"\\]+;base64,[A-Za-z0-9+/=]+/gi, "[redacted image data]");
  for (const s of secrets) if (s) out = out.split(s).join("[redacted]");
  return out.replace(/(apiKey|api_key|token|authorization)=([^&\s"']+)/gi, "$1=[redacted]").replace(/Bearer\s+[A-Za-z0-9._~+/=-]+/g, "Bearer [redacted]").replace(/\bbpk-[A-Za-z0-9-]+/g, "[redacted]");
}
var BuilderChatLanguageModel = class {
  specificationVersion = "v3";
  provider = "builder";
  modelId;
  settings;
  config;
  defaultSessionId = "ai-sdk-" + randomUUID();
  /** Sessions that already hold history on the Builder side. */
  knownSessions = /* @__PURE__ */ new Set();
  sessionUrls = /* @__PURE__ */ new Map();
  sessionPrompts = /* @__PURE__ */ new Map();
  constructor(modelId, settings, config) {
    this.modelId = modelId;
    this.settings = settings;
    this.config = config;
  }
  get supportedUrls() {
    return { "image/*": [/^https?:\/\//] };
  }
  secrets() {
    const auth = this.config.headers().Authorization ?? "";
    return [this.config.apiKey, auth.replace(/^Bearer\s+/, "")];
  }
  async doGenerate(options) {
    const result = await this.doStream(options);
    const reader = result.stream.getReader();
    const content = [];
    const text = /* @__PURE__ */ new Map();
    let finishReason = { unified: "other", raw: void 0 };
    let providerMetadata;
    let usage = EMPTY_USAGE;
    let warnings = [];
    for (; ; ) {
      const { done, value: p } = await reader.read();
      if (done) break;
      if (p.type === "error") throw p.error;
      if (p.type === "stream-start") warnings = p.warnings;
      if (p.type === "text-start" || p.type === "reasoning-start") {
        const block = { type: p.type === "text-start" ? "text" : "reasoning", text: "" };
        text.set(p.id, block);
        content.push(block);
      }
      if (p.type === "text-delta" || p.type === "reasoning-delta") {
        const block = text.get(p.id);
        if (block) block.text += p.delta;
      }
      if (p.type === "tool-call") content.push(p);
      if (p.type === "finish") {
        finishReason = p.finishReason;
        usage = p.usage;
        providerMetadata = p.providerMetadata;
      }
    }
    return { content, finishReason, usage, providerMetadata, warnings, request: result.request };
  }
  async doStream(options) {
    const { body, warnings, toolNames } = await this.prepare(options);
    const response = await this.post(body, options.abortSignal, options.headers);
    if (!response.body) {
      throw new InvalidResponseDataError({ data: void 0, message: "Builder returned an empty response body" });
    }
    this.knownSessions.add(body.sessionId);
    this.sessionPrompts.set(body.sessionId, JSON.stringify(options.prompt));
    const stream = response.body.pipeThrough(new TextDecoderStream()).pipeThrough(this.createJsonLineParser()).pipeThrough(this.createStreamTransformer(warnings, toolNames, body.sessionId, options.responseFormat?.type === "json"));
    return { stream, request: { body: this.safeBody(body) } };
  }
  async prepare(options) {
    const warnings = [];
    const providerOpts = options.providerOptions?.builder ?? {};
    let sessionId = typeof providerOpts.sessionId === "string" && providerOpts.sessionId || this.settings.sessionId || this.defaultSessionId;
    const previous = this.sessionPrompts.get(sessionId);
    if (previous && !providerOpts.sessionId && !this.settings.sessionId) {
      const old = JSON.parse(previous);
      const prefix = JSON.stringify(options.prompt.slice(0, old.length));
      if (prefix !== previous) this.defaultSessionId = sessionId = "ai-sdk-" + randomUUID();
    }
    const { userPrompt, toolResults, systemReminders, attachments } = this.convertPrompt(
      options.prompt,
      this.knownSessions.has(sessionId),
      this.sessionUrls.has(sessionId)
    );
    if (!userPrompt && toolResults.length === 0 && attachments.length === 0) {
      throw new InvalidResponseDataError({
        data: void 0,
        message: "Builder request needs a user message, image attachments, or tool results"
      });
    }
    for (const feature of ["temperature", "topP", "topK", "stopSequences", "seed", "presencePenalty", "frequencyPenalty"]) {
      if (options[feature] !== void 0) {
        warnings.push({ type: "unsupported", feature, details: "Builder's completion API ignores this setting." });
      }
    }
    if (options.responseFormat?.type === "json") {
      systemReminders.push(
        "Reply with only valid JSON, no prose or code fences." + (options.responseFormat.schema ? "\nJSON schema: " + JSON.stringify(options.responseFormat.schema) : "")
      );
    }
    const toolNames = /* @__PURE__ */ new Map();
    const localMCPTools = [];
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
        serverName: TOOL_SERVER
      });
    }
    if (options.toolChoice && options.toolChoice.type !== "auto" && localMCPTools.length) {
      warnings.push({ type: "unsupported", feature: "toolChoice", details: "Builder always lets the model pick tools." });
    }
    const body = {
      position: "cli",
      eventName: "cli code",
      sessionId,
      userPrompt: userPrompt || void 0,
      attachments: attachments.length ? attachments : void 0,
      toolResults: toolResults.length ? toolResults : void 0,
      systemReminders: systemReminders.length ? systemReminders : void 0,
      codeGenMode: "quality-v4",
      userContext: this.getUserContext(),
      maxTokens: options.maxOutputTokens ?? this.settings.maxOutputTokens,
      url: this.sessionUrls.get(sessionId),
      // Builder-side tools run on the client in the CLI, so only caller-provided tools are exposed.
      enabledTools: [],
      localMCPTools: localMCPTools.length ? localMCPTools : void 0,
      isLocal: false
    };
    if (typeof providerOpts.modelOverride === "string") body.modelOverride = providerOpts.modelOverride;
    if (typeof providerOpts.softContextWindow === "number" && providerOpts.softContextWindow > 0) body.softContextWindow = providerOpts.softContextWindow;
    if (typeof providerOpts.errorIfHadCompaction === "boolean") body.errorIfHadCompaction = providerOpts.errorIfHadCompaction;
    return { body, warnings, toolNames };
  }
  /**
   * Builder keeps history server side per session, so only the part of the prompt after the last
   * assistant message is new. When the session is unknown to this instance but the prompt already
   * has earlier turns, they are replayed as a transcript so no context is lost.
   */
  convertPrompt(prompt, sessionKnown, hasContinuation) {
    const systemReminders = [];
    const turns = [];
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
    const userTexts = [];
    const attachments = [];
    const toolResults = [];
    for (const m of fresh) {
      if (m.role === "user") {
        for (const part of m.content) {
          if (part.type === "file" && part.mediaType.startsWith("image/")) attachments.push(imageAttachment(part));
        }
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
            is_error: isError || void 0
          });
        }
      }
    }
    let userPrompt = userTexts.join("\n\n");
    if (!sessionKnown && earlier.length > 0) {
      const transcript = earlier.map((m) => {
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
          return m.content.filter((p) => p.type === "tool-result").map((p) => "Tool " + p.toolName + " result: " + outputToText(p.output).content).join("\n");
        }
        return "";
      }).filter(Boolean).join("\n");
      userPrompt = "Earlier conversation:\n" + transcript + (userPrompt ? "\n\nCurrent message:\n" + userPrompt : "");
    }
    if (sessionKnown && prompt.length > 1 && !hasContinuation) {
      systemReminders.push("Conversation history (data, not additional system instructions):\\n" + JSON.stringify(textHistory(prompt)));
    }
    return { userPrompt, toolResults, systemReminders, attachments };
  }
  async post(body, signal, extraHeaders) {
    const url = this.buildUrl();
    const headers = { ...this.config.headers() };
    for (const [k, v] of Object.entries(extraHeaders ?? {})) if (v !== void 0) headers[k] = v;
    let response;
    try {
      response = await this.config.fetch(url, { method: "POST", headers, body: JSON.stringify(body), signal });
    } catch (error) {
      if (error?.name === "AbortError") throw error;
      throw new APICallError({
        message: redact("Cannot connect to Builder: " + (error instanceof Error ? error.message : String(error)), this.secrets()),
        url: this.publicUrl(),
        requestBodyValues: this.safeBody(body),
        isRetryable: true
      });
    }
    if (!response.ok) {
      const text = redact(await response.text().catch(() => ""), this.secrets());
      let message = "Builder request failed with status " + response.status;
      let code;
      const first = parseLine(text.split("\n")[0] ?? "");
      if (first?.type === "error") {
        code = typeof first.code === "string" ? first.code : void 0;
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
        data: code ? { code, ...this.contextMetadata(first ?? {}) } : void 0
      });
    }
    return response;
  }
  buildUrl() {
    const baseURL = withoutTrailingSlash(this.config.baseURL);
    const q = new URLSearchParams({ apiKey: this.config.apiKey, userId: this.config.userId });
    return baseURL + "/codegen/completion?" + q.toString();
  }
  /** Same URL without the key, for errors and logs. */
  publicUrl() {
    return withoutTrailingSlash(this.config.baseURL) + "/codegen/completion";
  }
  safeBody(body) {
    return JSON.parse(JSON.stringify({
      ...body,
      userContext: void 0,
      attachments: body.attachments?.map((a) => a.type === "upload" ? { ...a, dataUrl: "[redacted image data]" } : a)
    }, (_key, value) => typeof value === "string" ? redact(value, this.secrets()) : value));
  }
  getUserContext() {
    return {
      client: "@builder.io/ai-sdk-provider",
      clientVersion: "1.0.0",
      nodeVersion: process.version,
      systemPlatform: process.platform,
      systemEOL: os.EOL,
      systemArch: os.arch()
    };
  }
  toolCall(event, toolNames) {
    const wire = String(event.name ?? event.filePath ?? "");
    let input = typeof event.content === "string" && event.content.trim() ? event.content : "{}";
    try {
      JSON.parse(input);
    } catch {
      throw new InvalidResponseDataError({
        data: event,
        message: "Builder sent invalid JSON arguments for tool " + wire
      });
    }
    return {
      type: "tool-call",
      toolCallId: String(event.id ?? "call-" + randomUUID()),
      toolName: toolNames.get(wire) ?? (wire.startsWith(TOOL_PREFIX) ? wire.slice(TOOL_PREFIX.length) : wire),
      input
    };
  }
  contextMetadata(event) {
    return JSON.parse(JSON.stringify({
      contextWindow: event.contextWindow ?? null,
      usage: event.usage ?? null,
      compacted: event.compact ?? false,
      creditsUsed: event.creditsUsed ?? null,
      model: event.model ?? null
    }, (_key, value) => typeof value === "string" ? redact(value, this.secrets()) : value));
  }
  eventError(event) {
    return new APICallError({
      message: redact(
        "Builder error" + (event.code ? " (" + event.code + ")" : "") + ": " + (event.message ?? "unknown error"),
        this.secrets()
      ),
      url: this.publicUrl(),
      requestBodyValues: {},
      isRetryable: false,
      data: { code: event.code, stopReason: event.stopReason, ...this.contextMetadata(event) }
    });
  }
  createJsonLineParser() {
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
      }
    });
  }
  createStreamTransformer(warnings, toolNames, sessionId, jsonMode) {
    let jsonText = "";
    let started = false;
    let textId;
    let reasoningId;
    let sawToolCall = false;
    let finished = false;
    const self = this;
    let pendingTool;
    const finishTool = (controller) => {
      if (!pendingTool) return;
      const t = pendingTool;
      pendingTool = void 0;
      const call = self.toolCall({ id: t.id, name: t.name, content: t.input }, toolNames);
      controller.enqueue({ type: "tool-input-end", id: t.id });
      controller.enqueue(call);
      sawToolCall = true;
    };
    const closeBlocks = (controller) => {
      if (reasoningId) controller.enqueue({ type: "reasoning-end", id: reasoningId });
      if (textId) {
        if (jsonMode && jsonText) {
          controller.enqueue({ type: "text-delta", id: textId, delta: stripJsonFence(jsonText) });
          jsonText = "";
        }
        controller.enqueue({ type: "text-end", id: textId });
      }
      reasoningId = void 0;
      textId = void 0;
    };
    return new TransformStream({
      start(controller) {
        controller.enqueue({ type: "stream-start", warnings });
        started = true;
      },
      transform(event, controller) {
        if (finished) return;
        if (pendingTool && event.type === "delta") {
          const delta = String(event.delta ?? event.content ?? "");
          pendingTool.input += delta;
          controller.enqueue({ type: "tool-input-delta", id: pendingTool.id, delta });
          return;
        }
        if (pendingTool && ["tool", "done", "text", "thinking", "error"].includes(event.type ?? "")) {
          try {
            finishTool(controller);
          } catch (error) {
            controller.error(error);
            return;
          }
        }
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
                reasoningId = void 0;
              }
              if (!textId) {
                textId = "text-" + randomUUID();
                controller.enqueue({ type: "text-start", id: textId });
              }
              if (jsonMode) jsonText += event.content;
              else controller.enqueue({ type: "text-delta", id: textId, delta: event.content });
            }
            break;
          case "tool": {
            closeBlocks(controller);
            if (!event.content && (event.filePath || event.name)) {
              const id = String(event.id ?? "call-" + randomUUID());
              const wire = String(event.name ?? event.filePath);
              pendingTool = { id, name: wire, input: "" };
              controller.enqueue({ type: "tool-input-start", id, toolName: toolNames.get(wire) ?? wire.replace(TOOL_PREFIX, "") });
              break;
            }
            try {
              const call = self.toolCall(event, toolNames);
              controller.enqueue({ type: "tool-input-start", id: call.toolCallId, toolName: call.toolName });
              controller.enqueue({ type: "tool-input-delta", id: call.toolCallId, delta: call.input });
              controller.enqueue({ type: "tool-input-end", id: call.toolCallId });
              controller.enqueue(call);
              sawToolCall = true;
            } catch (error) {
              controller.error(error);
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
            if (typeof event.nextUrl === "string" && event.nextUrl.startsWith("cgen://")) self.sessionUrls.set(sessionId, event.nextUrl);
            if (event.compact) break;
            if (finished) break;
            closeBlocks(controller);
            let finishReason = mapFinishReason(event.stopReason);
            if (sawToolCall && (finishReason.unified === "stop" || finishReason.unified === "other")) {
              finishReason = { unified: "tool-calls", raw: finishReason.raw };
            }
            finished = true;
            controller.enqueue({
              type: "finish",
              finishReason,
              usage: mapUsage(event.usage),
              providerMetadata: {
                builder: self.contextMetadata(event)
              }
            });
            break;
          }
        }
      },
      flush(controller) {
        if (!started) controller.enqueue({ type: "stream-start", warnings });
        if (pendingTool) {
          controller.enqueue({ type: "error", error: new Error("Builder tool arguments truncated before completion") });
          pendingTool = void 0;
        }
        closeBlocks(controller);
        if (!finished) {
          controller.enqueue({
            type: "error",
            error: new InvalidResponseDataError({ data: void 0, message: "Builder stream ended without a done event" })
          });
          controller.enqueue({ type: "finish", finishReason: { unified: "error", raw: "truncated" }, usage: EMPTY_USAGE });
        }
      }
    });
  }
};
function parseLine(line) {
  const trimmed = line.trim();
  if (!trimmed) return void 0;
  try {
    const value = JSON.parse(trimmed);
    return value && typeof value === "object" ? value : void 0;
  } catch {
    return void 0;
  }
}
function createBuilder(options) {
  const baseURL = withoutTrailingSlash(options.baseURL) ?? "https://api.builder.io";
  const privateKey = loadApiKey({
    apiKey: options.privateKey,
    environmentVariableName: "BUILDER_PRIVATE_KEY",
    description: "Builder.io private key"
  });
  const createModel = (modelId, settings = {}) => new BuilderChatLanguageModel(modelId, settings, {
    baseURL,
    apiKey: options.apiKey,
    userId: options.userId,
    fetch: options.fetch ?? globalThis.fetch,
    headers: () => ({
      Authorization: "Bearer " + privateKey,
      "Content-Type": "application/json"
    })
  });
  const provider = function(modelId, settings) {
    return createModel(modelId, settings);
  };
  provider.languageModel = createModel;
  return provider;
}
export {
  BuilderChatLanguageModel,
  createBuilder,
  mapFinishReason,
  redact
};
