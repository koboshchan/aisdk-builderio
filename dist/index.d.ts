import { LanguageModelV3, LanguageModelV3CallOptions, LanguageModelV3GenerateResult, LanguageModelV3StreamResult, LanguageModelV3FinishReason } from '@ai-sdk/provider';

interface BuilderProviderSettings {
    baseURL?: string;
    apiKey: string;
    userId: string;
    privateKey?: string;
    /** Custom fetch, mainly for tests. */
    fetch?: typeof fetch;
}
interface BuilderChatSettings {
    temperature?: number;
    maxOutputTokens?: number;
    topP?: number;
    /** Builder session to continue. Defaults to one session per model instance. */
    sessionId?: string;
}
interface BuilderProvider {
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
declare function mapFinishReason(reason?: string | null): LanguageModelV3FinishReason;
/** Removes anything credential-shaped before text reaches an error message or log. */
declare function redact(text: string, secrets: string[]): string;
declare class BuilderChatLanguageModel implements LanguageModelV3 {
    readonly specificationVersion = "v3";
    readonly provider = "builder";
    readonly modelId: string;
    private readonly settings;
    private readonly config;
    private defaultSessionId;
    /** Sessions that already hold history on the Builder side. */
    private readonly knownSessions;
    private readonly sessionUrls;
    private readonly sessionPrompts;
    constructor(modelId: string, settings: BuilderChatSettings, config: BuilderConfig);
    get supportedUrls(): {
        "image/*": RegExp[];
    };
    private secrets;
    doGenerate(options: LanguageModelV3CallOptions): Promise<LanguageModelV3GenerateResult>;
    doStream(options: LanguageModelV3CallOptions): Promise<LanguageModelV3StreamResult>;
    private prepare;
    /**
     * Builder keeps history server side per session, so only the part of the prompt after the last
     * assistant message is new. When the session is unknown to this instance but the prompt already
     * has earlier turns, they are replayed as a transcript so no context is lost.
     */
    private convertPrompt;
    private post;
    private buildUrl;
    /** Same URL without the key, for errors and logs. */
    private publicUrl;
    private safeBody;
    private getUserContext;
    private toolCall;
    private contextMetadata;
    private eventError;
    private createJsonLineParser;
    private createStreamTransformer;
}
declare function createBuilder(options: BuilderProviderSettings): BuilderProvider;

export { BuilderChatLanguageModel, type BuilderChatSettings, type BuilderProvider, type BuilderProviderSettings, createBuilder, mapFinishReason, redact };
