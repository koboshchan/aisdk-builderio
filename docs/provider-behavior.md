The provider targets AI SDK 6's LanguageModelV3 contract.

Caller function schemas are registered as local MCP tools. Builder's direct endpoint has been observed sending a tool header with its name in filePath, followed by argument deltas and a done event. These are forwarded as tool-input-start, tool-input-delta, tool-input-end and a complete tool-call. A complete tool event is also supported, with its arguments emitted as one input delta. No token-level deltas are invented for complete events. Multiple calls retain their own ids and names. AI SDK executes the tools and submits their results on the next step; use stopWhen with stepCountIs to allow more than one step.

Native continuation uses the previous done event's cgen nextUrl in the request body's url field, matching the official CLI. When that continuation is available, native server history is used and replay is omitted. Internal compact:true done events do not finish the requested answer.

Without a native continuation, a session id alone is not proof that the direct completion endpoint preserved prior turns. Each continuation includes the full caller-owned conversation as a system reminder, along with the original system prompts, while sending new tool results in Builder's native format. Fresh sessions replay earlier turns as a transcript. Reusing a model for an unrelated prompt rotates its default session. An explicit builder sessionId deliberately opts into that session. History reminders are serialized conversation data, not a replacement for system instructions.

History is not silently truncated locally. Builder's account-specific usage limits, contextWindow and compact flag are returned in providerMetadata.builder. Context exhaustion maps to length; credit/context errors remain errors with their codes and usage details, not successful empty answers. providerOptions.builder accepts softContextWindow and errorIfHadCompaction using the official completion fields. These cannot enlarge an account's allowance. Builder may compact context server-side; full-history forwarding does not guarantee that every old token remains in the model's effective context. It also increases input size, so callers should manage long histories deliberately.

Tests exercise actual generateText and streamText execution with two tools, streamed argument fragments, tool-result continuation, system prompts, historical tool inputs/results, independent chats, context/credit errors, and reported account limits. Live verification used the owner's credentials with normal credits. A two-tool streaming call returned add=5 and word=moss, and a separate multi-turn history test recalled violet-owl. No quota or billing bypass is implemented.

## OpenCode finish-step compatibility

The provider uses LanguageModel V3. Its internal finish reason must remain
`{ unified, raw }`; AI SDK streamText converts that to the string
`finish-step.finishReason` consumed by OpenCode's persisted `reason` field.
Changing the provider to return a string while still declaring V3 is not a fix.

The compatibility regression covers write/edit/bash tool continuations with
both current AI SDK and AI SDK 6.0.168 (OpenCode 1.14.20's pinned catalog
version). Both must emit three string `tool-calls` reasons followed by `stop`.

On October 14, a reported OpenCode retry error said `reason` received an
object instead of a string. Fresh CLI processes using the existing a3611d8
build could not reproduce it. Live write/edit/bash completed with OpenCode
1.14.20, 1.19.0, and 2.0.25 on the same Mac. The Homebrew executable had
changed to 2.0.25 during investigation. Historical logs confirm the error,
but do not establish its underlying cause. An already-running process can
retain an earlier imported provider; restart that client before retesting.
No runtime finish-reason workaround was added without a failing reproduction.
