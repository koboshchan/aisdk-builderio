# ai-sdk-provider-builder

custom ai sdk provider for builder.io

made by kobosh

## usage

```ts
import { createBuilder } from 'ai-sdk-provider-builder';
import { generateText } from 'ai';

async function main() {
  const builder = createBuilder({
    // found in your node_modules/.builder/data.json
    apiKey: 'apiKey', // user api key
    userId: 'userId', // credentials.userId
    privateKey: 'bpk-privateKey', // credentials.builderPrivateKey
  });

  const { text } = await generateText({
    model: builder('builder-model'), // modelId is required but currently unused by our internal mapping
    prompt: 'what is your name?',
  });
  console.log(text);
}

main();

```

## OpenCode config

The `dist` branch contains the built package, so Bun can install it without running a build script. Add this provider to `~/.config/opencode/opencode.json` and select `ai-sdk-provider-builder/builder-model`.

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "ai-sdk-provider-builder": {
      "npm": "github:koboshchan/aisdk-builderio#dist",
      "name": "Builder.io",
      "options": {
        "apiKey": "YOUR_BUILDER_API_KEY",
        "userId": "YOUR_BUILDER_USER_ID",
        "privateKey": "YOUR_BUILDER_PRIVATE_KEY"
      },
      "models": {
        "builder-model": {
          "name": "Builder (quality-v4)",
          "limit": { "context": 200000, "output": 65536 }
        }
      }
    }
  }
}
```

Use your own Builder credentials from `~/.builder/config/data.json`. Map `apiKey` to `options.apiKey`, `credentials.userId` to `options.userId`, and `credentials.builderPrivateKey` to `options.privateKey`. The provider does not read that file automatically. Keep the resulting config private and never commit it. Merge this provider into your existing config rather than replacing other settings.

Every push to `main` builds, tests, and publishes the package to `dist`. The workflow can also be dispatched manually after it is available on the default branch. Feature-branch pushes build and test without publishing. Bun caches Git dependencies, so refresh the dependency when you want a newer `dist` commit, or pin a commit instead of `dist` for reproducible installs.

## Output and backend limits

JSON response format is emulated with a system reminder, including the caller's JSON schema when supplied. The provider removes a surrounding JSON Markdown fence. This is not constrained decoding or a schema guarantee; keep AI SDK validation and handle validation failures. JSON-mode text is buffered until the text block ends so fences split across chunks can be removed. Normal text streaming stays incremental.

Image input supports PNG, JPEG, GIF, and WebP, including multiple images and image-only prompts. AI SDK file bytes/base64 become inline Builder uploads (maximum 5 MiB of base64 per image); HTTP(S) image URLs become URL attachments. Only new user turns send attachments, not historical images. Earlier images require native server continuation to retain visual context; textual history replay omits their binary payloads. Inline image data is redacted from request/error metadata. The wire format follows the Builder CLI; visual receipt has not been verified with a live paid call.

Explicit token counters in a final `done.usage` are forwarded to AI SDK input/output usage, from which the SDK computes `totalTokens`. Missing counters stay undefined, not zero. Some backend versions return account usage (`total`, `fast`, `quality`, limits) rather than token usage. Those are not token counts and remain available in `providerMetadata.builder.usage`, along with `contextWindow` and `creditsUsed`; they are never relabeled as tokens.

Builder always lets the model choose tools. Required/specific `toolChoice` is not enforced; validate the result or retry if a tool call is mandatory. Streaming may arrive in a few large chunks even though the provider parses NDJSON incrementally. Live checks on October 15, 2026 saw roughly 2.3 seconds to first text and 2–6 chunks; these are observations, not latency guarantees.

The free plan can return HTTP 402 with `credits-limit-daily` after only a handful of calls. The provider does not bypass credits or turn that failure into a successful empty response.

AI SDK's `streamText().textStream` does not throw stream error parts by default. Supply `onError` or inspect `fullStream` to surface them.

```ts
const result = streamText({
  model: builder('builder-model'),
  prompt: 'hello',
  onError({ error }) {
    console.error(error);
  },
});
for await (const part of result.fullStream) {
  if (part.type === 'error') throw part.error;
  if (part.type === 'text-delta') process.stdout.write(part.text);
}
```
