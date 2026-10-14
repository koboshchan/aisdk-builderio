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
