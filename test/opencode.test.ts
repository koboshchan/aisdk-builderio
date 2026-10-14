import { it, expect } from "vitest";
import * as current from "ai";
import * as opencode11420 from "ai-opencode-11420";
import { createBuilder } from "../src";

// OpenCode persists finish-step.finishReason in a string-only "reason" field.
// Keep the provider's V3 object intact: the AI SDK owns that conversion.
for (const [version, sdk] of [["current", current], ["OpenCode 1.14.20 / ai 6.0.168", opencode11420]] as const) {
  it(version + " converts V3 finish reasons across write/edit/bash continuations", async () => {
    const requests: any[] = [];
    const executed: string[] = [];
    let contents = "";
    const names = ["write", "edit", "bash"];
    const responses = names.map((name, index) => [
      { type: "tool", id: "call-" + index, filePath: "mcp__ai-sdk__" + name, content: "" },
      { type: "delta", content: "{}" },
      // Builder sometimes reports end_turn even after requesting a tool.
      { type: "done", stopReason: index === 1 ? "end_turn" : "tool_use" },
    ]);
    responses.push([{ type: "text", content: "verified" }, { type: "done", stopReason: "end_turn" }] as any);
    const model = createBuilder({
      apiKey: "test-public", userId: "test-user", privateKey: "test-private",
      fetch: (async (_url: any, init: any) => {
        requests.push(JSON.parse(init.body));
        const events = responses.shift();
        if (!events) throw new Error("Unexpected extra model step");
        return new Response(events.map(e => JSON.stringify(e)).join("\n") + "\n");
      }) as typeof fetch,
    })("test");
    const tools = Object.fromEntries(names.map(name => [name, sdk.tool({
      inputSchema: sdk.jsonSchema<{}>({ type: "object", properties: {} }),
      execute: async () => {
        executed.push(name);
        if (name === "write") contents = "one";
        if (name === "edit") contents = contents.replace("one", "two");
        return name === "bash" ? contents : "success";
      },
    })]));
    const result = sdk.streamText({ model, tools, prompt: "write, edit, then cat", stopWhen: sdk.stepCountIs(5) });
    const reasons: unknown[] = [];
    for await (const part of result.fullStream) {
      if (part.type === "error") throw part.error;
      if (part.type === "finish-step") {
        // This is the exact boundary that failed the OpenCode schema.
        expect(typeof part.finishReason).toBe("string");
        reasons.push(part.finishReason);
      }
    }
    expect(reasons).toEqual(["tool-calls", "tool-calls", "tool-calls", "stop"]);
    expect(executed).toEqual(names);
    expect(contents).toBe("two");
    expect(await result.text).toBe("verified");
    expect(requests).toHaveLength(4);
    expect(requests[3].toolResults[0].content).toBe("two");
    expect(requests[3].systemReminders.join(" ")).toContain("write");
    expect(model.specificationVersion).toBe("v3");
  });
}
