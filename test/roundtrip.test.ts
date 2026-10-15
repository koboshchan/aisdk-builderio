import { it, expect } from "vitest";
import { generateText, streamText, tool, jsonSchema, stepCountIs } from "ai";
import { createBuilder } from "../src";
const wire=(...events:object[])=>events.map(e=>JSON.stringify(e)).join("\n")+"\n";
function setup(responses:string[]) {
 const calls:any[]=[];
 const model=createBuilder({apiKey:"pub",userId:"user",privateKey:"private",
 fetch:(async (_url:any,init:any)=>{calls.push(JSON.parse(init.body));return new Response(responses.shift()!,{status:200});}) as typeof fetch})("test");
 return {model,calls};
}
const tools={
 add:tool({inputSchema:jsonSchema<{a:number}>({type:"object",properties:{a:{type:"number"}},required:["a"]}),execute:async ({a})=>({sum:a+1})}),
 other:tool({inputSchema:jsonSchema<{}>({type:"object"}),execute:async ()=>"second result"}),
};
const first=wire({type:"tool",id:"a1",filePath:"mcp__ai-sdk__add",content:""},{type:"delta",content:'{"a":'},{type:"delta",content:"4}"},{type:"tool",id:"b1",filePath:"mcp__ai-sdk__other",content:""},{type:"delta",content:"{}"},{type:"done",stopReason:"tool_use"});
const final=wire({type:"text",content:"five"},{type:"done",stopReason:"end_turn"});
it("generateText executes two tools and returns both outputs in the next request",async()=>{
 const {model,calls}=setup([first,final]);
 const r=await generateText({model,tools,prompt:"compute",system:"keep answers short",stopWhen:stepCountIs(3)});
 expect(r.text).toBe("five");
 expect(r.steps).toHaveLength(2);
 expect(calls[0].localMCPTools).toHaveLength(2);
 expect(calls[1].sessionId).toBe(calls[0].sessionId);
 expect(calls[1].toolResults.map((x:any)=>[x.tool_use_id,x.content])).toEqual([["a1",'{"sum":5}'],["b1","second result"]]);
 expect(calls[1].systemReminders[0]).toBe("keep answers short");
 expect(calls[1].systemReminders[1]).toContain('"sum":5');
});
it("streamText emits input deltas and executes multiple tools across steps",async()=>{
 const {model,calls}=setup([first,final]);
 const r=streamText({model,tools,prompt:"compute",stopWhen:stepCountIs(3)});
 const parts:any[]=[];for await(const p of r.fullStream)parts.push(p);
 expect(await r.text).toBe("five");
 expect(parts.filter(p=>p.type==="tool-input-start")).toHaveLength(2);
 expect(parts.filter(p=>p.type==="tool-input-delta").map(p=>p.delta)).toEqual(['{"a":',"4}","{}"]);
 expect(parts.filter(p=>p.type==="tool-result")).toHaveLength(2);
 expect(calls[1].toolResults).toHaveLength(2);
});
it("replays all earlier roles, tool inputs/results, and system prompts for fresh history",async()=>{
 const {model,calls}=setup([final]);
 await model.doGenerate({prompt:[
 {role:"system",content:"system one"},{role:"system",content:"system two"},
 {role:"user",content:[{type:"text",text:"first user"}]},
 {role:"assistant",content:[{type:"tool-call",toolCallId:"old",toolName:"add",input:{a:1}}]},
 {role:"tool",content:[{type:"tool-result",toolCallId:"old",toolName:"add",output:{type:"json",value:{sum:2}}}]},
 {role:"assistant",content:[{type:"text",text:"two"}]},
 {role:"user",content:[{type:"text",text:"second user"}]},
 ]});
 expect(calls[0].systemReminders).toEqual(["system one","system two"]);
 for(const text of ["first user","add",'{"a":1}','{"sum":2}',"two","second user"])expect(calls[0].userPrompt).toContain(text);
 expect(calls[0].toolResults).toBeUndefined();
});
it("unrelated chats on one model do not share history",async()=>{
 const {model,calls}=setup([final,final]);
 for(const text of ["chat one","chat two"])await model.doGenerate({prompt:[{role:"user",content:[{type:"text",text}]}]});
 expect(calls[0].sessionId).not.toBe(calls[1].sessionId);
});
it("exposes server compaction, context window and account usage limits without silently truncating",async()=>{
 const done={type:"done",stopReason:"model_context_window_exceeded",compact:false,contextWindow:{usedTokens:123,totalTokens:500},usage:{limits:{aiGenerationContextWindow:400}}};
 const {model,calls}=setup([wire({type:"text",content:"partial answer"},done)]);
 const text="long history ".repeat(1000);
 const r=await model.doGenerate({prompt:[{role:"system",content:"preserve system"},{role:"user",content:[{type:"text",text}]}]});
 expect(calls[0].userPrompt).toBe(text);
 expect(r.finishReason.unified).toBe("length");
 expect(r.providerMetadata?.builder).toMatchObject({compacted:false,contextWindow:done.contextWindow,usage:done.usage});
});
for(const code of ["credits-limit-daily","credits-limit-monthly","context-too-long","prompt-too-long","ask-to-continue"])it("surfaces "+code+" even if done follows",async()=>{
 const {model}=setup([wire({type:"error",code,message:"account limit",usage:{limits:{aiGenerationContextWindow:400}}},{type:"done",stopReason:"end_turn"})]);
 const e:any=await model.doGenerate({prompt:[{role:"user",content:[{type:"text",text:"hi"}]}]}).catch(e=>e);
 expect(e.data.code).toBe(code);expect(e.isRetryable).toBe(false);
 expect(e.data.usage.limits.aiGenerationContextWindow).toBe(400);
});

it("streaming context errors carry account limits and finish as error",async()=>{
 const {model}=setup([wire({type:"error",code:"context-too-long",usage:{limits:{aiGenerationContextWindow:200}}},{type:"done",stopReason:"end_turn"})]);
 const {stream}=await model.doStream({prompt:[{role:"user",content:[{type:"text",text:"hi"}]}]});
 const parts:any[]=[];for await(const p of stream)parts.push(p);
 expect(parts.find(p=>p.type==="error").error.data.usage.limits.aiGenerationContextWindow).toBe(200);
 expect(parts.filter(p=>p.type==="finish")).toHaveLength(1);
 expect(parts.at(-1).finishReason.unified).toBe("error");
});
it("passes supported compaction controls without inventing an account token limit",async()=>{
 const {model,calls}=setup([final]);
 await model.doGenerate({prompt:[{role:"user",content:[{type:"text",text:"hi"}]}],providerOptions:{builder:{softContextWindow:100,errorIfHadCompaction:true}}});
 expect(calls[0].softContextWindow).toBe(100);expect(calls[0].errorIfHadCompaction).toBe(true);
});
it("a subsequent user turn carries complete prior context and current system instructions",async()=>{
 const {model,calls}=setup([final,final]);
 const prompt:any[]=[{role:"system",content:"secret-code instruction"},{role:"user",content:[{type:"text",text:"remember moss"}]}];
 await model.doGenerate({prompt});
 await model.doGenerate({prompt:[...prompt,{role:"assistant",content:[{type:"text",text:"stored"}]},{role:"user",content:[{type:"text",text:"recall it"}]}]});
 expect(calls[1].sessionId).toBe(calls[0].sessionId);
 expect(calls[1].userPrompt).toBe("recall it");
 expect(calls[1].systemReminders.join(" ")).toContain("remember moss");
 expect(calls[1].systemReminders.join(" ")).toContain("secret-code instruction");
});
