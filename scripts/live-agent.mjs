// Explicitly opt-in: uses the owner's normal Builder credits. Never prints credentials.
import fs from "node:fs";
import assert from "node:assert/strict";
import { generateText, streamText, tool, jsonSchema, stepCountIs } from "ai";
import { getEncoding } from "js-tiktoken";
import { createBuilder } from "../dist/index.mjs";
const credentialFile=process.env.BUILDER_CREDENTIAL_FILE;
if(!credentialFile)throw new Error("Set BUILDER_CREDENTIAL_FILE to opt into paid live checks");
const c=JSON.parse(fs.readFileSync(credentialFile,"utf8")).credentials;
const enc=getEncoding("cl100k_base");
const filler=Array.from({length:1800},(_,i)=>"Archive entry "+i+": ordinary inventory checked; no special instructions.").join("\n");
let long=filler;
while(enc.encode(long).length>7900)long=long.slice(0,long.lastIndexOf("\n"));
const facts="Start needle: project = violet-owl.\n"+long.slice(0,Math.floor(long.length/2))+"\nMiddle needle: harbor = cedar-bay.\n"+long.slice(Math.floor(long.length/2))+"\nEnd needle: badge = amber-kite.";
const system="Follow the user's task exactly. Use one tool per step in the requested order. Never guess tool outputs. After the three tools, give only the requested final answer. Conversation history supplied in reminders includes your original user request and previous tool results.";
async function run(mode,longContext){
 const requests=[], executed=[], doneEvents=[];
 const model=createBuilder({apiKey:c.builderPublicKey,userId:c.userId,privateKey:c.builderPrivateKey,
 fetch:async(u,i)=>{
  const b=JSON.parse(i.body);requests.push({sessionId:b.sessionId,url:b.url,userPrompt:b.userPrompt,toolResults:b.toolResults,reminders:b.systemReminders});
  const r=await fetch(u,i);
  // Read only aggregate usage/context metadata, never credentials or raw response logs.
  r.clone().text().then(t=>{for(const l of t.split("\n")){try{const e=JSON.parse(l);if(e.type==="done")doneEvents.push({creditsUsed:e.creditsUsed,contextWindow:e.contextWindow,model:e.model});}catch{}}});
  return r;
 }})("live-check");
 const numberSchema=jsonSchema({type:"object",properties:{value:{type:"number"}},required:["value"],additionalProperties:false});
 const tools={
  seed:tool({description:"First step. Returns a private seed needed by double.",inputSchema:jsonSchema({type:"object",properties:{},additionalProperties:false}),execute:async()=>{executed.push({tool:"seed",output:37});return {value:37};}}),
  double:tool({description:"Second step. Pass the exact seed value; doubles it.",inputSchema:numberSchema,execute:async({value})=>{assert.equal(value,37);executed.push({tool:"double",input:value,output:value*2});return {value:value*2};}}),
  seal:tool({description:"Third step. Pass the exact double result to obtain final code.",inputSchema:numberSchema,execute:async({value})=>{assert.equal(value,74);executed.push({tool:"seal",input:value,output:"seal-74-ok"});return {code:"seal-74-ok"};}}),
 };
 const task="Call seed, then double using seed's returned value, then seal using double's returned value. Do not combine tool calls in a step. "+(longContext?"After all tools finish, return exactly project|harbor|badge|code using the three needle facts from the archive and the seal result.":"After all tools finish, return exactly the code returned by seal.");
 const messages=longContext?[{role:"user",content:facts},{role:"assistant",content:"Archive received."},{role:"user",content:task}]:[{role:"user",content:task}];
 const opts={model,system,messages,tools,stopWhen:stepCountIs(5)};
 let text,steps,streamDeltas=0;
 if(mode==="streamText"){
  const r=streamText(opts);for await(const p of r.fullStream){if(p.type==="error")throw p.error;if(p.type==="tool-input-delta")streamDeltas++;}
  text=await r.text;steps=await r.steps;
 }else{const r=await generateText(opts);text=r.text;steps=r.steps;}
 const expected=longContext?"violet-owl|cedar-bay|amber-kite|seal-74-ok":"seal-74-ok";
 if(text.trim()!==expected)console.log(JSON.stringify({failure:true,mode,text,executed,steps:steps.map(s=>({finish:s.finishReason,text:s.text,calls:s.toolCalls})),doneEvents}));
 assert.equal(text.trim(),expected);
 assert.deepEqual(executed.map(e=>e.tool),["seed","double","seal"]);
 assert.equal(steps.length,4);
 assert.equal(requests.length,4);
 for(let i=1;i<requests.length;i++){
  assert.equal(requests[i].sessionId,requests[0].sessionId);
  assert.equal(requests[i].toolResults.length,1);
  const prior=steps[i-1].toolCalls[0];
  assert.equal(requests[i].toolResults[0].tool_use_id,prior.toolCallId);
  assert(requests[i].url?.startsWith("cgen://completion/"));
 }
 const result={mode,longContext,archiveTokens:longContext?enc.encode(facts).length:0,tokenCounter:"cl100k_base (approximation, not backend tokenizer)",steps:steps.length,requests:requests.length,executed,final:text.trim(),streamDeltas,toolResultsForwarded:true,doneEvents};
 console.log(JSON.stringify(result));
 return result;
}
const results=[];
for(const [mode,longContext] of (process.env.LIVE_LONG_ONLY ? [["streamText",true]] : [["generateText",false],["streamText",true]]))results.push(await run(mode,longContext));
fs.writeFileSync(process.env.LIVE_RESULT_FILE??"/tmp/sdk-live-agent-result.json",JSON.stringify(results,null,2));
