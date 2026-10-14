import {it,expect} from "vitest";
import {createBuilder} from "../src";
const wire=(...e:object[])=>e.map(x=>JSON.stringify(x)).join("\n");
it("uses nextUrl to continue native tool history instead of restarting the agent",async()=>{
 const calls:any[]=[];
 const responses=[
 wire({type:"tool",id:"t",filePath:"mcp__ai-sdk__seed",content:""},{type:"delta",content:"{}"},{type:"done",stopReason:"tool_use",nextUrl:"cgen://completion/first"}),
 wire({type:"text",content:"37"},{type:"done",stopReason:"end_turn",nextUrl:"cgen://completion/second"})];
 const model=createBuilder({apiKey:"p",userId:"u",privateKey:"s",fetch:(async(u:any,i:any)=>{calls.push(JSON.parse(i.body));return new Response(responses.shift());}) as typeof fetch})("test");
 const prompt:any[]=[{role:"user",content:[{type:"text",text:"seed please"}]}];
 await model.doGenerate({prompt});
 await model.doGenerate({prompt:[...prompt,{role:"assistant",content:[{type:"tool-call",toolCallId:"t",toolName:"seed",input:{}}]},{role:"tool",content:[{type:"tool-result",toolCallId:"t",toolName:"seed",output:{type:"json",value:{value:37}}}]}]});
 expect(calls[1].url).toBe("cgen://completion/first");
 expect(calls[1].toolResults[0].content).toBe('{"value":37}');
 expect(calls[1].systemReminders).toBeUndefined();
});
it("does not finish on an internal compact done before the requested answer",async()=>{
 const model=createBuilder({apiKey:"p",userId:"u",privateKey:"s",fetch:(async()=>new Response(wire(
 {type:"done",compact:true,stopReason:"end_turn",nextUrl:"cgen://completion/compact"},
 {type:"text",content:"actual answer"},
 {type:"done",compact:false,stopReason:"end_turn",nextUrl:"cgen://completion/answer"}
 ))) as typeof fetch})("test");
 const r=await model.doGenerate({prompt:[{role:"user",content:[{type:"text",text:"hi"}]}]});
 expect(r.content).toEqual([{type:"text",text:"actual answer"}]);
});
it("compaction-only responses fail instead of returning a successful empty answer",async()=>{
 const model=createBuilder({apiKey:"p",userId:"u",privateKey:"s",fetch:(async()=>new Response(wire({type:"done",compact:true,stopReason:"end_turn"}))) as typeof fetch})("test");
 await expect(model.doGenerate({prompt:[{role:"user",content:[{type:"text",text:"hi"}]}]})).rejects.toThrow(/without a done/);
});
