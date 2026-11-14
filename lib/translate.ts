// translate.ts — wire translations between the client's wire and the
// provider's wire, plus the parsers that read a provider's answer and the
// renderers that write the client's answer.
//
// The design is deliberately buffered, not streamed through: the upstream
// answer is read whole (JSON or SSE), the text + usage + stop reason are
// extracted, and then the client's JSON or SSE response is rendered from
// those three values. That keeps the echo exact and the streaming event
// order correct on both wires, and it makes the replay deterministic.
//
// Supported translations:
//   openai <-> anthropic, and either -> gemini. Gemini's own wire only
//   ever appears upstream. Stop reasons map by family: the env's providers
//   always answer family 0 (stop / end_turn / STOP), which is the only
//   family the workload ever produces.
import type { Wire } from "./brief.ts";

export type JsonRecord = Record<string, unknown>;

export interface ProviderAnswer {
  text: string;
  inputTokens: number;
  outputTokens: number;
  /** Raw provider stop string, e.g. "stop" or "end_turn" or "STOP". */
  stop: string;
}

const FIXED_CREATED = 1767225600;

function isRecord(v: unknown): v is JsonRecord {
  return v !== null && typeof v === "object" && !Array.isArray(v);
}

function asText(v: unknown): string {
  return typeof v === "string" ? v : "";
}

function asNum(v: unknown, fallback = 0): number {
  return typeof v === "number" && Number.isFinite(v) ? v : fallback;
}

/** A string or array-of-strings field, normalized to an array. */
function stringList(v: unknown): string[] {
  if (typeof v === "string") return [v];
  if (Array.isArray(v)) return v.filter((x): x is string => typeof x === "string");
  return [];
}

/** Extract plain text from OpenAI content (string or [{type:"text"}]). */
export function openAIContentText(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .filter((p): p is JsonRecord => isRecord(p) && p.type === "text")
      .map((p) => asText(p.text))
      .join("");
  }
  return "";
}

/** Extract plain text from Anthropic content blocks. */
export function anthropicContentText(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .filter((p): p is JsonRecord => isRecord(p) && p.type === "text")
      .map((p) => asText(p.text))
      .join("");
  }
  return "";
}

/** Split a data: URL into [mimeType, data], or null when it is not one. */
function splitDataUrl(url: string): [string, string] | null {
  const m = /^data:([^;,]+)?(;base64)?,(.*)$/s.exec(url);
  if (!m) return null;
  return [m[1] || "image/png", m[3] || ""];
}

/** Guess a MIME type from a URL's extension (fallback for image URLs). */
function mimeFromUrl(url: string): string {
  const clean = url.split("?")[0].toLowerCase();
  if (clean.endsWith(".png")) return "image/png";
  if (clean.endsWith(".gif")) return "image/gif";
  if (clean.endsWith(".webp")) return "image/webp";
  if (clean.endsWith(".jpeg") || clean.endsWith(".jpg")) return "image/jpeg";
  return "image/jpeg";
}

/** Convert an OpenAI chat body into an Anthropic messages body. */
export function openAIToAnthropic(body: JsonRecord, model: string): JsonRecord {
  const systemParts: string[] = [];
  const messages: JsonRecord[] = [];
  const rawMessages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of rawMessages) {
    if (!isRecord(raw)) continue;
    const role = asText(raw.role);
    if (role === "system") {
      systemParts.push(openAIContentText(raw.content));
      continue;
    }
    if (role === "tool") {
      messages.push({
        role: "user",
        content: [
          {
            type: "tool_result",
            tool_use_id: asText(raw.tool_call_id),
            content: openAIContentText(raw.content),
          },
        ],
      });
      continue;
    }
    if (role === "assistant") {
      const content: JsonRecord[] = [];
      const text = openAIContentText(raw.content);
      if (text) content.push({ type: "text", text });
      if (Array.isArray(raw.tool_calls)) {
        for (const call of raw.tool_calls) {
          if (!isRecord(call)) continue;
          const fn = isRecord(call.function) ? call.function : {};
          let input: unknown = {};
          try {
            input = JSON.parse(asText(fn.arguments) || "{}");
          } catch {
            input = {};
          }
          content.push({
            type: "tool_use",
            id: asText(call.id),
            name: asText(fn.name),
            input,
          });
        }
      }
      messages.push({ role: "assistant", content });
      continue;
    }
    // role "user": text and image_url parts become text and image blocks.
    const content: JsonRecord[] = [];
    const parts = Array.isArray(raw.content)
      ? raw.content
      : typeof raw.content === "string"
        ? [{ type: "text", text: raw.content }]
        : [];
    for (const part of parts) {
      if (!isRecord(part)) continue;
      if (part.type === "text") {
        content.push({ type: "text", text: asText(part.text) });
      } else if (part.type === "image_url" && isRecord(part.image_url)) {
        const url = asText(part.image_url.url);
        const data = splitDataUrl(url);
        if (data) {
          content.push({
            type: "image",
            source: { type: "base64", media_type: data[0], data: data[1] },
          });
        } else {
          content.push({
            type: "image",
            source: { type: "url", url },
          });
        }
      }
    }
    messages.push({ role: "user", content });
  }
  const out: JsonRecord = {
    model,
    max_tokens: asNum(body.max_tokens, 1024),
    messages,
  };
  if (systemParts.length > 0) out.system = systemParts.join("\n\n");
  if (typeof body.temperature === "number") out.temperature = body.temperature;
  if (typeof body.top_p === "number") out.top_p = body.top_p;
  if (Array.isArray(body.tools)) {
    out.tools = body.tools
      .filter((t): t is JsonRecord => isRecord(t) && t.type === "function")
      .map((t) => {
        const fn = isRecord(t.function) ? t.function : {};
        return {
          name: asText(fn.name),
          description: asText(fn.description),
          input_schema: isRecord(fn.parameters) ? fn.parameters : {},
        };
      });
  }
  if (body.tool_choice !== undefined) {
    const tc = body.tool_choice;
    if (tc === "auto") out.tool_choice = { type: "auto" };
    else if (tc === "none") out.tool_choice = { type: "none" };
    else if (isRecord(tc) && tc.type === "function" && isRecord(tc.function)) {
      out.tool_choice = { type: "tool", name: asText(tc.function.name) };
    }
  }
  const stops = stringList(body.stop);
  if (stops.length > 0) out.stop_sequences = stops;
  if (body.stream === true) out.stream = true;
  return out;
}

/** Convert an Anthropic messages body into an OpenAI chat body. */
export function anthropicToOpenAI(body: JsonRecord, model: string): JsonRecord {
  const messages: JsonRecord[] = [];
  if (body.system !== undefined) {
    messages.push({
      role: "system",
      content: anthropicContentText(body.system),
    });
  }
  const rawMessages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of rawMessages) {
    if (!isRecord(raw)) continue;
    const role = asText(raw.role);
    if (role === "user") {
      const blocks = Array.isArray(raw.content)
        ? raw.content
        : typeof raw.content === "string"
          ? [{ type: "text", text: raw.content }]
          : [];
      const content: JsonRecord[] = [];
      for (const block of blocks) {
        if (!isRecord(block)) continue;
        if (block.type === "text") {
          content.push({ type: "text", text: asText(block.text) });
        } else if (block.type === "image" && isRecord(block.source)) {
          const src = block.source;
          if (src.type === "base64") {
            content.push({
              type: "image_url",
              image_url: {
                url: `data:${asText(src.media_type) || "image/png"};base64,${asText(src.data)}`,
              },
            });
          } else if (src.type === "url") {
            content.push({ type: "image_url", image_url: { url: asText(src.url) } });
          }
        } else if (block.type === "tool_result") {
          // A tool result becomes its own tool message, in place, so the
          // message order the client sent is preserved.
          if (content.length > 0) {
            messages.push({ role: "user", content });
            content.length = 0;
          }
          messages.push({
            role: "tool",
            tool_call_id: asText(block.tool_use_id),
            content: asText(block.content),
          });
        }
      }
      if (content.length > 0) messages.push({ role: "user", content });
      continue;
    }
    // role "assistant": text and tool_use blocks become content + tool_calls.
    const blocks = Array.isArray(raw.content)
      ? raw.content
      : typeof raw.content === "string"
        ? [{ type: "text", text: raw.content }]
        : [];
    const text = blocks
      .filter((b): b is JsonRecord => isRecord(b) && b.type === "text")
      .map((b) => asText(b.text))
      .join("");
    const toolCalls = blocks
      .filter((b): b is JsonRecord => isRecord(b) && b.type === "tool_use")
      .map((b) => ({
        id: asText(b.id),
        type: "function",
        function: {
          name: asText(b.name),
          arguments: JSON.stringify(isRecord(b.input) ? b.input : {}),
        },
      }));
    const assistant: JsonRecord = { role: "assistant", content: text };
    if (toolCalls.length > 0) assistant.tool_calls = toolCalls;
    messages.push(assistant);
  }
  const out: JsonRecord = {
    model,
    messages,
    max_tokens: asNum(body.max_tokens, 1024),
  };
  if (typeof body.temperature === "number") out.temperature = body.temperature;
  if (typeof body.top_p === "number") out.top_p = body.top_p;
  if (Array.isArray(body.tools)) {
    out.tools = body.tools
      .filter((t): t is JsonRecord => isRecord(t))
      .map((t) => ({
        type: "function",
        function: {
          name: asText(t.name),
          description: asText(t.description),
          parameters: isRecord(t.input_schema) ? t.input_schema : {},
        },
      }));
  }
  if (isRecord(body.tool_choice)) {
    const tc = body.tool_choice;
    if (tc.type === "auto") out.tool_choice = "auto";
    else if (tc.type === "none") out.tool_choice = "none";
    else if (tc.type === "tool") {
      out.tool_choice = {
        type: "function",
        function: { name: asText(tc.name) },
      };
    }
  }
  const stops = stringList(body.stop_sequences);
  if (stops.length > 0) out.stop = stops;
  if (body.stream === true) out.stream = true;
  return out;
}

/** One Gemini content part from an OpenAI content part. */
function geminiPartFromOpenAI(part: JsonRecord): JsonRecord | null {
  if (part.type === "text") return { text: asText(part.text) };
  if (part.type === "image_url" && isRecord(part.image_url)) {
    const url = asText(part.image_url.url);
    const data = splitDataUrl(url);
    if (data) return { inlineData: { mimeType: data[0], data: data[1] } };
    if (url) return { fileData: { mimeType: mimeFromUrl(url), fileUri: url } };
  }
  return null;
}

/** One Gemini content part from an Anthropic content block. */
function geminiPartFromAnthropic(block: JsonRecord): JsonRecord | null {
  if (block.type === "text") return { text: asText(block.text) };
  if (block.type === "image" && isRecord(block.source)) {
    const src = block.source;
    if (src.type === "base64") {
      return {
        inlineData: {
          mimeType: asText(src.media_type) || "image/png",
          data: asText(src.data),
        },
      };
    }
    if (src.type === "url") {
      return {
        fileData: { mimeType: mimeFromUrl(asText(src.url)), fileUri: asText(src.url) },
      };
    }
  }
  if (block.type === "tool_use") {
    return {
      functionCall: {
        name: asText(block.name),
        args: isRecord(block.input) ? block.input : {},
      },
    };
  }
  if (block.type === "tool_result") {
    return {
      functionResponse: {
        name: asText(block.tool_use_id),
        response: { result: asText(block.content) },
      },
    };
  }
  return null;
}

/** Gemini function declarations from either wire's tool list. */
function geminiTools(clientApi: "openai" | "anthropic", body: JsonRecord): JsonRecord[] {
  const declarations: JsonRecord[] = [];
  if (!Array.isArray(body.tools)) return declarations;
  for (const tool of body.tools) {
    if (!isRecord(tool)) continue;
    if (clientApi === "openai" && tool.type === "function" && isRecord(tool.function)) {
      declarations.push({
        name: asText(tool.function.name),
        description: asText(tool.function.description),
        parameters: isRecord(tool.function.parameters) ? tool.function.parameters : {},
      });
    } else if (clientApi === "anthropic") {
      declarations.push({
        name: asText(tool.name),
        description: asText(tool.description),
        parameters: isRecord(tool.input_schema) ? tool.input_schema : {},
      });
    }
  }
  return declarations.length > 0 ? [{ functionDeclarations: declarations }] : [];
}

/**
 * Build a Gemini request (path suffix + body) from either client wire.
 * Gemini carries system instructions separately, roles are "user"/"model",
 * and JSON mode is responseMimeType "application/json".
 */
export function toGemini(
  clientApi: "openai" | "anthropic",
  body: JsonRecord,
  model: string,
  stream: boolean,
): { path: string; body: JsonRecord } {
  const systemParts: JsonRecord[] = [];
  const contents: JsonRecord[] = [];
  const generationConfig: JsonRecord = {};

  if (clientApi === "openai") {
    const rawMessages = Array.isArray(body.messages) ? body.messages : [];
    for (const raw of rawMessages) {
      if (!isRecord(raw)) continue;
      const role = asText(raw.role);
      if (role === "system") {
        const text = openAIContentText(raw.content);
        if (text) systemParts.push({ text });
        continue;
      }
      if (role === "tool") {
        contents.push({
          role: "user",
          parts: [
            {
              functionResponse: {
                name: asText(raw.tool_call_id),
                response: { result: openAIContentText(raw.content) },
              },
            },
          ],
        });
        continue;
      }
      if (role === "assistant") {
        const parts: JsonRecord[] = [];
        const text = openAIContentText(raw.content);
        if (text) parts.push({ text });
        if (Array.isArray(raw.tool_calls)) {
          for (const call of raw.tool_calls) {
            if (!isRecord(call)) continue;
            const fn = isRecord(call.function) ? call.function : {};
            let args: unknown = {};
            try {
              args = JSON.parse(asText(fn.arguments) || "{}");
            } catch {
              args = {};
            }
            parts.push({
              functionCall: {
                name: asText(fn.name),
                args: isRecord(args) ? args : {},
              },
            });
          }
        }
        contents.push({ role: "model", parts });
        continue;
      }
      const parts: JsonRecord[] = [];
      const items = Array.isArray(raw.content)
        ? raw.content
        : typeof raw.content === "string"
          ? [{ type: "text", text: raw.content }]
          : [];
      for (const item of items) {
        if (!isRecord(item)) continue;
        const part = geminiPartFromOpenAI(item);
        if (part) parts.push(part);
      }
      contents.push({ role: "user", parts });
    }
    const fmt = isRecord(body.response_format) ? body.response_format : null;
    if (fmt && (fmt.type === "json_object" || fmt.type === "json_schema")) {
      generationConfig.responseMimeType = "application/json";
    }
    const stops = stringList(body.stop);
    if (stops.length > 0) generationConfig.stopSequences = stops;
  } else {
    if (body.system !== undefined) {
      const text = anthropicContentText(body.system);
      if (text) systemParts.push({ text });
    }
    const rawMessages = Array.isArray(body.messages) ? body.messages : [];
    for (const raw of rawMessages) {
      if (!isRecord(raw)) continue;
      const role = asText(raw.role);
      const blocks = Array.isArray(raw.content)
        ? raw.content
        : typeof raw.content === "string"
          ? [{ type: "text", text: raw.content }]
          : [];
      const parts: JsonRecord[] = [];
      for (const block of blocks) {
        if (!isRecord(block)) continue;
        const part = geminiPartFromAnthropic(block);
        if (part) parts.push(part);
      }
      if (parts.length > 0) {
        contents.push({ role: role === "assistant" ? "model" : "user", parts });
      }
    }
    const stops = stringList(body.stop_sequences);
    if (stops.length > 0) generationConfig.stopSequences = stops;
  }

  if (typeof body.max_tokens === "number") {
    generationConfig.maxOutputTokens = body.max_tokens;
  }
  if (typeof body.temperature === "number") {
    generationConfig.temperature = body.temperature;
  }
  if (typeof body.top_p === "number") {
    generationConfig.topP = body.top_p;
  }

  const request: JsonRecord = { contents };
  if (systemParts.length > 0) request.systemInstruction = { parts: systemParts };
  const tools = geminiTools(clientApi, body);
  if (tools.length > 0) request.tools = tools;
  if (Object.keys(generationConfig).length > 0) {
    request.generationConfig = generationConfig;
  }
  const verb = stream ? ":streamGenerateContent" : ":generateContent";
  return { path: `/v1beta/models/${model}${verb}`, body: request };
}

/** Parse an OpenAI chat completion JSON body into a ProviderAnswer. */
export function fromOpenAI(body: JsonRecord): ProviderAnswer {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const first = choices.length > 0 && isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(first.message) ? first.message : {};
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text: openAIContentText(message.content),
    inputTokens: asNum(usage.prompt_tokens),
    outputTokens: asNum(usage.completion_tokens),
    stop: asText(first.finish_reason) || "stop",
  };
}

/** Parse an Anthropic message JSON body into a ProviderAnswer. */
export function fromAnthropic(body: JsonRecord): ProviderAnswer {
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text: anthropicContentText(body.content),
    inputTokens: asNum(usage.input_tokens),
    outputTokens: asNum(usage.output_tokens),
    stop: asText(body.stop_reason) || "end_turn",
  };
}

/** Parse a Gemini generateContent JSON body into a ProviderAnswer. */
export function fromGemini(body: JsonRecord): ProviderAnswer {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const first = candidates.length > 0 && isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(first.content) ? first.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const text = parts
    .filter((p): p is JsonRecord => isRecord(p) && typeof p.text === "string")
    .map((p) => asText(p.text))
    .join("");
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text,
    inputTokens: asNum(usage.promptTokenCount),
    outputTokens: asNum(usage.candidatesTokenCount),
    stop: asText(first.finishReason) || "STOP",
  };
}

/** Parse a provider's SSE text into a ProviderAnswer (streaming mode). */
export function parseProviderSse(wire: Wire, text: string): ProviderAnswer {
  const events: JsonRecord[] = [];
  for (const frame of text.split(/\n\n/)) {
    for (const line of frame.split("\n")) {
      const trimmed = line.trim();
      if (!trimmed.startsWith("data:")) continue;
      const payload = trimmed.slice(5).trim();
      if (!payload || payload === "[DONE]") continue;
      try {
        const parsed: unknown = JSON.parse(payload);
        if (isRecord(parsed)) events.push(parsed);
      } catch {
        // A partial frame is not an answer; ignore it.
      }
    }
  }
  if (wire === "openai") {
    let out = "";
    let stop = "stop";
    let input = 0;
    let output = 0;
    for (const event of events) {
      const choices = Array.isArray(event.choices) ? event.choices : [];
      const first = choices.length > 0 && isRecord(choices[0]) ? choices[0] : {};
      const delta = isRecord(first.delta) ? first.delta : {};
      out += openAIContentText(delta.content);
      if (typeof first.finish_reason === "string") stop = first.finish_reason;
      const usage = isRecord(event.usage) ? event.usage : {};
      if (typeof usage.prompt_tokens === "number") input = usage.prompt_tokens;
      if (typeof usage.completion_tokens === "number") output = usage.completion_tokens;
    }
    return { text: out, inputTokens: input, outputTokens: output, stop };
  }
  if (wire === "anthropic") {
    let out = "";
    let stop = "end_turn";
    let input = 0;
    let output = 0;
    for (const event of events) {
      const type = asText(event.type);
      if (type === "message_start" && isRecord(event.message)) {
        const usage = isRecord(event.message.usage) ? event.message.usage : {};
        input = asNum(usage.input_tokens);
      } else if (type === "content_block_delta" && isRecord(event.delta)) {
        out += asText(event.delta.text);
      } else if (type === "message_delta") {
        const usage = isRecord(event.usage) ? event.usage : {};
        output = asNum(usage.output_tokens);
        if (typeof event.stop_reason === "string") stop = event.stop_reason;
      }
    }
    return { text: out, inputTokens: input, outputTokens: output, stop };
  }
  // gemini: each frame is a candidate with parts; the last carries usage.
  let out = "";
  let stop = "STOP";
  let input = 0;
  let output = 0;
  for (const event of events) {
    const candidates = Array.isArray(event.candidates) ? event.candidates : [];
    const first = candidates.length > 0 && isRecord(candidates[0]) ? candidates[0] : {};
    const content = isRecord(first.content) ? first.content : {};
    const parts = Array.isArray(content.parts) ? content.parts : [];
    for (const part of parts) {
      if (isRecord(part) && typeof part.text === "string") out += part.text;
    }
    if (typeof first.finishReason === "string") stop = first.finishReason;
    const usage = isRecord(event.usageMetadata) ? event.usageMetadata : {};
    if (typeof usage.promptTokenCount === "number") input = usage.promptTokenCount;
    if (typeof usage.candidatesTokenCount === "number") output = usage.candidatesTokenCount;
  }
  return { text: out, inputTokens: input, outputTokens: output, stop };
}

/** Map a provider stop string to the client's stop family. */
export function clientStop(
  stop: string,
  clientApi: "openai" | "anthropic",
): string {
  const s = stop.toLowerCase();
  if (s === "stop" || s === "end_turn" || s === "stop_sequence" || s === "end_of_turn") {
    return clientApi === "openai" ? "stop" : "end_turn";
  }
  if (s === "length" || s === "max_tokens" || s === "max_output_tokens") {
    return clientApi === "openai" ? "length" : "max_tokens";
  }
  if (s === "tool_calls" || s === "tool_use") {
    return clientApi === "openai" ? "tool_calls" : "tool_use";
  }
  if (s === "content_filter" || s === "refusal" || s === "safety") {
    return clientApi === "openai" ? "content_filter" : "refusal";
  }
  return clientApi === "openai" ? "stop" : "end_turn";
}

/** Render an OpenAI JSON chat completion from a provider answer. */
export function renderOpenAIJson(
  model: string,
  answer: ProviderAnswer,
): JsonRecord {
  return {
    id: "chatcmpl-agg",
    object: "chat.completion",
    created: FIXED_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: "assistant", content: answer.text },
        finish_reason: clientStop(answer.stop, "openai"),
      },
    ],
    usage: {
      prompt_tokens: answer.inputTokens,
      completion_tokens: answer.outputTokens,
      total_tokens: answer.inputTokens + answer.outputTokens,
    },
  };
}

/** Render an OpenAI SSE stream from a provider answer (buffered). */
export function renderOpenAIStream(
  model: string,
  answer: ProviderAnswer,
): string {
  const chunks: string[] = [];
  const base: JsonRecord = {
    id: "chatcmpl-agg",
    object: "chat.completion.chunk",
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta: {}, finish_reason: null }],
  };
  const role = { ...base, choices: [{ index: 0, delta: { role: "assistant" }, finish_reason: null }] };
  chunks.push(`data: ${JSON.stringify(role)}\n\n`);
  for (let i = 0; i < answer.text.length; i += 13) {
    const piece = answer.text.slice(i, i + 13);
    const chunk = {
      ...base,
      choices: [{ index: 0, delta: { content: piece }, finish_reason: null }],
    };
    chunks.push(`data: ${JSON.stringify(chunk)}\n\n`);
  }
  const finish = {
    ...base,
    choices: [
      {
        index: 0,
        delta: {},
        finish_reason: clientStop(answer.stop, "openai"),
      },
    ],
  };
  chunks.push(`data: ${JSON.stringify(finish)}\n\n`);
  const usage = {
    ...base,
    choices: [{ index: 0, delta: {}, finish_reason: null }],
    usage: {
      prompt_tokens: answer.inputTokens,
      completion_tokens: answer.outputTokens,
      total_tokens: answer.inputTokens + answer.outputTokens,
    },
  };
  chunks.push(`data: ${JSON.stringify(usage)}\n\n`);
  chunks.push("data: [DONE]\n\n");
  return chunks.join("");
}

/** Render an Anthropic JSON message from a provider answer. */
export function renderAnthropicJson(
  model: string,
  answer: ProviderAnswer,
): JsonRecord {
  return {
    id: "msg_agg",
    type: "message",
    role: "assistant",
    model,
    content: [{ type: "text", text: answer.text }],
    stop_reason: clientStop(answer.stop, "anthropic"),
    usage: {
      input_tokens: answer.inputTokens,
      output_tokens: answer.outputTokens,
    },
  };
}

/** Render an Anthropic SSE stream from a provider answer (buffered). */
export function renderAnthropicStream(
  model: string,
  answer: ProviderAnswer,
): string {
  const chunks: string[] = [];
  const start = {
    type: "message_start",
    message: {
      id: "msg_agg",
      type: "message",
      role: "assistant",
      model,
      content: [],
      stop_reason: null,
      usage: { input_tokens: answer.inputTokens, output_tokens: 0 },
    },
  };
  chunks.push(`event: message_start\ndata: ${JSON.stringify(start)}\n\n`);
  const blockStart = {
    type: "content_block_start",
    index: 0,
    content_block: { type: "text", text: "" },
  };
  chunks.push(`event: content_block_start\ndata: ${JSON.stringify(blockStart)}\n\n`);
  for (let i = 0; i < answer.text.length; i += 13) {
    const delta = {
      type: "content_block_delta",
      index: 0,
      delta: { type: "text_delta", text: answer.text.slice(i, i + 13) },
    };
    chunks.push(`event: content_block_delta\ndata: ${JSON.stringify(delta)}\n\n`);
  }
  const blockStop = { type: "content_block_stop", index: 0 };
  chunks.push(`event: content_block_stop\ndata: ${JSON.stringify(blockStop)}\n\n`);
  const messageDelta = {
    type: "message_delta",
    delta: { stop_reason: clientStop(answer.stop, "anthropic") },
    usage: { output_tokens: answer.outputTokens },
  };
  chunks.push(`event: message_delta\ndata: ${JSON.stringify(messageDelta)}\n\n`);
  chunks.push("event: message_stop\ndata: {}\n\n");
  return chunks.join("");
}
