// lib/translate.ts — free-model-aggregator v2
//
// Terjemahan wire OpenAI <-> Anthropic <-> Gemini + renderer jawaban klien.
//
// Desain: BUFFERED, bukan streaming-through. Jawaban upstream dibaca utuh
// (JSON atau SSE), diekstrak menjadi tiga nilai (teks, usage, stop), lalu
// respons klien di-render dari tiga nilai itu. Tidak ada wall clock dan
// tidak ada randomness di file ini: id dan created jawaban klien TETAP,
// sehingga replay trial menghasilkan byte yang sama.
//
// Node 22, TypeScript, stdlib saja. Semua helper di sini murni/deterministik.

export type JsonRecord = Record<string, unknown>;

export interface ProviderToolCall {
  /** id tool call (OpenAI) / tool_use id (Anthropic) / nama+id Gemini. */
  id: string;
  /** Nama fungsi / tool. */
  name: string;
  /** Argumen sebagai string JSON (selalu string, agar sama di semua wire). */
  arguments: string;
}

export interface ProviderAnswer {
  /** Teks jawaban utama (tool calls tidak termasuk di sini). */
  text: string;
  /** Token input (prompt) menurut usage provider; 0 bila tidak ada. */
  inputTokens: number;
  /** Token output (completion) menurut usage provider; 0 bila tidak ada. */
  outputTokens: number;
  /** Stop mentah dari provider (finish_reason / stop_reason / finishReason). */
  stop: string;
  /** Tool calls yang dipertahankan, sesuai urutan kemunculannya. */
  toolCalls: ProviderToolCall[];
}

export interface TokenUsage {
  inputTokens: number;
  outputTokens: number;
}

/** Id tetap jawaban klien OpenAI. */
export const FIXED_OPENAI_ID = "chatcmpl-agg";
/** Id tetap jawaban klien Anthropic. */
export const FIXED_ANTHROPIC_ID = "msg_agg";
/** Created tetap jawaban klien (epoch, deterministik untuk replay). */
export const FIXED_CREATED = 1767225600;
/** Lebar potongan content pada render stream (karakter). */
export const STREAM_CHUNK_SIZE = 13;

// ---------------------------------------------------------------------------
// Helper kecil (internal)
// ---------------------------------------------------------------------------

function isRecord(value: unknown): value is JsonRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function asRecord(value: unknown): JsonRecord | null {
  return isRecord(value) ? value : null;
}

function asArray(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function asString(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function asNumber(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

function asStringArray(value: unknown): string[] {
  if (typeof value === "string") return [value];
  if (Array.isArray(value)) {
    return value.filter((item): item is string => typeof item === "string");
  }
  return [];
}

/**
 * Guard: model tujuan konversi tidak boleh kosong. Melempar TypeError lebih
 * awal daripada membangun request tanpa model (yang pasti ditolak provider
 * dengan 400). Validasi ini tidak mengubah perilaku pemakaian normal —
 * di sana model selalu string non-kosong.
 */
function requireModel(model: string): string {
  if (typeof model !== "string" || model.trim() === "") {
    throw new TypeError("translate: model must be a non-empty string");
  }
  return model;
}

/** Salinan JSON polos (tidak pernah memutasi body milik pemanggil). */
function cloneJson<T>(value: T): T {
  if (value === undefined) return value;
  return JSON.parse(JSON.stringify(value)) as T;
}

function safeJsonParse(text: string): JsonRecord | null {
  try {
    const parsed: unknown = JSON.parse(text);
    return isRecord(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

/** Gabungkan bagian-bagian teks dari sebuah content (string | array blok). */
function joinTextParts(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    let out = "";
    for (const part of content) {
      if (!isRecord(part)) continue;
      if (part.type === "text" && typeof part.text === "string") out += part.text;
    }
    return out;
  }
  return "";
}

/** Potong teks menjadi potongan-potongan berukuran tetap (untuk stream). */
export function splitTextChunks(text: string, size: number = STREAM_CHUNK_SIZE): string[] {
  const chunks: string[] = [];
  const width = size > 0 ? size : STREAM_CHUNK_SIZE;
  for (let i = 0; i < text.length; i += width) {
    chunks.push(text.slice(i, i + width));
  }
  return chunks;
}

// ---------------------------------------------------------------------------
// Ekstraktor teks konten
// ---------------------------------------------------------------------------

/**
 * Teks dari pesan OpenAI (choices[].message): content string, atau gabungan
 * blok {type:"text"} bila content berupa array. Bagian non-teks (mis. image)
 * diabaikan untuk teks utama.
 */
export function extractOpenAIText(message: unknown): string {
  const record = asRecord(message);
  if (!record) return "";
  return joinTextParts(record.content);
}

/**
 * Teks dari content Anthropic (array blok): gabungan blok {type:"text"}.
 * Blok tool_use / tool_result / image diabaikan untuk teks utama.
 */
export function extractAnthropicText(content: unknown): string {
  return joinTextParts(content);
}

/** Ekstraktor generik: string langsung, atau gabungan blok teks dari array. */
export function extractContentText(content: unknown): string {
  return joinTextParts(content);
}

/** Alias ekstraktor OpenAI. */
export const extractTextFromOpenAI = extractOpenAIText;
/** Alias ekstraktor Anthropic. */
export const extractTextFromAnthropic = extractAnthropicText;

/**
 * Tool calls OpenAI dari sebuah pesan: choices[].message.tool_calls dengan
 * id/type/function {name, arguments (string JSON)}.
 */
export function extractOpenAIToolCalls(message: unknown): ProviderToolCall[] {
  const record = asRecord(message);
  if (!record) return [];
  const out: ProviderToolCall[] = [];
  for (const call of asArray(record.tool_calls)) {
    if (!isRecord(call)) continue;
    const fn = asRecord(call.function);
    if (!fn) continue;
    out.push({
      id: asString(call.id),
      name: asString(fn.name),
      arguments: typeof fn.arguments === "string" ? fn.arguments : JSON.stringify(fn.arguments ?? {}),
    });
  }
  return out;
}

/**
 * Tool use Anthropic dari content: blok {type:"tool_use", id, name, input}.
 * Input (objek) diserialisasi menjadi string JSON agar seragam dengan OpenAI.
 */
export function extractAnthropicToolUse(content: unknown): ProviderToolCall[] {
  const out: ProviderToolCall[] = [];
  for (const block of asArray(content)) {
    if (!isRecord(block) || block.type !== "tool_use") continue;
    out.push({
      id: asString(block.id),
      name: asString(block.name),
      arguments: JSON.stringify(block.input ?? {}),
    });
  }
  return out;
}

// ---------------------------------------------------------------------------
// Konversi tool / tool_choice / stop (dipakai kedua arah)
// ---------------------------------------------------------------------------

/** OpenAI tools -> Anthropic tools ({name, description, input_schema}). */
export function openAIToolsToAnthropic(tools: unknown): JsonRecord[] {
  const out: JsonRecord[] = [];
  for (const tool of asArray(tools)) {
    if (!isRecord(tool)) continue;
    const fn = asRecord(tool.function);
    if (!fn) continue;
    const entry: JsonRecord = {
      name: asString(fn.name),
      input_schema: isRecord(fn.parameters)
        ? cloneJson(fn.parameters)
        : { type: "object", properties: {} },
    };
    if (typeof fn.description === "string") entry.description = fn.description;
    out.push(entry);
  }
  return out;
}

/** Anthropic tools -> OpenAI tools ({type:"function", function:{...}}). */
export function anthropicToolsToOpenAI(tools: unknown): JsonRecord[] {
  const out: JsonRecord[] = [];
  for (const tool of asArray(tools)) {
    if (!isRecord(tool)) continue;
    const entry: JsonRecord = {
      type: "function",
      function: {
        name: asString(tool.name),
        parameters: isRecord(tool.input_schema)
          ? cloneJson(tool.input_schema)
          : { type: "object", properties: {} },
      },
    };
    if (typeof tool.description === "string") {
      (entry.function as JsonRecord).description = tool.description;
    }
    out.push(entry);
  }
  return out;
}

/** Tools salah satu wire klien -> Gemini functionDeclarations. */
export function toolsToGeminiDeclarations(clientApi: string, tools: unknown): JsonRecord[] {
  const declarations: JsonRecord[] = [];
  if (clientApi === "anthropic") {
    for (const tool of asArray(tools)) {
      if (!isRecord(tool)) continue;
      const entry: JsonRecord = {
        name: asString(tool.name),
        parameters: isRecord(tool.input_schema)
          ? cloneJson(tool.input_schema)
          : { type: "object", properties: {} },
      };
      if (typeof tool.description === "string") entry.description = tool.description;
      declarations.push(entry);
    }
  } else {
    for (const tool of asArray(tools)) {
      if (!isRecord(tool)) continue;
      const fn = asRecord(tool.function);
      if (!fn) continue;
      const entry: JsonRecord = {
        name: asString(fn.name),
        parameters: isRecord(fn.parameters)
          ? cloneJson(fn.parameters)
          : { type: "object", properties: {} },
      };
      if (typeof fn.description === "string") entry.description = fn.description;
      declarations.push(entry);
    }
  }
  return declarations;
}

/** OpenAI tool_choice -> Anthropic tool_choice. */
export function openAIToolChoiceToAnthropic(choice: unknown): JsonRecord | undefined {
  if (choice === "auto") return { type: "auto" };
  if (choice === "none") return { type: "none" };
  if (choice === "required") return { type: "any" };
  if (isRecord(choice)) {
    if (choice.type === "function") {
      const fn = asRecord(choice.function);
      if (fn) return { type: "tool", name: asString(fn.name) };
    }
    if (choice.type === "tool") return { type: "tool", name: asString(choice.name) };
  }
  return undefined;
}

/** Anthropic tool_choice -> OpenAI tool_choice. */
export function anthropicToolChoiceToOpenAI(choice: unknown): unknown {
  if (!isRecord(choice)) return undefined;
  if (choice.type === "auto") return "auto";
  if (choice.type === "none") return "none";
  if (choice.type === "any") return "required";
  if (choice.type === "tool") {
    return { type: "function", function: { name: asString(choice.name) } };
  }
  return undefined;
}

// ---------------------------------------------------------------------------
// Konversi image
// ---------------------------------------------------------------------------

/** Satu blok image OpenAI -> blok image Anthropic. */
export function openAIImageToAnthropic(part: JsonRecord): JsonRecord | null {
  const imageUrl = asRecord(part.image_url);
  if (!imageUrl) return null;
  const url = asString(imageUrl.url);
  const dataMatch = /^data:([^;]+);base64,(.+)$/.exec(url);
  if (dataMatch) {
    return {
      type: "image",
      source: { type: "base64", media_type: dataMatch[1], data: dataMatch[2] },
    };
  }
  if (url !== "") {
    return { type: "image", source: { type: "url", url } };
  }
  return null;
}

/** Satu blok image Anthropic -> blok image OpenAI. */
export function anthropicImageToOpenAI(block: JsonRecord): JsonRecord | null {
  const source = asRecord(block.source);
  if (!source) return null;
  if (source.type === "base64") {
    const mediaType = asString(source.media_type) || "image/png";
    return {
      type: "image_url",
      image_url: { url: `data:${mediaType};base64,${asString(source.data)}` },
    };
  }
  if (source.type === "url" && asString(source.url) !== "") {
    return { type: "image_url", image_url: { url: asString(source.url) } };
  }
  return null;
}

/** Satu blok image (OpenAI atau Anthropic) -> part Gemini (inlineData/fileData). */
export function imageToGeminiPart(part: JsonRecord): JsonRecord | null {
  // Bentuk OpenAI: {type:"image_url", image_url:{url}}.
  const imageUrl = asRecord(part.image_url);
  if (imageUrl) {
    const url = asString(imageUrl.url);
    const dataMatch = /^data:([^;]+);base64,(.+)$/.exec(url);
    if (dataMatch) {
      return { inlineData: { mimeType: dataMatch[1], data: dataMatch[2] } };
    }
    if (url !== "") return { fileData: { fileUri: url } };
    return null;
  }
  // Bentuk Anthropic: {type:"image", source:{type:"base64"/"url", ...}}.
  const source = asRecord(part.source);
  if (source) {
    if (source.type === "base64") {
      return {
        inlineData: {
          mimeType: asString(source.media_type) || "image/png",
          data: asString(source.data),
        },
      };
    }
    if (source.type === "url" && asString(source.url) !== "") {
      return { fileData: { fileUri: asString(source.url) } };
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// Konversi pesan
// ---------------------------------------------------------------------------

/** Pesan OpenAI -> blok-blok content Anthropic (teks / image / tool_use). */
function openAIMessageToAnthropicBlocks(message: JsonRecord): JsonRecord[] {
  const blocks: JsonRecord[] = [];
  const content = message.content;
  if (typeof content === "string" && content !== "") {
    blocks.push({ type: "text", text: content });
  } else if (Array.isArray(content)) {
    for (const part of content) {
      if (!isRecord(part)) continue;
      if (part.type === "text" && typeof part.text === "string") {
        blocks.push({ type: "text", text: part.text });
      } else if (part.type === "image_url") {
        const image = openAIImageToAnthropic(part);
        if (image) blocks.push(image);
      }
    }
  }
  // Tool calls asisten -> blok tool_use, urutan dipertahankan setelah teks.
  for (const call of extractOpenAIToolCalls(message)) {
    let input: unknown = {};
    try {
      input = JSON.parse(call.arguments);
    } catch {
      input = {};
    }
    blocks.push({ type: "tool_use", id: call.id, name: call.name, input });
  }
  return blocks;
}

/** Blok content Anthropic -> pesan OpenAI (teks + image, tanpa tool_result). */
function anthropicBlocksToOpenAIContent(blocks: unknown[]): JsonRecord[] {
  const parts: JsonRecord[] = [];
  for (const block of blocks) {
    if (!isRecord(block)) continue;
    if (block.type === "text" && typeof block.text === "string") {
      parts.push({ type: "text", text: block.text });
    } else if (block.type === "image") {
      const image = anthropicImageToOpenAI(block);
      if (image) parts.push(image);
    }
  }
  return parts;
}

// ---------------------------------------------------------------------------
// openAIToAnthropic / anthropicToOpenAI
// ---------------------------------------------------------------------------

/**
 * Konversi body OpenAI Chat Completions -> body Anthropic Messages.
 * System (pesan role system) pindah ke field top-level `system` (string).
 * Tool calls dipertahankan sebagai blok tool_use; pesan role tool menjadi
 * blok tool_result di dalam pesan user. JSON mode dipertahankan: penanda
 * response_format tidak dihilangkan dan instruksi JSON ditambahkan ke
 * system agar provider tetap mengembalikan JSON.
 */
export function openAIToAnthropic(body: JsonRecord, model: string): JsonRecord {
  const input = cloneJson(body);
  const out: JsonRecord = { model: requireModel(model) };

  const systemTexts: string[] = [];
  const messages: JsonRecord[] = [];

  for (const raw of asArray(input.messages)) {
    if (!isRecord(raw)) continue;
    const role = asString(raw.role);
    if (role === "system") {
      const text = joinTextParts(raw.content);
      if (text !== "") systemTexts.push(text);
      continue;
    }
    if (role === "tool") {
      // Pesan tool OpenAI -> blok tool_result dalam pesan user Anthropic.
      const result: JsonRecord = {
        type: "tool_result",
        tool_use_id: asString(raw.tool_call_id),
        content: joinTextParts(raw.content),
      };
      messages.push({ role: "user", content: [result] });
      continue;
    }
    const anthropicRole = role === "assistant" ? "assistant" : "user";
    const blocks = openAIMessageToAnthropicBlocks(raw);
    messages.push({ role: anthropicRole, content: blocks });
  }

  // JSON mode: jaga agar provider tetap mengembalikan JSON.
  const responseFormat = asRecord(input.response_format);
  const jsonMode = responseFormat !== null && responseFormat.type === "json_object";
  if (jsonMode) {
    systemTexts.push("Respond with valid JSON only.");
    // Penanda response_format dipertahankan apa adanya.
    out.response_format = cloneJson(responseFormat);
  }
  if (systemTexts.length > 0) out.system = systemTexts.join("\n");

  out.messages = messages;

  if (typeof input.max_tokens === "number") out.max_tokens = input.max_tokens;
  if (typeof input.temperature === "number") out.temperature = input.temperature;
  if (typeof input.top_p === "number") out.top_p = input.top_p;
  if (input.stream === true) out.stream = true;

  const stopSequences = asStringArray(input.stop);
  if (stopSequences.length > 0) out.stop_sequences = stopSequences;

  const tools = openAIToolsToAnthropic(input.tools);
  if (tools.length > 0) out.tools = tools;

  const toolChoice = openAIToolChoiceToAnthropic(input.tool_choice);
  if (toolChoice !== undefined) out.tool_choice = toolChoice;

  return out;
}

/**
 * Konversi body Anthropic Messages -> body OpenAI Chat Completions.
 * Field top-level `system` (string) menjadi pesan role system pertama.
 * Blok tool_use menjadi tool_calls asisten; blok tool_result menjadi
 * pesan role tool. JSON mode: response_format yang ada dipertahankan.
 */
export function anthropicToOpenAI(body: JsonRecord, model: string): JsonRecord {
  const input = cloneJson(body);
  const out: JsonRecord = { model: requireModel(model) };
  const messages: JsonRecord[] = [];

  const system = input.system;
  if (typeof system === "string" && system !== "") {
    messages.push({ role: "system", content: system });
  } else if (Array.isArray(system)) {
    const text = joinTextParts(system);
    if (text !== "") messages.push({ role: "system", content: text });
  }

  for (const raw of asArray(input.messages)) {
    if (!isRecord(raw)) continue;
    const role = asString(raw.role);
    // Content Anthropic boleh string polos ATAU array blok. String polos
    // harus dinormalisasi jadi satu blok teks — asArray(string) memberi []
    // sehingga pesan user/asisten string akan hilang (user kosong).
    const rawContent = raw.content;
    const blocks: unknown[] =
      typeof rawContent === "string"
        ? rawContent === ""
          ? []
          : [{ type: "text", text: rawContent }]
        : asArray(rawContent);

    const toolResults: JsonRecord[] = [];
    const plainParts = anthropicBlocksToOpenAIContent(blocks);
    const toolUse = extractAnthropicToolUse(blocks);
    for (const block of blocks) {
      if (isRecord(block) && block.type === "tool_result") {
        toolResults.push(block);
      }
    }

    if (role === "assistant") {
      const text = joinTextParts(blocks);
      const message: JsonRecord = { role: "assistant" };
      // content null bila asisten hanya memanggil tool (tanpa teks).
      message.content = text !== "" ? text : null;
      if (toolUse.length > 0) {
        message.tool_calls = toolUse.map((call) => ({
          id: call.id,
          type: "function",
          function: { name: call.name, arguments: call.arguments },
        }));
      }
      messages.push(message);
      continue;
    }

    // Pesan user: teks/image jadi satu pesan user; tiap tool_result jadi
    // pesan role tool sendiri, urutan dipertahankan (teks dulu, lalu hasil).
    if (plainParts.length > 0) {
      const content = plainParts.length === 1 && plainParts[0].type === "text"
        ? asString(plainParts[0].text)
        : plainParts;
      messages.push({ role: "user", content });
    }
    for (const result of toolResults) {
      messages.push({
        role: "tool",
        tool_call_id: asString(result.tool_use_id),
        content: joinTextParts(result.content),
      });
    }
    if (plainParts.length === 0 && toolResults.length === 0) {
      messages.push({ role: "user", content: "" });
    }
  }

  out.messages = messages;

  if (typeof input.max_tokens === "number") out.max_tokens = input.max_tokens;
  if (typeof input.temperature === "number") out.temperature = input.temperature;
  if (typeof input.top_p === "number") out.top_p = input.top_p;
  if (input.stream === true) out.stream = true;

  const stopSequences = asStringArray(input.stop_sequences);
  if (stopSequences.length > 0) out.stop = stopSequences;

  const tools = anthropicToolsToOpenAI(input.tools);
  if (tools.length > 0) out.tools = tools;

  const toolChoice = anthropicToolChoiceToOpenAI(input.tool_choice);
  if (toolChoice !== undefined) out.tool_choice = toolChoice;

  const responseFormat = asRecord(input.response_format);
  if (responseFormat !== null) out.response_format = cloneJson(responseFormat);

  return out;
}

// ---------------------------------------------------------------------------
// toGemini
// ---------------------------------------------------------------------------

export interface GeminiRequest {
  /** Path tanpa host, mis. /v1beta/models/<model>:generateContent. */
  path: string;
  /** Body JSON untuk request Gemini. */
  body: JsonRecord;
}

/**
 * Bangun request Gemini dari body klien (OpenAI atau Anthropic).
 * Roles user/model; system dipisah ke systemInstruction; tools menjadi
 * functionDeclarations; JSON mode memakai responseMimeType application/json;
 * max_tokens -> maxOutputTokens; temperature/topP diteruskan bila ada;
 * stop/stop_sequences -> stopSequences.
 */
export function toGemini(
  clientApi: string,
  body: JsonRecord,
  model: string,
  stream: boolean,
): GeminiRequest {
  const input = cloneJson(body);
  model = requireModel(model);
  const path = stream
    ? `/v1beta/models/${model}:streamGenerateContent`
    : `/v1beta/models/${model}:generateContent`;

  // Normalisasi ke bentuk OpenAI dulu bila klien Anthropic, agar satu jalur
  // konversi pesan yang sama dipakai untuk kedua wire klien.
  const openaiBody: JsonRecord =
    clientApi === "anthropic" ? anthropicToOpenAI(input, model) : input;

  const contents: JsonRecord[] = [];
  const systemTexts: string[] = [];
  for (const raw of asArray(openaiBody.messages)) {
    if (!isRecord(raw)) continue;
    if (asString(raw.role) === "system") {
      const text = joinTextParts(raw.content);
      if (text !== "") systemTexts.push(text);
      continue;
    }
    const role = asString(raw.role) === "assistant" ? "model" : "user";
    const parts: JsonRecord[] = [];
    const content = raw.content;
    if (typeof content === "string" && content !== "") {
      parts.push({ text: content });
    } else if (Array.isArray(content)) {
      for (const part of content) {
        if (!isRecord(part)) continue;
        if (part.type === "text" && typeof part.text === "string") {
          parts.push({ text: part.text });
        } else if (part.type === "image_url") {
          const imagePart = imageToGeminiPart(part);
          if (imagePart) parts.push(imagePart);
        }
      }
    }
    // Tool calls asisten -> functionCall parts (urutan setelah teks).
    for (const call of extractOpenAIToolCalls(raw)) {
      let args: unknown = {};
      try {
        args = JSON.parse(call.arguments);
      } catch {
        args = {};
      }
      parts.push({ functionCall: { name: call.name, args } });
    }
    // Hasil tool (pesan role tool) -> functionResponse parts.
    if (asString(raw.role) === "tool") {
      const response: JsonRecord = { name: "" };
      // Nama fungsi tidak dibawa pesan tool OpenAI; isi tetap diteruskan
      // sebagai hasil generik agar tidak ada data yang hilang.
      response.response = { result: joinTextParts(raw.content) };
      parts.push({ functionResponse: response });
    }
    if (parts.length > 0) contents.push({ role, parts });
  }

  const geminiBody: JsonRecord = { contents };

  if (systemTexts.length > 0) {
    geminiBody.systemInstruction = { parts: [{ text: systemTexts.join("\n") }] };
  }

  const declarations = toolsToGeminiDeclarations(
    clientApi === "anthropic" ? "anthropic" : "openai",
    clientApi === "anthropic" ? input.tools : openaiBody.tools,
  );
  if (declarations.length > 0) {
    geminiBody.tools = [{ functionDeclarations: declarations }];
  }

  const generationConfig: JsonRecord = {};
  if (typeof openaiBody.max_tokens === "number") {
    generationConfig.maxOutputTokens = openaiBody.max_tokens;
  }
  if (typeof openaiBody.temperature === "number") {
    generationConfig.temperature = openaiBody.temperature;
  }
  if (typeof openaiBody.top_p === "number") {
    generationConfig.topP = openaiBody.top_p;
  }
  const stopSequences = asStringArray(
    clientApi === "anthropic" ? input.stop_sequences : openaiBody.stop,
  );
  if (stopSequences.length > 0) generationConfig.stopSequences = stopSequences;

  // JSON mode -> responseMimeType application/json.
  const responseFormat = asRecord(openaiBody.response_format);
  if (responseFormat !== null && responseFormat.type === "json_object") {
    generationConfig.responseMimeType = "application/json";
  }
  if (Object.keys(generationConfig).length > 0) {
    geminiBody.generationConfig = generationConfig;
  }

  return { path, body: geminiBody };
}

// ---------------------------------------------------------------------------
// Parser jawaban provider (JSON utuh)
// ---------------------------------------------------------------------------

function emptyAnswer(): ProviderAnswer {
  return { text: "", inputTokens: 0, outputTokens: 0, stop: "", toolCalls: [] };
}

/** Parser jawaban JSON OpenAI: choices[0].message + usage + finish_reason. */
export function fromOpenAI(body: unknown): ProviderAnswer {
  const answer = emptyAnswer();
  const record = asRecord(body);
  if (!record) return answer;
  const choices = asArray(record.choices);
  const first = asRecord(choices[0]);
  if (first) {
    const message = asRecord(first.message);
    if (message) {
      answer.text = extractOpenAIText(message);
      answer.toolCalls = extractOpenAIToolCalls(message);
    }
    answer.stop = asString(first.finish_reason);
  }
  const usage = asRecord(record.usage);
  if (usage) {
    answer.inputTokens = asNumber(usage.prompt_tokens);
    answer.outputTokens = asNumber(usage.completion_tokens);
  }
  return answer;
}

/** Parser jawaban JSON Anthropic: content blocks + usage + stop_reason. */
export function fromAnthropic(body: unknown): ProviderAnswer {
  const answer = emptyAnswer();
  const record = asRecord(body);
  if (!record) return answer;
  answer.text = extractAnthropicText(record.content);
  answer.toolCalls = extractAnthropicToolUse(record.content);
  answer.stop = asString(record.stop_reason);
  const usage = asRecord(record.usage);
  if (usage) {
    answer.inputTokens = asNumber(usage.input_tokens);
    answer.outputTokens = asNumber(usage.output_tokens);
  }
  return answer;
}

/** Parser jawaban JSON Gemini: candidates[0] + usageMetadata + finishReason. */
export function fromGemini(body: unknown): ProviderAnswer {
  const answer = emptyAnswer();
  const record = asRecord(body);
  if (!record) return answer;
  const candidates = asArray(record.candidates);
  const first = asRecord(candidates[0]);
  if (first) {
    const content = asRecord(first.content);
    if (content) {
      let text = "";
      for (const part of asArray(content.parts)) {
        if (!isRecord(part)) continue;
        if (typeof part.text === "string") text += part.text;
        const call = asRecord(part.functionCall);
        if (call) {
          answer.toolCalls.push({
            id: asString(call.name),
            name: asString(call.name),
            arguments: JSON.stringify(call.args ?? {}),
          });
        }
      }
      answer.text = text;
    }
    answer.stop = asString(first.finishReason);
  }
  const usage = asRecord(record.usageMetadata);
  if (usage) {
    answer.inputTokens = asNumber(usage.promptTokenCount);
    answer.outputTokens = asNumber(usage.candidatesTokenCount);
  }
  return answer;
}

// ---------------------------------------------------------------------------
// parseProviderSse — kumpulkan teks + usage + stop dari frame SSE per wire
// ---------------------------------------------------------------------------

/** Urai teks SSE mentah menjadi daftar frame JSON {event, data}. */
function parseSseFrames(text: string): Array<{ event: string; data: JsonRecord | null }> {
  const frames: Array<{ event: string; data: JsonRecord | null }> = [];
  // Frame dipisah baris kosong; satu frame bisa membawa beberapa baris data:.
  const rawFrames = text.split(/\r?\n\r?\n/);
  for (const raw of rawFrames) {
    if (raw.trim() === "") continue;
    let event = "";
    const dataLines: string[] = [];
    for (const line of raw.split(/\r?\n/)) {
      if (line.startsWith("event:")) event = line.slice("event:".length).trim();
      else if (line.startsWith("data:")) dataLines.push(line.slice("data:".length).trim());
    }
    if (dataLines.length === 0) continue;
    const joined = dataLines.join("\n");
    if (joined === "[DONE]") continue;
    frames.push({ event, data: safeJsonParse(joined) });
  }
  return frames;
}

/**
 * Kumpulkan satu ProviderAnswer dari teks SSE provider.
 * - openai: choices delta/content + usage di chunk akhir (bila stream_options
 *   include_usage); delta.tool_calls digabung per index.
 * - anthropic: content_block_delta (teks / input_json_delta) + message_delta
 *   (usage output + stop_reason); message_start membawa usage input.
 * - gemini: candidates[].content.parts teks + usageMetadata.
 */
export function parseProviderSse(wire: string, text: string): ProviderAnswer {
  const answer = emptyAnswer();
  const frames = parseSseFrames(text);

  if (wire === "openai") {
    const toolFragments = new Map<number, { id: string; name: string; arguments: string }>();
    for (const frame of frames) {
      const data = frame.data;
      if (!data) continue;
      const usage = asRecord(data.usage);
      if (usage) {
        answer.inputTokens = asNumber(usage.prompt_tokens);
        answer.outputTokens = asNumber(usage.completion_tokens);
      }
      const choices = asArray(data.choices);
      const first = asRecord(choices[0]);
      if (!first) continue;
      const delta = asRecord(first.delta);
      if (delta) {
        if (typeof delta.content === "string") answer.text += delta.content;
        for (const call of asArray(delta.tool_calls)) {
          if (!isRecord(call)) continue;
          const index = typeof call.index === "number" ? call.index : 0;
          const fn = asRecord(call.function);
          const prev = toolFragments.get(index) ?? { id: "", name: "", arguments: "" };
          if (typeof call.id === "string" && call.id !== "") prev.id = call.id;
          if (fn) {
            if (typeof fn.name === "string" && fn.name !== "") prev.name = fn.name;
            if (typeof fn.arguments === "string") prev.arguments += fn.arguments;
          }
          toolFragments.set(index, prev);
        }
      }
      const finish = asString(first.finish_reason);
      if (finish !== "") answer.stop = finish;
    }
    const sorted = [...toolFragments.entries()].sort((a, b) => a[0] - b[0]);
    for (const [, fragment] of sorted) {
      answer.toolCalls.push({
        id: fragment.id,
        name: fragment.name,
        arguments: fragment.arguments === "" ? "{}" : fragment.arguments,
      });
    }
    return answer;
  }

  if (wire === "anthropic") {
    // tool_use per index blok: kumpulkan id/nama dari content_block_start,
    // lalu gabung partial_json dari input_json_delta.
    const toolBlocks = new Map<number, { id: string; name: string; json: string }>();
    for (const frame of frames) {
      const data = frame.data;
      if (!data) continue;
      if (frame.event === "message_start") {
        const message = asRecord(data.message);
        const usage = message ? asRecord(message.usage) : null;
        if (usage) answer.inputTokens = asNumber(usage.input_tokens);
      }
      if (frame.event === "content_block_start") {
        const block = asRecord(data.content_block);
        const index = typeof data.index === "number" ? data.index : 0;
        if (block && block.type === "tool_use") {
          toolBlocks.set(index, {
            id: asString(block.id),
            name: asString(block.name),
            json: "",
          });
        }
      }
      if (frame.event === "content_block_delta") {
        const delta = asRecord(data.delta);
        const index = typeof data.index === "number" ? data.index : 0;
        if (delta) {
          if (delta.type === "text_delta" && typeof delta.text === "string") {
            answer.text += delta.text;
          }
          if (delta.type === "input_json_delta" && typeof delta.partial_json === "string") {
            const prev = toolBlocks.get(index);
            if (prev) prev.json += delta.partial_json;
          }
        }
      }
      if (frame.event === "message_delta") {
        const delta = asRecord(data.delta);
        if (delta) {
          const stop = asString(delta.stop_reason);
          if (stop !== "") answer.stop = stop;
        }
        const usage = asRecord(data.usage);
        if (usage) answer.outputTokens = asNumber(usage.output_tokens);
      }
      // Fallback: frame tanpa event yang membawa content/usage langsung.
      if (frame.event === "") {
        if (typeof data.text === "string") answer.text += data.text;
      }
    }
    const sorted = [...toolBlocks.entries()].sort((a, b) => a[0] - b[0]);
    for (const [, block] of sorted) {
      answer.toolCalls.push({
        id: block.id,
        name: block.name,
        arguments: block.json === "" ? "{}" : block.json,
      });
    }
    return answer;
  }

  // Wire tidak dikenal: tidak ada frame yang bisa diartikan dengan benar,
  // jadi kembalikan jawaban kosong daripada menebak wire lain (guard
  // deterministik — pemanggil yang salah wire langsung terlihat dari
  // teks/usage/stop yang kosong, bukan dari hasil tebak-tebakan).
  if (wire !== "gemini") return answer;

  // gemini
  for (const frame of frames) {
    const data = frame.data;
    if (!data) continue;
    for (const candidate of asArray(data.candidates)) {
      if (!isRecord(candidate)) continue;
      const content = asRecord(candidate.content);
      if (content) {
        for (const part of asArray(content.parts)) {
          if (!isRecord(part)) continue;
          if (typeof part.text === "string") answer.text += part.text;
          const call = asRecord(part.functionCall);
          if (call) {
            answer.toolCalls.push({
              id: asString(call.name),
              name: asString(call.name),
              arguments: JSON.stringify(call.args ?? {}),
            });
          }
        }
      }
      const finish = asString(candidate.finishReason);
      if (finish !== "") answer.stop = finish;
    }
    const usage = asRecord(data.usageMetadata);
    if (usage) {
      answer.inputTokens = asNumber(usage.promptTokenCount);
      answer.outputTokens = asNumber(usage.candidatesTokenCount);
    }
  }
  return answer;
}

// ---------------------------------------------------------------------------
// clientStop — petakan stop mentah provider ke stop klien
// ---------------------------------------------------------------------------

/**
 * Petakan stop mentah ke keluarga stop klien.
 * clientApi "openai": stop / length / tool_calls / content_filter.
 * clientApi "anthropic": end_turn / max_tokens / tool_use / refusal.
 * Nilai mentah dari wire mana pun diterima (stop, end_turn, length,
 * max_tokens, tool_calls, tool_use, content_filter, refusal); default
 * mengikuti klien: stop (OpenAI) / end_turn (Anthropic).
 */
export function clientStop(stop: string, clientApi: string): string {
  const raw = (stop || "").trim();
  const anthropic = clientApi === "anthropic";
  if (raw === "length" || raw === "max_tokens") return anthropic ? "max_tokens" : "length";
  if (raw === "tool_calls" || raw === "tool_use") return anthropic ? "tool_use" : "tool_calls";
  if (raw === "content_filter" || raw === "refusal") {
    return anthropic ? "refusal" : "content_filter";
  }
  if (raw === "stop" || raw === "end_turn" || raw === "") {
    return anthropic ? "end_turn" : "stop";
  }
  // Stop sequence kustom (teks biasa) dihitung sebagai stop normal.
  return anthropic ? "end_turn" : "stop";
}

// ---------------------------------------------------------------------------
// Renderer jawaban klien
// ---------------------------------------------------------------------------

/** Normalisasi argumen renderer ke (model, answer, ekstra). */
function normalizeRenderArgs(
  a: string,
  b: ProviderAnswer | TokenUsage,
  c?: string,
  d?: string,
  e?: ProviderToolCall[],
): { model: string; answer: ProviderAnswer } {
  if (isRecord(b) && typeof (b as ProviderAnswer).text === "string") {
    return { model: a, answer: b as ProviderAnswer };
  }
  const usage = b as TokenUsage;
  return {
    model: d ?? "",
    answer: {
      text: a,
      inputTokens: usage.inputTokens ?? 0,
      outputTokens: usage.outputTokens ?? 0,
      stop: c ?? "",
      toolCalls: e ?? [],
    },
  };
}

function openAIToolCallsJson(toolCalls: ProviderToolCall[]): JsonRecord[] {
  return toolCalls.map((call) => ({
    id: call.id,
    type: "function",
    function: { name: call.name, arguments: call.arguments },
  }));
}

function anthropicContentJson(
  text: string,
  toolCalls: ProviderToolCall[],
): JsonRecord[] {
  const content: JsonRecord[] = [];
  if (text !== "") content.push({ type: "text", text });
  for (const call of toolCalls) {
    let input: unknown = {};
    try {
      input = JSON.parse(call.arguments);
    } catch {
      input = {};
    }
    content.push({ type: "tool_use", id: call.id, name: call.name, input });
  }
  return content;
}

/**
 * Renderer OpenAI JSON: {id chatcmpl-agg, object chat.completion, created
 * 1767225600, model, choices[{index 0, message{role assistant, content,
 * tool_calls bila ada}, finish_reason}], usage{prompt/completion/total}}.
 * Dipanggil sebagai renderOpenAIJson(model, answer) atau
 * renderOpenAIJson(text, usage, stop, model, toolCalls?).
 */
export function renderOpenAIJson(
  modelOrText: string,
  answerOrUsage: ProviderAnswer | TokenUsage,
  stop?: string,
  model?: string,
  toolCalls?: ProviderToolCall[],
): JsonRecord {
  const { model: resolvedModel, answer } = normalizeRenderArgs(
    modelOrText,
    answerOrUsage,
    stop,
    model,
    toolCalls,
  );
  const message: JsonRecord = { role: "assistant", content: answer.text };
  if (answer.toolCalls.length > 0) {
    message.tool_calls = openAIToolCallsJson(answer.toolCalls);
  }
  return {
    id: FIXED_OPENAI_ID,
    object: "chat.completion",
    created: FIXED_CREATED,
    model: resolvedModel,
    choices: [
      {
        index: 0,
        message,
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

/**
 * Renderer OpenAI stream (BUFFERED): jawaban utuh dipecah jadi chunk
 * 13 karakter. Urutan: chunk role, chunk-chunk content, chunk tool_calls
 * (bila ada), chunk finish (finish_reason), chunk usage terakhir (bila
 * stream_options include_usage). Diakhiri `data: [DONE]`.
 * Mengembalikan satu string SSE utuh.
 * Dipanggil sebagai renderOpenAIStream(model, answer, includeUsage?) atau
 * renderOpenAIStream(text, usage, stop, model, toolCalls?, includeUsage?).
 */
export function renderOpenAIStream(
  modelOrText: string,
  answerOrUsage: ProviderAnswer | TokenUsage,
  stopOrIncludeUsage?: string | boolean,
  modelOrIncludeUsage?: string | boolean,
  toolCallsOrIncludeUsage?: ProviderToolCall[] | boolean,
  includeUsageArg?: boolean,
): string {
  return renderOpenAIStreamChunks(
    modelOrText,
    answerOrUsage,
    stopOrIncludeUsage,
    modelOrIncludeUsage,
    toolCallsOrIncludeUsage,
    includeUsageArg,
  ).join("");
}

/** Varian array: tiap elemen satu frame SSE (`data: {...}\n\n` / `[DONE]`). */
export function renderOpenAIStreamChunks(
  modelOrText: string,
  answerOrUsage: ProviderAnswer | TokenUsage,
  stopOrIncludeUsage?: string | boolean,
  modelOrIncludeUsage?: string | boolean,
  toolCallsOrIncludeUsage?: ProviderToolCall[] | boolean,
  includeUsageArg?: boolean,
): string[] {
  let stop: string | undefined;
  let model: string | undefined;
  let toolCalls: ProviderToolCall[] | undefined;
  let includeUsage = false;
  if (typeof stopOrIncludeUsage === "boolean") {
    includeUsage = stopOrIncludeUsage;
  } else {
    stop = stopOrIncludeUsage;
    if (typeof modelOrIncludeUsage === "boolean") includeUsage = modelOrIncludeUsage;
    else model = modelOrIncludeUsage;
    if (typeof toolCallsOrIncludeUsage === "boolean") includeUsage = toolCallsOrIncludeUsage;
    else toolCalls = toolCallsOrIncludeUsage;
    if (includeUsageArg === true) includeUsage = true;
  }
  const { model: resolvedModel, answer } = normalizeRenderArgs(
    modelOrText,
    answerOrUsage,
    stop,
    model,
    toolCalls,
  );
  const chunks: string[] = [];
  const sse = (payload: JsonRecord): string => `data: ${JSON.stringify(payload)}\n\n`;

  chunks.push(
    sse({
      id: FIXED_OPENAI_ID,
      object: "chat.completion.chunk",
      created: FIXED_CREATED,
      model: resolvedModel,
      choices: [{ index: 0, delta: { role: "assistant" }, finish_reason: null }],
    }),
  );
  for (const piece of splitTextChunks(answer.text)) {
    chunks.push(
      sse({
        id: FIXED_OPENAI_ID,
        object: "chat.completion.chunk",
        created: FIXED_CREATED,
        model: resolvedModel,
        choices: [{ index: 0, delta: { content: piece }, finish_reason: null }],
      }),
    );
  }
  if (answer.toolCalls.length > 0) {
    chunks.push(
      sse({
        id: FIXED_OPENAI_ID,
        object: "chat.completion.chunk",
        created: FIXED_CREATED,
        model: resolvedModel,
        choices: [
          {
            index: 0,
            delta: { tool_calls: openAIToolCallsJson(answer.toolCalls) },
            finish_reason: null,
          },
        ],
      }),
    );
  }
  chunks.push(
    sse({
      id: FIXED_OPENAI_ID,
      object: "chat.completion.chunk",
      created: FIXED_CREATED,
      model: resolvedModel,
      choices: [{ index: 0, delta: {}, finish_reason: clientStop(answer.stop, "openai") }],
    }),
  );
  if (includeUsage) {
    chunks.push(
      sse({
        id: FIXED_OPENAI_ID,
        object: "chat.completion.chunk",
        created: FIXED_CREATED,
        model: resolvedModel,
        choices: [],
        usage: {
          prompt_tokens: answer.inputTokens,
          completion_tokens: answer.outputTokens,
          total_tokens: answer.inputTokens + answer.outputTokens,
        },
      }),
    );
  }
  chunks.push("data: [DONE]\n\n");
  return chunks;
}

/**
 * Renderer Anthropic JSON: {id msg_agg, type message, role assistant,
 * content [teks/tool_use], model, stop_reason, stop_sequence bila ada,
 * usage{input_tokens, output_tokens}}.
 * Dipanggil sebagai renderAnthropicJson(model, answer, stopSequence?) atau
 * renderAnthropicJson(text, usage, stop, model, toolCalls?, stopSequence?).
 */
export function renderAnthropicJson(
  modelOrText: string,
  answerOrUsage: ProviderAnswer | TokenUsage,
  stopOrStopSequence?: string,
  modelOrStopSequence?: string,
  toolCalls?: ProviderToolCall[],
  stopSequenceArg?: string,
): JsonRecord {
  let stop: string | undefined;
  let model: string | undefined;
  let stopSequence: string | undefined;
  if (isRecord(answerOrUsage) && typeof (answerOrUsage as ProviderAnswer).text === "string") {
    stopSequence = stopOrStopSequence;
  } else {
    stop = stopOrStopSequence;
    model = modelOrStopSequence;
    stopSequence = stopSequenceArg;
  }
  const { model: resolvedModel, answer } = normalizeRenderArgs(
    modelOrText,
    answerOrUsage,
    stop,
    model,
    toolCalls,
  );
  const out: JsonRecord = {
    id: FIXED_ANTHROPIC_ID,
    type: "message",
    role: "assistant",
    content: anthropicContentJson(answer.text, answer.toolCalls),
    model: resolvedModel,
    stop_reason: clientStop(answer.stop, "anthropic"),
    usage: {
      input_tokens: answer.inputTokens,
      output_tokens: answer.outputTokens,
    },
  };
  if (stopSequence !== undefined && stopSequence !== "") {
    out.stop_sequence = stopSequence;
  }
  return out;
}

/**
 * Renderer Anthropic stream (BUFFERED): event message_start (usage input),
 * content_block_start/delta (teks 13 karakter per delta, tool_use bila ada),
 * message_delta (stop_reason + usage output), message_stop.
 * Mengembalikan satu string SSE utuh.
 * Dipanggil sebagai renderAnthropicStream(model, answer, stopSequence?)
 * atau renderAnthropicStream(text, usage, stop, model, toolCalls?,
 * stopSequence?).
 */
export function renderAnthropicStream(
  modelOrText: string,
  answerOrUsage: ProviderAnswer | TokenUsage,
  stopOrStopSequence?: string,
  modelOrStopSequence?: string,
  toolCalls?: ProviderToolCall[],
  stopSequenceArg?: string,
): string {
  return renderAnthropicStreamChunks(
    modelOrText,
    answerOrUsage,
    stopOrStopSequence,
    modelOrStopSequence,
    toolCalls,
    stopSequenceArg,
  ).join("");
}

/** Varian array: tiap elemen satu event SSE (`event: ...\ndata: {...}\n\n`). */
export function renderAnthropicStreamChunks(
  modelOrText: string,
  answerOrUsage: ProviderAnswer | TokenUsage,
  stopOrStopSequence?: string,
  modelOrStopSequence?: string,
  toolCalls?: ProviderToolCall[],
  stopSequenceArg?: string,
): string[] {
  let stop: string | undefined;
  let model: string | undefined;
  let stopSequence: string | undefined;
  if (isRecord(answerOrUsage) && typeof (answerOrUsage as ProviderAnswer).text === "string") {
    stopSequence = stopOrStopSequence;
  } else {
    stop = stopOrStopSequence;
    model = modelOrStopSequence;
    stopSequence = stopSequenceArg;
  }
  const { model: resolvedModel, answer } = normalizeRenderArgs(
    modelOrText,
    answerOrUsage,
    stop,
    model,
    toolCalls,
  );
  const events: string[] = [];
  const sse = (event: string, payload: JsonRecord): string =>
    `event: ${event}\ndata: ${JSON.stringify(payload)}\n\n`;

  events.push(
    sse("message_start", {
      type: "message_start",
      message: {
        id: FIXED_ANTHROPIC_ID,
        type: "message",
        role: "assistant",
        content: [],
        model: resolvedModel,
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.inputTokens, output_tokens: 0 },
      },
    }),
  );

  // Blok teks (index 0) bila ada teks.
  if (answer.text !== "") {
    events.push(
      sse("content_block_start", {
        type: "content_block_start",
        index: 0,
        content_block: { type: "text", text: "" },
      }),
    );
    for (const piece of splitTextChunks(answer.text)) {
      events.push(
        sse("content_block_delta", {
          type: "content_block_delta",
          index: 0,
          delta: { type: "text_delta", text: piece },
        }),
      );
    }
    events.push(
      sse("content_block_stop", { type: "content_block_stop", index: 0 }),
    );
  }

  // Blok tool_use, satu blok per tool call (index lanjut setelah teks).
  answer.toolCalls.forEach((call, offset) => {
    const index = (answer.text !== "" ? 1 : 0) + offset;
    events.push(
      sse("content_block_start", {
        type: "content_block_start",
        index,
        content_block: { type: "tool_use", id: call.id, name: call.name, input: {} },
      }),
    );
    events.push(
      sse("content_block_delta", {
        type: "content_block_delta",
        index,
        delta: { type: "input_json_delta", partial_json: call.arguments },
      }),
    );
    events.push(
      sse("content_block_stop", { type: "content_block_stop", index }),
    );
  });

  const delta: JsonRecord = { stop_reason: clientStop(answer.stop, "anthropic") };
  if (stopSequence !== undefined && stopSequence !== "") {
    delta.stop_sequence = stopSequence;
  }
  events.push(
    sse("message_delta", {
      type: "message_delta",
      delta,
      usage: { output_tokens: answer.outputTokens },
    }),
  );
  events.push(sse("message_stop", { type: "message_stop" }));
  return events;
}
