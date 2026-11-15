// main.ts — Entrypoint server free-model-aggregator-01 v2 (Node 22, stdlib saja).
//
// Alur: baca brief -> bangun katalog (probe provider) -> --list mencetak
// katalog dan keluar, atau: nyalakan HTTP server di port acak, daftarkan
// diri ke env, layani request klien satu per satu, polling /work sampai
// done:true, lalu tutup server dan exit 0.
//
// PERAN FILE INI (v2): main.ts HANYA mengeksekusi. Semua keputusan
// pemilihan provider/model tinggal di router.ts (global planner): main
// meminta rencana, mencoba provider yang direncanakan, mencatat hasil ke
// router, dan mengulang bila router me-re-draft rencana sisa trial.
// Tidak ada logika greedy di sini, tidak ada state persisten antar trial
// (semua state in-memory per proses), tidak ada penulisan file.
//
// Deterministik: tidak ada Date.now(), Math.random(), atau wall clock.
// Pilihan berasal dari brief + request + jawaban provider. Menunggu
// (setTimeout untuk polling) boleh; waktu tidak pernah ditulis ke request.
//
// Interface yang dipakai file ini (lihat README.md untuk daftar lengkap):
//   - ./lib/brief.ts     : parseBrief/readBriefFile/budgetRange/ratingRow/
//                          meetsRequirements/isFreeModel/EMPTY_BRIEF + tipe.
//   - ./lib/catalog.ts   : buildCatalog, authHeaders(auth, key, wire),
//                          servedFreeModelIds, EMPTY_CATALOG + tipe Catalog.
//                          TIDAK ADA openAIModelList/anthropicModelList:
//                          bentuk daftar GET /v1/models dibangun di sini.
//   - ./lib/router.ts    : class Router(deps: BriefPort, providers)
//                          (legalCandidates, draftPlan, nextProvider,
//                          recordSuccess/recordFailure/recordServed,
//                          isDead) + tipe RouterRequest/Candidate/BriefPort.
//                          Router TIDAK mengimpor brief.ts: main.ts
//                          mengadaptasi brief.ts ke BriefPort.
//   - ./lib/translate.ts : openAIToAnthropic(body, model),
//                          anthropicToOpenAI(body, model),
//                          toGemini(clientApi, body, model, stream) ->
//                          {path, body}, fromOpenAI/fromAnthropic/fromGemini,
//                          parseProviderSse(wire, teks),
//                          renderOpenAIJson/Stream(model, answer),
//                          renderAnthropicJson/Stream(model, answer).
//                          Pemetaan stop ada DI DALAM renderer (clientStop).

import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import {
  budgetRange,
  EMPTY_BRIEF,
  isFreeModel,
  meetsRequirements,
  ratingRow,
  readBriefFile,
  type Brief,
  type Requirements,
  type WorkloadRequest,
} from "./lib/brief.ts";
import {
  authHeaders,
  buildCatalog,
  EMPTY_CATALOG,
  servedFreeModelIds,
  type Brief as CatalogBrief,
  type Catalog,
} from "./lib/catalog.ts";
import {
  Router,
  type BriefPort,
  type Candidate,
  type ProviderEntry as RouterProviderEntry,
  type RouterRequest,
} from "./lib/router.ts";
import {
  anthropicToOpenAI,
  FIXED_CREATED,
  fromAnthropic,
  fromGemini,
  fromOpenAI,
  openAIToAnthropic,
  parseProviderSse,
  renderAnthropicJson,
  renderAnthropicStream,
  renderOpenAIJson,
  renderOpenAIStream,
  toGemini,
  type ProviderAnswer,
} from "./lib/translate.ts";

// ---------------------------------------------------------------------------
// Konstanta
// ---------------------------------------------------------------------------

/** Timeout satu panggilan upstream (provider). */
const UPSTREAM_TIMEOUT_MS = 10_000;
/** Interval polling GET /work. */
const POLL_INTERVAL_MS = 100;
/** Timeout register (dan satu polling /work). */
const REGISTER_TIMEOUT_MS = 5_000;
/** Batas body request klien: 32 MiB. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** Pengaman: percobaan maksimum per request klien (anti loop abadi). */
const MAX_ATTEMPTS_PER_REQUEST = 40;

// ---------------------------------------------------------------------------
// State per proses (in-memory saja; hilang saat proses keluar)
// ---------------------------------------------------------------------------

let brief: Brief = EMPTY_BRIEF;
let catalog: Catalog = EMPTY_CATALOG;
let router: Router | null = null;
let envUrl = "";
/** Nomor kedatangan request klien (0, 1, 2, ...), sesuai urutan tiba. */
let arrivalCount = 0;

// ---------------------------------------------------------------------------
// Adaptor brief.ts -> router.ts (BriefPort) + provider router dari katalog
// ---------------------------------------------------------------------------

/**
 * BriefPort untuk Router: keempat fungsi brief.ts, dengan bentuk yang
 * diminta router. ratingRow brief.ts mengembalikan RatingRow | undefined;
 * router menormalisasinya sendiri (rating 0 bila tak ada baris).
 */
const briefPort: BriefPort = {
  budgetRange: (providerId: string) => budgetRange(providerId),
  meetsRequirements: (providerId: string, modelId: string, req) =>
    meetsRequirements(ratingRow(brief, providerId, modelId), {
      tools: req.tools,
      vision: req.vision,
      json: req.json,
      minContext: req.minContext,
    }),
  ratingRow: (providerId: string, modelId: string) =>
    ratingRow(brief, providerId, modelId),
  isFreeModel: (providerId: string, modelId: string) =>
    isFreeModel(brief, providerId, modelId),
};

/**
 * Brief dalam bentuk yang diminta catalog.buildCatalog: field key_env dan
 * rating_milli/context_length (snake_case). brief.ts memakai keyEnv /
 * ratingMilli / contextLength (camelCase), jadi tanpa adaptasi ini katalog
 * akan kehilangan key_env dan context_length.
 */
function catalogBrief(): CatalogBrief {
  return {
    providers: brief.providers.map((entry) => ({
      id: entry.id,
      name: entry.name,
      wire: entry.wire,
      auth: entry.auth,
      key_env: entry.keyEnv,
      key: entry.key,
      path: entry.path,
    })),
    ratings: brief.ratings.map((row) => ({
      model: row.model,
      provider: row.provider,
      rating_milli: row.ratingMilli,
      context_length: row.contextLength,
      tools: row.tools,
      vision: row.vision,
      json: row.json,
      free: row.free,
    })),
  };
}

/**
 * Entri provider untuk Router, dari katalog: kemampuan (tools/vision/json)
 * diambil dari tabel rating brief (otoritas kemampuan); model tanpa baris
 * rating mendapat false — model seperti itu tidak pernah legal karena
 * isFreeModel juga false untuknya. context_length null -> 0.
 */
function routerProviders(): RouterProviderEntry[] {
  return catalog.providers.map((entry) => ({
    id: entry.id,
    wire: entry.wire,
    models: entry.models.map((model) => {
      const row = ratingRow(brief, entry.id, model.id);
      return {
        id: model.id,
        contextLength: model.context_length ?? 0,
        tools: row?.tools ?? false,
        vision: row?.vision ?? false,
        json: row?.json ?? false,
      };
    }),
  }));
}

type ClientApi = "openai" | "anthropic";

interface ClientRequest {
  api: ClientApi;
  stream: boolean;
  body: Record<string, unknown>;
}

// ---------------------------------------------------------------------------
// Util kecil
// ---------------------------------------------------------------------------

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

function sendJson(res: ServerResponse, status: number, value: unknown): void {
  const text = typeof value === "string" ? value : JSON.stringify(value);
  res.writeHead(status, { "content-type": "application/json" });
  res.end(text);
}

/** Error berbentuk wire klien (OpenAI vs Anthropic). */
function wireErrorBody(api: ClientApi, message: string): Record<string, unknown> {
  if (api === "anthropic") {
    return { type: "error", error: { type: "api_error", message } };
  }
  return { error: { message, type: "server_error", code: "server_error" } };
}

function sendWireError(res: ServerResponse, api: ClientApi, status: number, message: string): void {
  sendJson(res, status, wireErrorBody(api, message));
}

// ---------------------------------------------------------------------------
// ProviderLocks: mutex per provider — tidak pernah ada dua panggilan
// bersamaan ke provider yang sama (provider menolak panggilan kedua
// dengan 429 concurrent_requests, dan panggilan itu sia-sia).
// ---------------------------------------------------------------------------

const providerLocks = new Map<string, Promise<void>>();

async function withProviderLock<T>(providerId: string, fn: () => Promise<T>): Promise<T> {
  const previous = providerLocks.get(providerId) ?? Promise.resolve();
  let release!: () => void;
  const current = new Promise<void>((resolve) => {
    release = resolve;
  });
  // Rantai: janji yang disimpan selesai saat giliran ini selesai.
  providerLocks.set(
    providerId,
    previous.then(() => current),
  );
  await previous;
  try {
    return await fn();
  } finally {
    release();
  }
}

// ---------------------------------------------------------------------------
// requirementsFor + clientJsonMode
// ---------------------------------------------------------------------------

/** Benar bila body membawa response_format JSON (OpenAI). */
function isJsonResponseFormat(body: Record<string, unknown>): boolean {
  if (!isRecord(body.response_format)) return false;
  const type = body.response_format.type;
  return type === "json_object" || type === "json_schema";
}

/** Benar bila ada bagian konten gambar di body (kedua wire). */
function hasImageContent(body: Record<string, unknown>): boolean {
  const messages = body.messages;
  if (!Array.isArray(messages)) return false;
  for (const message of messages) {
    const content = isRecord(message) ? message.content : undefined;
    if (typeof content === "string") continue;
    if (!Array.isArray(content)) continue;
    for (const part of content) {
      if (!isRecord(part)) continue;
      // OpenAI: {type:"image_url", ...}; Anthropic: {type:"image", ...}.
      if (part.type === "image_url" || part.type === "image") return true;
    }
  }
  return false;
}

/**
 * Syarat untuk request kedatangan ke-`index`.
 * Entri manifest ke-n (r01..r20 berurutan) adalah otoritasnya; bila
 * manifest tak punya entri untuk nomor itu, tebak dari body: ada gambar
 * -> vision, array tools tak kosong -> tools, response_format JSON -> json.
 */
function requirementsFor(index: number, body: Record<string, unknown>): Requirements {
  const entry = brief.manifest[index];
  if (entry !== undefined) return { ...entry.requires };
  const guessed: Requirements = {};
  if (hasImageContent(body)) guessed.vision = true;
  if (Array.isArray(body.tools) && body.tools.length > 0) guessed.tools = true;
  if (isJsonResponseFormat(body)) guessed.json = true;
  return guessed;
}

/**
 * Mode JSON klien: hanya dihitung untuk API OpenAI yang body-nya membawa
 * response_format JSON. Untuk API Anthropic selalu false (wire Anthropic
 * tidak membawa response_format).
 */
function clientJsonMode(api: ClientApi, body: Record<string, unknown>): boolean {
  if (api !== "openai") return false;
  return isJsonResponseFormat(body);
}

/** Deskriptor request untuk router, dari entri manifest + wire klien. */
function toRouterRequest(
  entry: WorkloadRequest,
  manifestIndex: number,
  api: ClientApi,
): RouterRequest {
  const requires = entry.requires;
  return {
    id: entry.id,
    weight: entry.weight,
    manifestIndex,
    clientWire: api,
    tools: requires.tools === true,
    vision: requires.vision === true,
    json: requires.json === true,
    minContext: requires.minContext ?? 0,
  };
}

/**
 * Deskriptor request kedatangan ke-`index`: entri manifest adalah
 * otoritasnya; bila manifest tak punya entri untuk nomor itu, tebak dari
 * body. Bila manifest diam soal JSON tetapi body memintanya (hanya mungkin
 * di OpenAI), syarat json tetap dicatat agar kandidat legal terjaga.
 */
function describeRequest(index: number, api: ClientApi, body: Record<string, unknown>): RouterRequest {
  const entry = brief.manifest[index];
  if (entry !== undefined) {
    const req = toRouterRequest(entry, index, api);
    if (clientJsonMode(api, body)) req.json = true;
    return req;
  }
  const requires = requirementsFor(index, body);
  if (clientJsonMode(api, body)) requires.json = true;
  return {
    id: `r${String(index + 1).padStart(2, "0")}`,
    weight: 1,
    manifestIndex: index,
    clientWire: api,
    tools: requires.tools === true,
    vision: requires.vision === true,
    json: requires.json === true,
    minContext: requires.minContext ?? 0,
  };
}

/**
 * Daftar request tersisa untuk draf router: request berjalan (dengan
 * syarat dari body) + semua entri manifest sesudahnya (wire klien menurut
 * manifest). Bila manifest habis, hanya request berjalan.
 */
function remainingRequests(index: number, api: ClientApi, body: Record<string, unknown>): RouterRequest[] {
  const current = describeRequest(index, api, body);
  const rest: RouterRequest[] = [current];
  for (let i = index + 1; i < brief.manifest.length; i++) {
    const entry = brief.manifest[i];
    rest.push(toRouterRequest(entry, i, entry.api));
  }
  return rest;
}

// ---------------------------------------------------------------------------
// callUpstream: satu panggilan ke satu provider.
// ---------------------------------------------------------------------------

/**
 * Satu panggilan upstream ke provider pilihan router.
 * Mengembalikan jawaban provider yang sudah di-parse (bentuk opaque dari
 * translate.ts), atau null bila: fetch gagal/timeout, status bukan 2xx,
 * atau body tak ter-parse. Null berarti "coba kandidat berikut".
 */
async function callUpstream(
  choice: Candidate,
  request: ClientRequest,
): Promise<ProviderAnswer | null> {
  const provider = brief.providers.find((entry) => entry.id === choice.providerId);
  if (provider === undefined) return null;
  // Guard: pilihan tanpa nama model, atau provider tanpa path direktori
  // (brief cacat), tidak pernah di-dial — langsung gagal supaya router
  // pindah ke kandidat berikut tanpa membuang satu panggilan upstream.
  if (choice.modelId === "" || provider.path === "") return null;
  const stream = request.stream === true;
  const bodyWithModel: Record<string, unknown> = { ...request.body, model: choice.modelId };

  let path: string;
  let wireBody: unknown;
  if (provider.wire === "openai") {
    path = "/v1/chat/completions";
    wireBody =
      request.api === "anthropic" ? anthropicToOpenAI(bodyWithModel, choice.modelId) : bodyWithModel;
    if (stream && isRecord(wireBody)) {
      wireBody.stream = true;
      wireBody.stream_options = { include_usage: true };
    }
  } else if (provider.wire === "anthropic") {
    path = "/v1/messages";
    wireBody =
      request.api === "openai" ? openAIToAnthropic(bodyWithModel, choice.modelId) : bodyWithModel;
    if (stream && isRecord(wireBody)) wireBody.stream = true;
  } else {
    // Gemini: toGemini mengembalikan {path, body} yang sudah jadi
    // (varian stream di path, tanpa ?alt=sse); pakai apa adanya.
    const geminiRequest = toGemini(request.api, bodyWithModel, choice.modelId, stream);
    path = geminiRequest.path;
    wireBody = geminiRequest.body;
  }

  // authHeaders(auth, key, wire): 3 argumen; untuk wire Anthropic ia
  // sudah menambahkan header anthropic-version sendiri.
  const headers: Record<string, string> = {
    "content-type": "application/json",
    ...authHeaders(provider.auth, provider.key, provider.wire),
  };

  let response: Response;
  try {
    response = await fetch(envUrl + provider.path + path, {
      method: "POST",
      headers,
      body: JSON.stringify(wireBody),
      signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
    });
  } catch {
    return null;
  }
  if (!response.ok) return null;
  let text: string;
  try {
    text = await response.text();
  } catch {
    return null;
  }
  try {
    // parseProviderSse(wire, teks): wire dulu, lalu teks.
    if (stream) return parseProviderSse(provider.wire, text);
    const parsed: unknown = JSON.parse(text);
    if (provider.wire === "openai") return fromOpenAI(parsed);
    if (provider.wire === "anthropic") return fromAnthropic(parsed);
    return fromGemini(parsed);
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// Penanganan satu request klien (inti v2: eksekutor planner)
// ---------------------------------------------------------------------------

function renderAnswer(
  res: ServerResponse,
  api: ClientApi,
  stream: boolean,
  model: string,
  answer: ProviderAnswer,
): void {
  // Renderer translate.ts: (model, answer). Pemetaan stop mentah ke stop
  // klien (clientStop) sudah dilakukan di dalam renderer. Stream OpenAI
  // SELALU membawa chunk usage terakhir: usage upstream selalu tersedia
  // (main meminta stream_options include_usage ke provider), dan klien
  // memerlukannya — body klien sendiri tidak membawa stream_options.
  if (api === "openai") {
    if (stream) {
      res.writeHead(200, {
        "content-type": "text/event-stream",
        "cache-control": "no-cache",
        connection: "keep-alive",
      });
      res.end(renderOpenAIStream(model, answer, true));
    } else {
      sendJson(res, 200, renderOpenAIJson(model, answer));
    }
    return;
  }
  if (stream) {
    res.writeHead(200, {
      "content-type": "text/event-stream",
      "cache-control": "no-cache",
      connection: "keep-alive",
    });
    res.end(renderAnthropicStream(model, answer));
  } else {
    sendJson(res, 200, renderAnthropicJson(model, answer));
  }
}

/**
 * Satu request klien, dieksekusi dari rencana router:
 * draf rencana sisa trial -> coba provider yang direncanakan (satu
 * panggilan, di bawah mutex provider itu) -> sukses: catat dilayani
 * (recordServed: sukses provider + realized loss), render jawaban ke wire
 * klien, selesai; gagal: catat gagal ke provider ITU (providerId saja),
 * draf ulang rencana sisa, ulangi untuk request yang sama — retry ke
 * provider yang sama diizinkan bila router memilihnya lagi dengan risiko
 * terbaru. Tak ada kandidat sama sekali: error 503 berformat wire klien
 * ("all providers refused").
 *
 * Provider mati TIDAK PERNAH dicatat gagal: nextProvider hanya
 * mengembalikan provider hidup, jadi provider mati cukup tidak dipilih.
 * Mencatat gagal tanpa panggilan akan mengotori counter router
 * (callsMade/failuresSeen hanya boleh berubah untuk panggilan nyata).
 */
async function handleRequest(
  api: ClientApi,
  body: Record<string, unknown>,
  res: ServerResponse,
): Promise<void> {
  const index = arrivalCount;
  arrivalCount += 1;
  const current = router;
  if (current === null) {
    sendWireError(res, api, 503, "all providers refused");
    return;
  }
  const desc = describeRequest(index, api, body);
  const stream = body.stream === true;
  current.beginRequest(desc.id);
  // Sisa trial dari nomor ini ke depan adalah bahan draf router.
  const remaining: RouterRequest[] = remainingRequests(index, api, body);
  current.draftPlan(remaining);

  let attempts = 0;
  while (attempts < MAX_ATTEMPTS_PER_REQUEST) {
    attempts += 1;
    const choice = current.nextProvider(desc);
    if (choice === null) break;
    const answer = await withProviderLock(choice.providerId, () =>
      callUpstream(choice, { api, stream, body }),
    );
    if (answer !== null && answer !== undefined) {
      current.recordServed(desc, choice);
      renderAnswer(res, api, stream, choice.modelId, answer);
      return;
    }
    current.recordFailure(choice.providerId);
    // Draf ulang rencana sisa dengan kesehatan terbaru.
    current.draftPlan(current.remainingRequests());
  }
  sendWireError(res, api, 503, "all providers refused");
}

// ---------------------------------------------------------------------------
// HTTP server: rute + body
// ---------------------------------------------------------------------------

class BodyTooLargeError extends Error {}

async function readBody(req: IncomingMessage): Promise<string> {
  const chunks: Buffer[] = [];
  let total = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk as string);
    total += buffer.length;
    if (total > MAX_BODY_BYTES) throw new BodyTooLargeError("request body too large");
    chunks.push(buffer);
  }
  return Buffer.concat(chunks).toString("utf8");
}

function handleModels(req: IncomingMessage, res: ServerResponse): void {
  const auth = req.headers.authorization;
  const apiKey = req.headers["x-api-key"];
  const hasBearer =
    typeof auth === "string" && auth.toLowerCase().startsWith("bearer ");
  const hasApiKey = typeof apiKey === "string" && apiKey !== "";
  const ids = servedFreeModelIds(catalog);
  // Keduanya -> Bearer menang. Tak ada keduanya -> daftar kosong.
  if (hasBearer) {
    sendJson(res, 200, {
      object: "list",
      data: ids.map((id) => ({
        id,
        object: "model",
        created: FIXED_CREATED,
        owned_by: "aggregator",
      })),
    });
  } else if (hasApiKey) {
    sendJson(res, 200, {
      data: ids.map((id) => ({
        id,
        created_at: "2026-01-01T00:00:00Z",
        display_name: id,
        type: "model",
      })),
      first_id: ids.length > 0 ? ids[0] : null,
      last_id: ids.length > 0 ? ids[ids.length - 1] : null,
      has_more: false,
    });
  } else {
    sendJson(res, 200, { object: "list", data: [] });
  }
}

async function onRequest(req: IncomingMessage, res: ServerResponse): Promise<void> {
  try {
    const url = new URL(req.url ?? "/", "http://127.0.0.1");
    if (req.method === "GET" && url.pathname === "/v1/models") {
      handleModels(req, res);
      return;
    }
    if (
      req.method === "POST" &&
      (url.pathname === "/v1/chat/completions" || url.pathname === "/v1/messages")
    ) {
      const api: ClientApi =
        url.pathname === "/v1/messages" ? "anthropic" : "openai";
      let text: string;
      try {
        text = await readBody(req);
      } catch (error) {
        if (error instanceof BodyTooLargeError) {
          sendWireError(res, api, 413, "request body too large");
        } else {
          sendWireError(res, api, 400, "request body could not be read");
        }
        return;
      }
      let body: unknown;
      try {
        body = JSON.parse(text);
      } catch {
        sendWireError(res, api, 400, "request body is not valid JSON");
        return;
      }
      if (!isRecord(body)) {
        sendWireError(res, api, 400, "request body must be a JSON object");
        return;
      }
      await handleRequest(api, body, res);
      return;
    }
    sendJson(res, 404, { error: { message: "not found", type: "not_found" } });
  } catch {
    // Error internal: 500. Bila header sudah terkirim, cukup tutup.
    if (!res.headersSent) {
      sendJson(res, 500, { error: { message: "internal error", type: "server_error" } });
    } else {
      res.end();
    }
  }
}

// ---------------------------------------------------------------------------
// Startup: brief -> katalog -> --list / server + register + polling /work
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  const promptFile = process.env.MAKEFUN_PROMPT_FILE ?? "";
  brief = promptFile !== "" ? readBriefFile(promptFile) : EMPTY_BRIEF;
  envUrl = (process.env.MAKEFUN_ENV_URL ?? "").replace(/\/+$/, "");

  // Bangun katalog: probe semua provider paralel, timeout 5 detik per
  // probe; provider yang gagal probe mendapat daftar model kosong.
  // brief diadaptasi ke bentuk catalog (key_env, rating_milli,
  // context_length) supaya katalog yang diregistrasi lengkap.
  catalog = await buildCatalog(catalogBrief(), envUrl);

  if (process.argv.includes("--list")) {
    process.stdout.write(JSON.stringify(catalog) + "\n");
    process.exit(0);
  }

  // Tanpa env, tidak ada trial untuk dilayani: keluar 0 langsung.
  if (envUrl === "") process.exit(0);

  router = new Router(briefPort, routerProviders());

  const server = createServer((req, res) => {
    void onRequest(req, res);
  });
  await new Promise<void>((resolve) => {
    server.listen(0, "0.0.0.0", resolve);
  });
  const address = server.address();
  const port =
    typeof address === "object" && address !== null ? address.port : 0;
  // Guard: bila port hasil listen tak terbaca (address null/0), jangan
  // mendaftarkan port 0 ke env — env tak akan bisa memanggil balik.
  if (port === 0) {
    server.close();
    process.exit(1);
  }

  // Daftarkan diri ke env. Respons tak OK / exception -> exit 1.
  let registered: Response;
  try {
    registered = await fetch(`${envUrl}/register`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ protocol_version: 1, port, catalog }),
      signal: AbortSignal.timeout(REGISTER_TIMEOUT_MS),
    });
  } catch {
    server.close();
    process.exit(1);
  }
  if (!registered.ok) {
    server.close();
    process.exit(1);
  }
  try {
    await registered.text();
  } catch {
    // Body register tidak penting; abaikan bila tak terbaca.
  }

  // Polling GET /work tiap 100 ms sampai done:true, lalu tutup + exit 0.
  // Kegagalan polling (exception / status tak OK / body tak ter-parse)
  // bukan akhir trial: terus polling sampai env menyatakan done.
  for (;;) {
    await sleep(POLL_INTERVAL_MS);
    let done = false;
    try {
      const work = await fetch(`${envUrl}/work`, {
        signal: AbortSignal.timeout(REGISTER_TIMEOUT_MS),
      });
      if (work.ok) {
        const data: unknown = await work.json().catch(() => null);
        if (isRecord(data) && data.done === true) done = true;
      }
    } catch {
      // Tetap polling.
    }
    if (done) break;
  }
  server.close();
  process.exit(0);
}

main().catch(() => {
  process.exit(1);
});
