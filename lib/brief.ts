// lib/brief.ts — Pembaca brief free-model-aggregator-01 (v2).
//
// Tugas file ini: mengubah teks mentah MAKEFUN_PROMPT_FILE menjadi sebuah
// Brief yang terstruktur (daftar provider, tabel rating, manifest workload,
// dan tabel budget). File ini TIDAK tahu apa-apa soal HTTP, provider,
// maupun pemilihan model: ia hanya mem-parsing dokumen.
//
// Tabel budget v2 adalah RANGE [minimum, maksimum] per provider, hardcode
// dari spesifikasi kontes (bukan dari brief, karena brief tidak memuat
// angka budget). Rentang ini dipakai router untuk merencanakan alokasi
// panggilan sepanjang trial; angka pastinya tetap tidak dipublikasikan
// oleh env, jadi range di sini adalah perkiraan aman dari kit.
//
// Deterministik: tidak ada Date.now(), Math.random(), atau wall clock.
// Semua fungsi di sini murni: input sama -> output sama.

import { readFileSync } from "node:fs";

// ---------------------------------------------------------------------------
// Tipe yang diekspor
// ---------------------------------------------------------------------------

/** Wire yang dipakai sebuah provider untuk berbicara dengan modelnya. */
export type Wire = "openai" | "anthropic" | "gemini";

/** Skema auth sebuah provider: nama header yang membawa key-nya. */
export type AuthScheme = "bearer" | "x-api-key" | "x-goog-api-key" | "none";

/** Satu entri direktori provider dari dokumen providers. */
export interface ProviderEntry {
  /** Id provider, mis. "groq". */
  id: string;
  /** Nama tampilan, mis. "Groq". */
  name: string;
  /** Wire vendor yang dipakai provider ini. */
  wire: Wire;
  /** Skema auth provider ini (bukan wire klien). */
  auth: AuthScheme;
  /** Path direktori provider di env, mis. "/p/groq". */
  path: string;
  /** Nama variabel env key-nya, mis. "GROQ_API_KEY" ("" bila auth none). */
  keyEnv: string;
  /** Key provider (key simulasi di trial, key asli di layanan nyata). */
  key: string;
}

/** Satu baris tabel rating (otoritas soal model gratis dan kemampuan). */
export interface RatingRow {
  /** Id model, mis. "z-ai/glm-5.3-flash:free". */
  model: string;
  /** Id provider pemilik model ini. */
  provider: string;
  /** Rating kualitas dalam milli (1475 berarti 1.475). */
  ratingMilli: number;
  /** Panjang konteks dalam token, atau null bila tidak diketahui. */
  contextLength: number | null;
  /** Model menerima pemanggilan tools. */
  tools: boolean;
  /** Model menerima gambar. */
  vision: boolean;
  /** Model menerima mode JSON. */
  json: boolean;
  /** Model gratis (otoritas: tabel rating, bukan katalog provider). */
  free: boolean;
}

/** Syarat yang harus dipenuhi model untuk melayani satu request. */
export interface Requirements {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  /** Panjang konteks minimum dalam token. */
  minContext?: number;
}

/** Satu entri manifest workload. */
export interface WorkloadRequest {
  /** Id request, "r01".."r20" berurutan. */
  id: string;
  /** API klien yang dipakai request ini. */
  api: "openai" | "anthropic";
  /** Benar bila klien meminta jawaban streaming (SSE). */
  stream: boolean;
  /** Bobot skor request ini (1 atau 2). */
  weight: number;
  /** Syarat kemampuan model untuk request ini. */
  requires: Requirements;
}

/** Rating terbaik per request menurut dokumen ratings ("best"). */
export interface BestEntry {
  id: string;
  ratingMilli: number;
}

/** Brief lengkap hasil parse: semua yang router dan main butuhkan. */
export interface Brief {
  /** Direktori provider (10 entri). */
  providers: ProviderEntry[];
  /** Tabel rating model gratis. */
  ratings: RatingRow[];
  /** Manifest workload (20 entri). */
  manifest: WorkloadRequest[];
  /** Rating terbaik per request (batas kolom quality). */
  best: BestEntry[];
  /** Tabel budget: range [min, max] panggilan per provider. */
  budget: Record<string, [number, number]>;
}

// ---------------------------------------------------------------------------
// Tabel budget v2: range [minimum, maksimum] panggilan per provider.
// ---------------------------------------------------------------------------

/** Range budget per provider, hardcode dari spesifikasi v2. */
export const BUDGET_RANGES: Record<string, [number, number]> = {
  groq: [9, 12],
  cerebras: [5, 6],
  gemini: [6, 8],
  zai: [4, 5],
  openrouter: [10, 12],
  sambanova: [4, 5],
  pollinations: [10, 14],
  cloudflare: [7, 8],
  ovh: [5, 6],
  kilol: [12, 16],
};

/** Default aman untuk provider yang tidak dikenal: range paling longgar. */
export const DEFAULT_BUDGET_RANGE: [number, number] = [4, 16];

/**
 * Range budget [min, max] untuk satu provider.
 * Id tak dikenal -> default aman [4, 16]. Selalu mengembalikan pasangan
 * baru, jadi pemanggil bebas mengubahnya tanpa merusak tabel.
 */
export function budgetRange(providerId: string): [number, number] {
  const found = BUDGET_RANGES[providerId];
  if (found === undefined) return [DEFAULT_BUDGET_RANGE[0], DEFAULT_BUDGET_RANGE[1]];
  return [found[0], found[1]];
}

/** Brief kosong: dipakai bila tidak ada brief / file tak terbaca. */
export const EMPTY_BRIEF: Brief = {
  providers: [],
  ratings: [],
  manifest: [],
  best: [],
  budget: { ...BUDGET_RANGES },
};

// ---------------------------------------------------------------------------
// Util parsing kecil (toleran: field hilang diberi default)
// ---------------------------------------------------------------------------

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function asString(value: unknown, fallback = ""): string {
  return typeof value === "string" ? value : fallback;
}

function asBoolean(value: unknown, fallback = false): boolean {
  return typeof value === "boolean" ? value : fallback;
}

function asNumber(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function asWire(value: unknown): Wire {
  if (value === "anthropic" || value === "gemini") return value;
  return "openai";
}

function asAuth(value: unknown): AuthScheme {
  const raw = typeof value === "string" ? value.toLowerCase().replace(/_/g, "-") : "";
  if (raw === "x-api-key") return "x-api-key";
  if (raw === "x-goog-api-key") return "x-goog-api-key";
  if (raw === "none" || raw === "") return "none";
  return "bearer";
}

function asApi(value: unknown): "openai" | "anthropic" {
  return value === "anthropic" ? "anthropic" : "openai";
}

/**
 * Pindai teks mentah dan ekstrak semua objek JSON berimbang.
 * Penyeimbang kurung sadar string dan escape, jadi kurung di dalam
 * string JSON (mis. di dalam teks request) tidak mengacaukan hitungan.
 * Objek yang gagal di-JSON.parse dilewati diam-diam.
 */
function extractJsonObjects(text: string): unknown[] {
  const found: unknown[] = [];
  let i = 0;
  while (i < text.length) {
    const start = text.indexOf("{", i);
    if (start === -1) break;
    let depth = 0;
    let inString = false;
    let escaped = false;
    let j = start;
    let closed = false;
    for (; j < text.length; j++) {
      const ch = text[j];
      if (inString) {
        if (escaped) {
          escaped = false;
        } else if (ch === "\\") {
          escaped = true;
        } else if (ch === '"') {
          inString = false;
        }
        continue;
      }
      if (ch === '"') {
        inString = true;
      } else if (ch === "{") {
        depth += 1;
      } else if (ch === "}") {
        depth -= 1;
        if (depth === 0) {
          closed = true;
          break;
        }
      }
    }
    if (!closed) break;
    try {
      found.push(JSON.parse(text.slice(start, j + 1)));
    } catch {
      // Dokumen rusak: lewati, lanjut memindai setelah objek ini.
    }
    i = j + 1;
  }
  return found;
}

function parseProviders(doc: Record<string, unknown>): ProviderEntry[] {
  if (!Array.isArray(doc.providers)) return [];
  const out: ProviderEntry[] = [];
  // Guard: id provider ganda (dokumen cacat / gabungan dua direktori) —
  // entri pertama menang, sisanya dibuang supaya satu provider tidak
  // muncul dua kali di katalog dan dihitung dua kali oleh router.
  const seen = new Set<string>();
  for (const raw of doc.providers) {
    if (!isRecord(raw)) continue;
    const id = asString(raw.id);
    if (id === "" || seen.has(id)) continue;
    seen.add(id);
    out.push({
      id,
      name: asString(raw.name, id),
      wire: asWire(raw.wire),
      auth: asAuth(raw.auth),
      // Direktori provider; terima juga base_url bila dokumen memakainya.
      path: asString(raw.path, asString(raw.base_url)),
      keyEnv: asString(raw.key_env, asString(raw.keyEnv)),
      key: asString(raw.key),
    });
  }
  return out;
}

function parseRatings(doc: Record<string, unknown>): RatingRow[] {
  if (!Array.isArray(doc.ratings)) return [];
  const out: RatingRow[] = [];
  // Guard sepasang parseProviders: pasangan (provider, model) ganda —
  // baris pertama menang, supaya ratingRow() deterministik dan satu
  // model tidak punya dua rating yang saling bertentangan.
  const seen = new Set<string>();
  for (const raw of doc.ratings) {
    if (!isRecord(raw)) continue;
    const model = asString(raw.model);
    const provider = asString(raw.provider);
    if (model === "" || provider === "") continue;
    const key = `${provider}|${model}`;
    if (seen.has(key)) continue;
    seen.add(key);
    const contextRaw = raw.context_length !== undefined ? raw.context_length : raw.contextLength;
    out.push({
      model,
      provider,
      ratingMilli: asNumber(raw.rating_milli !== undefined ? raw.rating_milli : raw.ratingMilli, 0),
      contextLength:
        typeof contextRaw === "number" && Number.isFinite(contextRaw) ? contextRaw : null,
      tools: asBoolean(raw.tools),
      vision: asBoolean(raw.vision),
      json: asBoolean(raw.json),
      // Dokumen ratings berisi model gratis saja; flag yang hilang berarti
      // gratis, hanya "free": false yang eksplisit yang membuatnya berbayar.
      free: raw.free === undefined ? true : asBoolean(raw.free, true),
    });
  }
  return out;
}

function parseBest(doc: Record<string, unknown>): BestEntry[] {
  if (!Array.isArray(doc.best)) return [];
  const out: BestEntry[] = [];
  for (const raw of doc.best) {
    if (!isRecord(raw)) continue;
    const id = asString(raw.id);
    if (id === "") continue;
    out.push({
      id,
      ratingMilli: asNumber(raw.rating_milli !== undefined ? raw.rating_milli : raw.ratingMilli, 0),
    });
  }
  return out;
}

function parseManifest(doc: Record<string, unknown>): WorkloadRequest[] {
  if (!Array.isArray(doc.manifest)) return [];
  const out: WorkloadRequest[] = [];
  for (const raw of doc.manifest) {
    if (!isRecord(raw)) continue;
    const id = asString(raw.id);
    if (id === "") continue;
    const reqRaw = isRecord(raw.requires) ? raw.requires : {};
    const requires: Requirements = {};
    if (asBoolean(reqRaw.tools)) requires.tools = true;
    if (asBoolean(reqRaw.vision)) requires.vision = true;
    if (asBoolean(reqRaw.json)) requires.json = true;
    const minRaw = reqRaw.min_context !== undefined ? reqRaw.min_context : reqRaw.minContext;
    if (typeof minRaw === "number" && Number.isFinite(minRaw)) requires.minContext = minRaw;
    out.push({
      id,
      api: asApi(raw.api),
      stream: asBoolean(raw.stream),
      weight: asNumber(raw.weight, 1),
      requires,
    });
  }
  return out;
}

// ---------------------------------------------------------------------------
// Fungsi yang diekspor
// ---------------------------------------------------------------------------

/**
 * Parse teks brief mentah menjadi Brief.
 *
 * Mengenali ketiga dokumen dari field schema-nya, masing-masing dalam dua
 * varian nama skema (mis. ".../providers/v1" dan ".../providers/v2"):
 * direktori provider, tabel rating, dan manifest workload. Klasifikasi
 * mengandalkan isi dokumen (array providers / ratings / manifest), jadi
 * varian skema baru atau dokumen tanpa schema tetap dikenali selama
 * field-fieldnya ada. Dokumen yang gagal parse dilewati; field yang
 * hilang diberi default (lihat parser di atas).
 */
export function parseBrief(text: string): Brief {
  const brief: Brief = {
    providers: [],
    ratings: [],
    manifest: [],
    best: [],
    budget: { ...BUDGET_RANGES },
  };
  if (typeof text !== "string" || text === "") return brief;
  for (const doc of extractJsonObjects(text)) {
    if (!isRecord(doc)) continue;
    const schema = asString(doc.schema);
    // Varian skema dikenali lewat kata kuncinya; klasifikasi akhir tetap
    // memakai keberadaan array-nya, supaya dokumen tanpa schema pun lolos.
    const isProvidersDoc = schema.includes("providers") || Array.isArray(doc.providers);
    const isRatingsDoc = schema.includes("ratings") || Array.isArray(doc.ratings);
    const isWorkloadDoc = schema.includes("workload") || Array.isArray(doc.manifest);
    if (isProvidersDoc && brief.providers.length === 0) {
      brief.providers = parseProviders(doc);
    }
    if (isRatingsDoc && brief.ratings.length === 0) {
      brief.ratings = parseRatings(doc);
      brief.best = parseBest(doc);
    }
    if (isWorkloadDoc && brief.manifest.length === 0) {
      brief.manifest = parseManifest(doc);
    }
  }
  return brief;
}

/**
 * Baca file brief dari path dan parse. Bila path tak ada, tak terbaca,
 * atau isinya bukan teks yang bisa di-parse, kembalikan brief kosong
 * (bukan throw) — pemanggil tetap bisa jalan dengan katalog kosong.
 */
export function readBriefFile(path: string): Brief {
  try {
    const text = readFileSync(path, "utf8");
    return parseBrief(text);
  } catch {
    return {
      providers: [],
      ratings: [],
      manifest: [],
      best: [],
      budget: { ...BUDGET_RANGES },
    };
  }
}

/** Baris rating untuk satu (provider, model); undefined bila tak ada. */
export function ratingRow(
  brief: Brief,
  provider: string,
  model: string,
): RatingRow | undefined {
  return brief.ratings.find((row) => row.provider === provider && row.model === model);
}

/**
 * Benar bila baris rating memenuhi semua syarat request.
 * Syarat yang tidak diminta (undefined/false) tidak membatasi.
 * minContext gagal bila context length tak diketahui (null).
 */
export function meetsRequirements(row: RatingRow | undefined, req: Requirements): boolean {
  if (row === undefined) return false;
  if (req.tools === true && row.tools !== true) return false;
  if (req.vision === true && row.vision !== true) return false;
  if (req.json === true && row.json !== true) return false;
  if (req.minContext !== undefined) {
    if (row.contextLength === null || row.contextLength < req.minContext) return false;
  }
  return true;
}

/**
 * Benar bila (provider, model) gratis. Tabel rating adalah otoritas soal
 * gratis: model tanpa baris rating, atau barisnya bertanda free false,
 * dianggap TIDAK gratis dan tidak pernah boleh melayani request.
 */
export function isFreeModel(brief: Brief, provider: string, model: string): boolean {
  const row = ratingRow(brief, provider, model);
  return row !== undefined && row.free === true;
}
