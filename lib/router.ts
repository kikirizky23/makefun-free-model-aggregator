/**
 * lib/router.ts — Router v2 (perencana + eksekutor) untuk
 * free-model-aggregator-01.
 *
 * Ditulis dari spesifikasi v2 (brief tugas) saja, tanpa membaca implementasi
 * bot v1 atau bot lain. Semua pilihan routing berasal dari brief + request +
 * jawaban provider; tidak ada wall clock, tidak ada acak.
 *
 * Determinisme: tidak ada Date.now() / Math.random() / wall clock di file
 * ini. Semua pengurutan punya tie-break total (rating -> plafon budget ->
 * id provider -> id model; kritikalitas -> bobot -> urutan manifest -> id),
 * sehingga draf dan pilihan reproduksibel untuk state yang sama.
 *
 * Peran dalam v2
 * -----------
 * Satu-satunya tuas v2 adalah JUMLAH upstream calls (tie-break skor imbang:
 * lebih sedikit calls menang). Quality 99 adalah plafon (budget openrouter
 * 10-12 < 17 request yang butuh 1475), jadi router TIDAK mengorbankan
 * quality demi calls: anggaran loss keras CAP = 350 (batas quality-99
 * adalah 382) dijaga oleh filter kelayakan loss di setiap pilihan.
 *
 * Aturan prioritas (1)-(4) dari brief BUKAN cabang `if` per request-id di
 * sini — mereka MUNCUL dari mekanisme umum di bawah (urutan kritikalitas
 * draf + fungsi skor + filter kelayakan + estimator risiko):
 *  - r03 mendapat openrouter #1 karena ia teralokasi pertama di antara
 *    request tools-only bobot-2 (kritikalitas sama, urutan manifest dulu),
 *    dan openrouter #1 berisiko 0 dengan loss 0.
 *  - r04/r05 tidak mempertaruhkan openrouter #2/#3 karena skor openrouter
 *    #2 (loss 0 tapi risiko ~0.24 x (loss-aman 10 + Q 50)) kalah dari
 *    cloudflare/zai #1 (loss 10, risiko 0, skor 10).
 *  - Bobot-1 tidak mencoba 1475 selama ada opsi risiko-0 dalam anggaran
 *    loss, karena skor opsi aman selalu di bawah skor pertaruhan 1475.
 *  - Vision: r01 -> gemini #1, r02 mempertaruhkan gemini #2 (skornya
 *    mengalahkan zai #1), r06 ke sambanova #1 / gemini #3 menurut skor;
 *    panggilan pertama gemini/sambanova tidak pernah dipakai request
 *    non-vision sebelum r06 teralokasi karena urutan kritikalitas draf
 *    menempatkan r01/r02/r06 sebelum semua request bobot-1.
 *
 * Interface ke main.ts (urutan pemakaian yang diharapkan)
 * ------------------------------------------------------
 *   const router = new Router(briefPort, providers);
 *   router.beginRequest(id)                 // saat satu request mulai
 *   const draft = router.draftPlan(sisa)    // draf global (awal + tiap gagal)
 *   const cand = router.nextProvider(req)   // provider berikutnya untuk req
 *   ... panggil provider ...
 *   router.recordSuccess(pid) / router.recordFailure(pid)
 *   router.recordServed(req, cand)          // sukses + catat realized loss
 *   router.isDead(pid) / router.cumulativeLoss()
 *   router.futureMinimumLoss(sisa, reqId)
 *
 * Interface ke brief.ts (dependency injection — main.ts yang mengadaptasi)
 * ------------------------------------------------------------------------
 * Constructor menerima `BriefPort` + katalog provider. main.ts mengisi
 * BriefPort dari brief.ts:
 *   budgetRange        <- brief.budgetRange(providerId) -> [min, max]
 *   meetsRequirements  <- brief.meetsRequirements(providerId, modelId, req)
 *   ratingRow          <- brief.ratingRow(providerId, modelId); nilai
 *                         kembalinya dinormalisasi oleh ratingMilliFromRow
 *                         (number, atau { ratingMilli }, atau undefined
 *                         bila model tak ada di tabel rating -> rating 0)
 *   isFreeModel        <- brief.isFreeModel(providerId, modelId)
 * Router tidak pernah mengimpor brief.ts langsung supaya unit ini dapat
 * diuji dengan BriefPort palsu.
 *
 * Aturan kematian (M1-M4)
 * -----------------------
 *   M1: 3 kegagalan beruntun -> mati selamanya.
 *   M2: jumlah panggilan >= maksimum range -> mati.
 *   M3: kegagalan pada ordinal di atas minimum range menaikkan risiko
 *       retry lewat komponen budget estimator (bukan kematian langsung;
 *       retry hanya diambil bila skor memenangkannya).
 *   M4: setelah 2 kegagalan terlihat, bagian terjadwal risiko = 0.
 */

// ---------------------------------------------------------------------------
// Konstanta
// ---------------------------------------------------------------------------

/** CAP: batas keras anggaran loss kumulatif (skor quality-99). */
export const LOSS_CAP = 350;
/** Q0: bobot dasar penalti risiko dalam fungsi skor pilihan. */
export const Q0 = 50;
/** Faktor minimum Q saat loss mendekati CAP (Q tidak pernah di bawah 15%). */
export const MIN_Q_FACTOR = 0.15;
/** Q saat loss kumulatif sudah mencapai / melampaui CAP. */
export const FALLBACK_Q = 5;
/** M1: jumlah kegagalan beruntun yang mematikan provider selamanya. */
export const MAX_CONSECUTIVE_FAILURES = 3;
/** Batas atas estimator risiko (total tidak pernah melebihi ini). */
export const RISK_CAP = 0.95;
/** Jatah kegagalan terjadwal per provider (fakta generator). */
export const SCHEDULED_FAILURE_ALLOWANCE = 1.5;

// ---------------------------------------------------------------------------
// Tipe
// ---------------------------------------------------------------------------

/** Wire protokol provider, mis. "openai" / "anthropic" (string bebas). */
export type Wire = string;

export interface ModelEntry {
  id: string;
  contextLength: number;
  tools: boolean;
  vision: boolean;
  json: boolean;
}

export interface ProviderEntry {
  id: string;
  wire: Wire;
  models: ModelEntry[];
}

export interface RequestRequirements {
  tools: boolean;
  vision: boolean;
  json: boolean;
  minContext: number;
}

/** Satu request dari manifest. `clientWire` = wire yang dipakai klien. */
export interface RouterRequest extends RequestRequirements {
  id: string;
  weight: number;
  /** Posisi request di manifest (0-based); dipakai sebagai tie-break. */
  manifestIndex: number;
  /** Wire klien, mis. "openai" / "anthropic" / "generic". */
  clientWire: string;
}

export interface Candidate {
  providerId: string;
  /** Entri provider dari katalog (referensi yang sama dengan input). */
  provider: ProviderEntry;
  wire: Wire;
  modelId: string;
  /** Rating dalam milli-poin; 0 bila model tak ada di tabel rating. */
  ratingMilli: number;
  contextLength: number;
  tools: boolean;
  vision: boolean;
  json: boolean;
}

/** Kesehatan per provider (state antar request). */
export interface ProviderHealth {
  providerId: string;
  callsMade: number;
  failuresSeen: number;
  consecutiveFailures: number;
  dead: boolean;
}

/** Satu baris draf global. Provider null = tak ada provider hidup legal. */
export interface DraftEntry {
  requestId: string;
  manifestIndex: number;
  providerId: string | null;
  modelId: string | null;
  ratingMilli: number | null;
  risk: number | null;
  score: number | null;
  /** Loss yang diharapkan bila baris ini dieksekusi sesuai draf. */
  expectedLoss: number;
}

/** Hasil pemilihan satu opsi (dipakai draf + eksekutor). */
export interface PickedOption {
  candidate: Candidate;
  risk: number;
  score: number;
  loss: number;
}

/**
 * Port ke brief.ts. main.ts mengadaptasi fungsi-fungsi brief.ts ke bentuk
 * ini; router tidak mengimpor brief.ts langsung.
 */
export interface BriefPort {
  /** Range budget panggilan provider: [min, max] (inklusif). */
  budgetRange(providerId: string): [number, number];
  /** Apakah model memenuhi semua requirement request. */
  meetsRequirements(
    providerId: string,
    modelId: string,
    req: RequestRequirements,
  ): boolean;
  /**
   * Baris rating model. Boleh mengembalikan angka ratingMilli langsung,
   * objek { ratingMilli }, atau undefined/null bila model tak ada di tabel
   * rating (model seperti itu tetap legal dengan rating 0).
   */
  ratingRow(
    providerId: string,
    modelId: string,
  ): number | { ratingMilli: number } | undefined | null;
  /** Apakah model gratis. Model berbayar tidak pernah legal. */
  isFreeModel(providerId: string, modelId: string): boolean;
}

// ---------------------------------------------------------------------------
// Util kecil (deterministik)
// ---------------------------------------------------------------------------

function compareStrings(a: string, b: string): number {
  if (a === b) return 0;
  return a < b ? -1 : 1;
}

function validateProvider(p: ProviderEntry): void {
  if (!p || typeof p.id !== "string" || p.id.length === 0) {
    throw new TypeError("Router: provider.id harus string non-kosong");
  }
  if (typeof p.wire !== "string" || p.wire.length === 0) {
    throw new TypeError(`Router: provider ${p.id} wire harus string non-kosong`);
  }
  if (!Array.isArray(p.models)) {
    throw new TypeError(`Router: provider ${p.id} models harus array`);
  }
  const seen = new Set<string>();
  for (const m of p.models) {
    if (!m || typeof m.id !== "string" || m.id.length === 0) {
      throw new TypeError(`Router: provider ${p.id} punya model tanpa id`);
    }
    if (seen.has(m.id)) {
      throw new TypeError(`Router: provider ${p.id} model duplikat: ${m.id}`);
    }
    seen.add(m.id);
    if (!Number.isFinite(m.contextLength) || m.contextLength < 0) {
      throw new TypeError(`Router: model ${p.id}/${m.id} contextLength tidak valid`);
    }
    if (typeof m.tools !== "boolean" || typeof m.vision !== "boolean" || typeof m.json !== "boolean") {
      throw new TypeError(`Router: model ${p.id}/${m.id} capability harus boolean`);
    }
  }
}

function validateRequest(req: RouterRequest): void {
  if (!req || typeof req.id !== "string" || req.id.length === 0) {
    throw new TypeError("Router: request.id harus string non-kosong");
  }
  if (!Number.isFinite(req.weight) || req.weight <= 0) {
    throw new TypeError(`Router: request ${req.id} weight harus > 0`);
  }
  if (!Number.isInteger(req.manifestIndex) || req.manifestIndex < 0) {
    throw new TypeError(`Router: request ${req.id} manifestIndex harus integer >= 0`);
  }
  if (typeof req.tools !== "boolean" || typeof req.vision !== "boolean" || typeof req.json !== "boolean") {
    throw new TypeError(`Router: request ${req.id} requirement harus boolean`);
  }
  if (!Number.isFinite(req.minContext) || req.minContext < 0) {
    throw new TypeError(`Router: request ${req.id} minContext tidak valid`);
  }
  if (typeof req.clientWire !== "string") {
    throw new TypeError(`Router: request ${req.id} clientWire harus string`);
  }
}

/** Normalisasi nilai kembalian ratingRow -> ratingMilli (0 bila tak ada). */
export function ratingMilliFromRow(row: unknown): number {
  if (typeof row === "number" && Number.isFinite(row)) return row;
  if (typeof row === "object" && row !== null) {
    const v = (row as { ratingMilli?: unknown }).ratingMilli;
    if (typeof v === "number" && Number.isFinite(v)) return v;
  }
  return 0;
}

// ---------------------------------------------------------------------------
// Estimator risiko — unit murni yang dapat diuji terpisah
// ---------------------------------------------------------------------------

export interface RiskInput {
  /** Panggilan yang sudah dibuat ke provider ini. */
  callsMade: number;
  /** Kegagalan yang sudah terlihat dari provider ini. */
  failuresSeen: number;
  /** Minimum range budget provider. */
  budgetMin: number;
  /** Maksimum range budget provider. */
  budgetMax: number;
}

/**
 * Peluang panggilan BERIKUTNYA ke provider P gagal, pada ordinal
 * k = callsMade + 1.
 *
 * Fakta generator yang dikodekan di sini:
 *  - panggilan ke-1 tiap provider SELALU sukses (k=1 -> risiko 0);
 *  - jadwal gagal mulai ordinal 2, outage mulai ordinal 2, budget >= 4;
 *  - setelah satu kegagalan, ordinal berikutnya lebih aman (maks 2 gagal
 *    terjadwal); 3 kegagalan beruntun = provider mati selamanya (M1).
 *
 *   terjadwal(k=2) = 0.1 + 0.9 * 1.5 / (max - 2)
 *   terjadwal(k>=3) = sisa-jatah-gagal / max(1, max - k),
 *     sisa-jatah-gagal = max(0, 1.5 - failuresSeen); 0 bila failuresSeen >= 2 (M4)
 *   budget = 0 bila k <= min; (k - min) / (max - min + 1) bila min < k <= max
 *   outage = 0.1 / max(1, max - k + 1) untuk k >= 2 (0 bila k = 1)
 *   total = 1 - (1 - terjadwal)(1 - budget)(1 - outage), dibatasi <= 0.95
 */
export function estimateRisk(input: RiskInput): number {
  const { callsMade, failuresSeen, budgetMin, budgetMax } = input;
  // Guard input: ordinal dan range harus masuk akal; input tak valid adalah
  // kesalahan pemanggil, bukan risiko — lempar daripada menebak.
  if (
    !Number.isInteger(callsMade) ||
    callsMade < 0 ||
    !Number.isInteger(failuresSeen) ||
    failuresSeen < 0 ||
    !Number.isFinite(budgetMin) ||
    !Number.isFinite(budgetMax) ||
    budgetMax < 1 ||
    budgetMin < 0 ||
    budgetMin > budgetMax
  ) {
    throw new RangeError("estimateRisk: input tidak valid");
  }
  const k = callsMade + 1;
  if (k <= 1) return 0;
  if (k > budgetMax) return 1;

  // Bagian terjadwal (M4: setelah 2 kegagalan terlihat, bagian ini 0).
  let scheduled = 0;
  if (failuresSeen < 2) {
    if (k === 2) {
      scheduled =
        0.1 +
        (0.9 * SCHEDULED_FAILURE_ALLOWANCE) / Math.max(1, budgetMax - 2);
    } else {
      const remainingAllowance = Math.max(
        0,
        SCHEDULED_FAILURE_ALLOWANCE - failuresSeen,
      );
      // (max - 1) - k + 1 = max - k.
      scheduled = remainingAllowance / Math.max(1, budgetMax - k);
    }
  }

  // Guard komponen: rumus k=2 dapat memberi terjadwal > 1 bila max kecil
  // (mis. max=3 -> 1.45); dijepit ke [0, 1]. Netral-perilaku: total tetap
  // dibatasi RISK_CAP di bawah, dan terjadwal=1 sudah berarti gagal pasti
  // dari komponen ini.
  scheduled = Math.min(1, Math.max(0, scheduled));

  // Bagian budget (M3: kegagalan di atas minimum menaikkan risiko retry
  // lewat komponen ini — bukan kematian langsung).
  let budget = 0;
  if (k > budgetMin) {
    budget = (k - budgetMin) / (budgetMax - budgetMin + 1);
  }

  // Bagian outage (mulai ordinal 2).
  const outage = 0.1 / Math.max(1, budgetMax - k + 1);

  const total = 1 - (1 - scheduled) * (1 - budget) * (1 - outage);
  return Math.min(RISK_CAP, Math.max(0, total));
}

/**
 * Penalti risiko Q dari loss kumulatif:
 * Q = Q0 * max(0.15, (CAP - loss) / CAP) bila loss < CAP, else Q = 5.
 * Q menyusut saat anggaran loss menipis, tapi tidak pernah di bawah 15%
 * dari Q0 selama masih di bawah CAP.
 */
export function qualityPenalty(cumulativeLoss: number): number {
  if (cumulativeLoss >= LOSS_CAP) return FALLBACK_Q;
  return Q0 * Math.max(MIN_Q_FACTOR, (LOSS_CAP - cumulativeLoss) / LOSS_CAP);
}

export interface ChoiceScoreInput {
  weight: number;
  /** Rating legal terbaik R* untuk request ini. */
  bestRating: number;
  /** Rating opsi yang dinilai. */
  rating: number;
  /** Risiko opsi (dari estimateRisk). */
  risk: number;
  /** Loss-aman: loss bila jatuh ke opsi risiko-0 terbaik. */
  safeLoss: number;
  cumulativeLoss: number;
}

/**
 * Fungsi skor pilihan:
 *   loss(o)    = w * (R* - R_o)
 *   skor(o)    = (1 - r) * loss(o) + r * loss-aman + Q * r
 * Pilih skor terendah; seri -> rating lebih tinggi -> id provider
 * alfabetis -> id model alfabetis (lihat pickLowestScore).
 */
export function scoreChoice(input: ChoiceScoreInput): number {
  const loss = input.weight * (input.bestRating - input.rating);
  // Guard: risiko di luar [0, 1] (mis. dari estimator lain) dijepit agar
  // skor tetap bermakna — netral untuk risiko dari estimateRisk yang
  // memang sudah berada di [0, 0.95].
  const risk = Math.min(1, Math.max(0, input.risk));
  const q = qualityPenalty(input.cumulativeLoss);
  return (1 - risk) * loss + risk * input.safeLoss + q * risk;
}

/** Pilih opsi berskor terendah dengan tie-break skor yang ditentukan. */
function pickLowestScore(scored: PickedOption[]): PickedOption {
  const sorted = [...scored].sort((a, b) => {
    if (a.score !== b.score) return a.score - b.score;
    if (a.candidate.ratingMilli !== b.candidate.ratingMilli) {
      return b.candidate.ratingMilli - a.candidate.ratingMilli;
    }
    const byProvider = compareStrings(a.candidate.providerId, b.candidate.providerId);
    if (byProvider !== 0) return byProvider;
    return compareStrings(a.candidate.modelId, b.candidate.modelId);
  });
  return sorted[0];
}

// ---------------------------------------------------------------------------
// Router — perencana + eksekutor
// ---------------------------------------------------------------------------

export class Router {
  private readonly deps: BriefPort;
  private readonly providers: ProviderEntry[];
  private readonly providerById: Map<string, ProviderEntry>;
  private readonly budgets: Map<string, [number, number]>;
  private readonly health: Map<string, ProviderHealth>;
  /** Loss kumulatif terealisasi (hanya bertambah via recordServed). */
  private realizedLoss = 0;
  /** Nomor kedatangan global (bertambah tiap beginRequest). */
  private arrivals = 0;
  /** Daftar request tersisa (disetel tiap draftPlan, susut tiap recordServed). */
  private remaining: RouterRequest[] = [];
  /** Percobaan per provider untuk request yang sedang berjalan (transien). */
  private readonly attempts: Map<string, Map<string, number>> = new Map();

  constructor(deps: BriefPort, providers: ProviderEntry[]) {
    if (
      !deps ||
      typeof deps.budgetRange !== "function" ||
      typeof deps.meetsRequirements !== "function" ||
      typeof deps.ratingRow !== "function" ||
      typeof deps.isFreeModel !== "function"
    ) {
      throw new TypeError("Router: BriefPort harus menyediakan 4 fungsi brief.ts");
    }
    if (!Array.isArray(providers) || providers.length === 0) {
      throw new TypeError("Router: providers harus array non-kosong");
    }
    this.deps = deps;
    this.providers = [...providers];
    this.providerById = new Map();
    this.budgets = new Map();
    this.health = new Map();
    for (const p of providers) {
      validateProvider(p);
      if (this.providerById.has(p.id)) {
        throw new TypeError(`Router: id provider duplikat: ${p.id}`);
      }
      this.providerById.set(p.id, p);
      const range = deps.budgetRange(p.id);
      if (
        !Array.isArray(range) ||
        range.length !== 2 ||
        !Number.isFinite(range[0]) ||
        !Number.isFinite(range[1]) ||
        range[0] < 0 ||
        range[1] < range[0]
      ) {
        throw new RangeError(`Router: budgetRange tidak valid untuk ${p.id}`);
      }
      this.budgets.set(p.id, [range[0], range[1]]);
      this.health.set(p.id, {
        providerId: p.id,
        callsMade: 0,
        failuresSeen: 0,
        consecutiveFailures: 0,
        dead: false,
      });
    }
  }

  // -- akses state ----------------------------------------------------------

  providerIds(): string[] {
    return [...this.providerById.keys()];
  }

  /** Salinan kesehatan satu provider (bukan referensi internal). */
  healthOf(providerId: string): ProviderHealth {
    return { ...this.requireHealth(providerId) };
  }

  /** (f) Apakah provider mati (flag M1/M2, atau callsMade >= maksimum). */
  isDead(providerId: string): boolean {
    const h = this.requireHealth(providerId);
    return h.dead || h.callsMade >= this.budgetMaxOf(providerId);
  }

  /** Risiko panggilan berikutnya ke provider ini (state live). */
  riskOf(providerId: string): number {
    const h = this.requireHealth(providerId);
    const [min, max] = this.budgetRangeOf(providerId);
    return estimateRisk({
      callsMade: h.callsMade,
      failuresSeen: h.failuresSeen,
      budgetMin: min,
      budgetMax: max,
    });
  }

  /** (g) Loss kumulatif terealisasi sejauh ini. */
  cumulativeLoss(): number {
    return this.realizedLoss;
  }

  /** Jumlah request yang sudah dimulai (nomor kedatangan global). */
  arrivalCount(): number {
    return this.arrivals;
  }

  /** Salinan daftar request tersisa yang dipegang router. */
  remainingRequests(): RouterRequest[] {
    return [...this.remaining];
  }

  // -- (a) kandidat legal ---------------------------------------------------

  /**
   * Semua kandidat legal untuk satu request, terurut menurut urutan dasar:
   * rating menurun -> plafon budget maksimum provider menurun -> id
   * provider menaik (alfabetis) -> id model menaik.
   *
   * Legal = model gratis (isFreeModel) + memenuhi semua requirement
   * (meetsRequirements) + khusus klien OpenAI yang meminta JSON mode,
   * provider BUKAN wire Anthropic. Model katalog yang tak ada di tabel
   * rating tetap legal dengan rating 0. Model berbayar tidak pernah legal.
   */
  legalCandidates(req: RouterRequest): Candidate[] {
    validateRequest(req);
    const out: Candidate[] = [];
    for (const provider of this.providers) {
      for (const model of provider.models) {
        if (!this.deps.isFreeModel(provider.id, model.id)) continue;
        if (
          !this.deps.meetsRequirements(provider.id, model.id, {
            tools: req.tools,
            vision: req.vision,
            json: req.json,
            minContext: req.minContext,
          })
        ) {
          continue;
        }
        if (
          req.json &&
          req.clientWire === "openai" &&
          provider.wire === "anthropic"
        ) {
          continue;
        }
        out.push({
          providerId: provider.id,
          provider,
          wire: provider.wire,
          modelId: model.id,
          ratingMilli: ratingMilliFromRow(
            this.deps.ratingRow(provider.id, model.id),
          ),
          contextLength: model.contextLength,
          tools: model.tools,
          vision: model.vision,
          json: model.json,
        });
      }
    }
    out.sort((a, b) => this.compareCandidates(a, b));
    return out;
  }

  /** Satu model terbaik per provider untuk satu request (urutan dasar). */
  bestPerProvider(req: RouterRequest): Map<string, Candidate> {
    const out = new Map<string, Candidate>();
    for (const cand of this.legalCandidates(req)) {
      if (!out.has(cand.providerId)) out.set(cand.providerId, cand);
    }
    return out;
  }

  /** Rating legal terbaik R* untuk satu request (0 bila tak ada kandidat). */
  bestRating(req: RouterRequest): number {
    const cands = this.legalCandidates(req);
    return cands.length === 0 ? 0 : cands[0].ratingMilli;
  }

  /** Urutan dasar kandidat (dipakai legalCandidates + fallback rating murni). */
  private compareCandidates(a: Candidate, b: Candidate): number {
    if (a.ratingMilli !== b.ratingMilli) return b.ratingMilli - a.ratingMilli;
    const maxA = this.budgetMaxOf(a.providerId);
    const maxB = this.budgetMaxOf(b.providerId);
    if (maxA !== maxB) return maxB - maxA;
    const byProvider = compareStrings(a.providerId, b.providerId);
    if (byProvider !== 0) return byProvider;
    return compareStrings(a.modelId, b.modelId);
  }

  // -- (b) kritikalitas ------------------------------------------------------

  /**
   * Kritikalitas satu request = bobot * (rating terbaik - rating
   * terbaik-kedua dari provider BERBEDA). Bila hanya satu provider legal,
   * rating terbaik-kedua dianggap 0 (request tanpa fallback = paling
   * kritis). Dipakai untuk mengurutkan draf global.
   */
  criticality(req: RouterRequest): number {
    validateRequest(req);
    const ratings: number[] = [];
    for (const cand of this.bestPerProvider(req).values()) {
      ratings.push(cand.ratingMilli);
    }
    if (ratings.length === 0) return 0;
    ratings.sort((a, b) => b - a);
    const best = ratings[0];
    const second = ratings.length > 1 ? ratings[1] : 0;
    return req.weight * (best - second);
  }

  // -- loss -----------------------------------------------------------------

  /** Loss terealisasi bila request dilayani kandidat ini: w * (R* - R). */
  lossOf(req: RouterRequest, candidate: Candidate): number {
    return Math.max(
      0,
      req.weight * (this.bestRating(req) - candidate.ratingMilli),
    );
  }

  /**
   * Loss-aman untuk satu request: w * (R* - rating-terbaik-di-antara-opsi-
   * berisiko-0), atau w * R* bila tak ada opsi berisiko-0. `risks` memetakan
   * providerId -> risiko opsi itu (live atau simulasi draf).
   */
  private safeLossForOptions(
    req: RouterRequest,
    options: Candidate[],
    risks: Map<string, number>,
  ): number {
    const best = this.bestRating(req);
    let bestSafe: number | null = null;
    for (const cand of options) {
      const r = risks.get(cand.providerId);
      if (r !== undefined && r <= 0) {
        if (bestSafe === null || cand.ratingMilli > bestSafe) {
          bestSafe = cand.ratingMilli;
        }
      }
    }
    return bestSafe === null
      ? req.weight * best
      : req.weight * (best - bestSafe);
  }

  /** Loss-aman dengan risiko live (dipakai eksekutor / pengujian). */
  safeLoss(req: RouterRequest, options: Candidate[]): number {
    const risks = new Map<string, number>();
    for (const cand of options) risks.set(cand.providerId, this.riskOf(cand.providerId));
    return this.safeLossForOptions(req, options, risks);
  }

  /**
   * (g) Loss-minimum-masa-depan: jumlah atas semua request tersisa dari
   * bobot * (rating-terbaik - rating-terbaik-dari-provider-yang-masih-hidup-
   * dan-legal), atau bobot * rating-terbaik bila tak ada provider hidup.
   * `excludeId` mengeluarkan satu request (yang sedang dipilih) dari jumlah.
   * `healthView` memungkinkan evaluasi atas kesehatan simulasi draf.
   */
  futureMinimumLoss(
    remaining: RouterRequest[],
    excludeId?: string,
    healthView?: Map<string, ProviderHealth>,
  ): number {
    const view = healthView ?? this.health;
    let total = 0;
    for (const r of remaining) {
      if (excludeId !== undefined && r.id === excludeId) continue;
      const best = this.bestRating(r);
      let bestLive: number | null = null;
      for (const cand of this.bestPerProvider(r).values()) {
        if (!this.isAliveInView(cand.providerId, view)) continue;
        if (bestLive === null || cand.ratingMilli > bestLive) {
          bestLive = cand.ratingMilli;
        }
      }
      total +=
        bestLive === null ? r.weight * best : r.weight * (best - bestLive);
    }
    return total;
  }

  // -- (c) draf global -------------------------------------------------------

  /**
   * Menyusun draf untuk daftar request tersisa. Dipanggil di awal dan
   * di-draft ulang tiap kegagalan.
   *
   *  1. Urutkan request tersisa menurut kritikalitas menurun; seri ->
   *     bobot menurun -> urutan manifest -> id. Inilah yang melindungi
   *     request kritis: r01/r02 dan r06 direncanakan SEBELUM request
   *     bobot-1, sehingga panggilan pertama gemini/sambanova tidak
   *     dihabiskan request non-vision.
   *  2. Alokasikan satu demi satu dengan fungsi skor (pickBest), sambil
   *     memajukan simulasi ordinal tiap provider (callsMade simulasi +1
   *     per alokasi; failuresSeen simulasi tetap 0 karena draf mengasumsi
   *     sukses) agar ordinal yang direncanakan konsisten.
   *  3. Loss kumulatif simulasi ikut maju (realized + loss draf sejauh ini)
   *     sehingga filter kelayakan loss melihat anggaran yang tersisa.
   *
   * Mengembalikan entri dalam URUTAN ALOKASI (kritikalitas); main.ts dapat
   * mengurutkan ulang menurut manifestIndex bila perlu. Menyimpan salinan
   * `remaining` untuk dipakai nextProvider tanpa argumen.
   */
  draftPlan(remaining: RouterRequest[]): DraftEntry[] {
    const ids = new Set<string>();
    for (const r of remaining) {
      validateRequest(r);
      if (ids.has(r.id)) {
        throw new TypeError(`Router: request duplikat di draf: ${r.id}`);
      }
      ids.add(r.id);
    }
    this.remaining = [...remaining];

    const withCrit = remaining.map((req) => ({
      req,
      crit: this.criticality(req),
    }));
    withCrit.sort((a, b) => {
      if (a.crit !== b.crit) return b.crit - a.crit;
      if (a.req.weight !== b.req.weight) return b.req.weight - a.req.weight;
      if (a.req.manifestIndex !== b.req.manifestIndex) {
        return a.req.manifestIndex - b.req.manifestIndex;
      }
      return compareStrings(a.req.id, b.req.id);
    });

    // Kesehatan simulasi: salinan state live; draf tidak mengubah state live.
    const sim = new Map<string, ProviderHealth>();
    for (const [id, h] of this.health) sim.set(id, { ...h });

    let simLoss = this.realizedLoss;
    const entries: DraftEntry[] = [];
    for (const { req } of withCrit) {
      const rest = remaining.filter((r) => r.id !== req.id);
      const pick = this.pickBest(req, sim, simLoss, rest);
      if (!pick) {
        entries.push({
          requestId: req.id,
          manifestIndex: req.manifestIndex,
          providerId: null,
          modelId: null,
          ratingMilli: null,
          risk: null,
          score: null,
          expectedLoss: req.weight * this.bestRating(req),
        });
        continue;
      }
      entries.push({
        requestId: req.id,
        manifestIndex: req.manifestIndex,
        providerId: pick.candidate.providerId,
        modelId: pick.candidate.modelId,
        ratingMilli: pick.candidate.ratingMilli,
        risk: pick.risk,
        score: pick.score,
        expectedLoss: pick.loss,
      });
      simLoss += pick.loss;
      const hv = sim.get(pick.candidate.providerId);
      if (hv) {
        hv.callsMade += 1;
        if (hv.callsMade >= this.budgetMaxOf(pick.candidate.providerId)) {
          hv.dead = true; // M2 dalam simulasi
        }
      }
    }
    return entries;
  }

  // -- (d) pilih provider berikutnya ----------------------------------------

  /**
   * Memilih provider berikutnya untuk satu request yang sedang berjalan
   * (eksekutor). Retry provider yang sama diizinkan: pilihan dinilai ulang
   * dengan risiko terbaru, dan loop eksekutor tidak pernah meninggalkan
   * request tanpa jawaban selama masih ada satu provider legal yang tidak
   * terbukti mati (M1/M2) — bila null, berarti memang tak ada opsi hidup.
   * Mencatat percobaan transien (noteAttempt) untuk provider yang dipilih.
   */
  nextProvider(req: RouterRequest, remaining?: RouterRequest[]): Candidate | null {
    validateRequest(req);
    const rest = (remaining ?? this.remaining).filter((r) => r.id !== req.id);
    const pick = this.pickBest(req, this.health, this.realizedLoss, rest);
    if (!pick) return null;
    this.noteAttempt(req.id, pick.candidate.providerId);
    return pick.candidate;
  }

  /**
   * Inti pemilihan (dipakai draf + eksekutor). `healthView` = kesehatan live
   * atau simulasi; `cumLoss` = loss kumulatif (real atau simulasi).
   *
   *  1. Opsi = satu model terbaik per provider yang masih hidup di view.
   *  2. Tiap opsi dinilai dengan fungsi skor (Q dari cumLoss).
   *  3. Filter kelayakan loss: opsi DITERIMA hanya bila
   *     cumLoss + loss(o) + loss-minimum-masa-depan <= CAP.
   *  4. Bila ada opsi layak -> skor terendah di antaranya.
   *  5. Bila semua opsi melanggar DAN bahkan pilihan loss-nol pun sudah
   *     melampaui CAP (anggaran tipis) -> urutan rating murni (aturan
   *     prioritas 3: minimalkan loss lanjutan).
   *  6. Bila semua opsi melanggar tapi anggaran belum setipis itu -> opsi
   *     berskor terendah.
   */
  private pickBest(
    req: RouterRequest,
    healthView: Map<string, ProviderHealth>,
    cumLoss: number,
    remainingForFuture: RouterRequest[],
  ): PickedOption | null {
    const options: Candidate[] = [];
    for (const cand of this.bestPerProvider(req).values()) {
      if (this.isAliveInView(cand.providerId, healthView)) options.push(cand);
    }
    if (options.length === 0) return null;

    const best = this.bestRating(req);
    const risks = new Map<string, number>();
    for (const cand of options) {
      const hv = healthView.get(cand.providerId);
      const [min, max] = this.budgetRangeOf(cand.providerId);
      risks.set(
        cand.providerId,
        estimateRisk({
          callsMade: hv ? hv.callsMade : 0,
          failuresSeen: hv ? hv.failuresSeen : 0,
          budgetMin: min,
          budgetMax: max,
        }),
      );
    }
    const safe = this.safeLossForOptions(req, options, risks);
    const futureMin = this.futureMinimumLoss(
      remainingForFuture,
      req.id,
      healthView,
    );

    const scored: PickedOption[] = options.map((cand) => {
      const risk = risks.get(cand.providerId) ?? 0;
      const loss = req.weight * (best - cand.ratingMilli);
      return {
        candidate: cand,
        risk,
        score: scoreChoice({
          weight: req.weight,
          bestRating: best,
          rating: cand.ratingMilli,
          risk,
          safeLoss: safe,
          cumulativeLoss: cumLoss,
        }),
        loss,
      };
    });

    const feasible = scored.filter(
      (s) => cumLoss + s.loss + futureMin <= LOSS_CAP,
    );
    if (feasible.length > 0) return pickLowestScore(feasible);
    if (cumLoss + futureMin > LOSS_CAP) {
      // Anggaran sudah tipis: urutan rating murni.
      const byRating = [...scored].sort((a, b) =>
        this.compareCandidates(a.candidate, b.candidate),
      );
      return byRating[0];
    }
    return pickLowestScore(scored);
  }

  // -- (e) catat sukses / gagal ----------------------------------------------

  /**
   * Mencatat sukses: callsMade +1, kegagalan beruntun direset, M2 bila
   * callsMade mencapai maksimum range.
   */
  recordSuccess(providerId: string): void {
    const h = this.requireHealth(providerId);
    h.callsMade += 1;
    h.consecutiveFailures = 0;
    if (h.callsMade >= this.budgetMaxOf(providerId)) h.dead = true; // M2
  }

  /**
   * Mencatat gagal: callsMade +1, failuresSeen +1, kegagalan beruntun +1;
   * M1 bila 3 beruntun, M2 bila callsMade mencapai maksimum range.
   * (M3/M4 bukan kematian: M3 bekerja lewat komponen budget estimator,
   * M4 lewat bagian terjadwal estimator yang menjadi 0 setelah 2 gagal.)
   */
  recordFailure(providerId: string): void {
    const h = this.requireHealth(providerId);
    h.callsMade += 1;
    h.failuresSeen += 1;
    h.consecutiveFailures += 1;
    if (h.consecutiveFailures >= MAX_CONSECUTIVE_FAILURES) h.dead = true; // M1
    if (h.callsMade >= this.budgetMaxOf(providerId)) h.dead = true; // M2
  }

  /**
   * Mencatat satu request selesai dilayani kandidat ini: sukses provider +
   * realized loss (w * (R* - R)) + keluarkan request dari daftar tersisa +
   * hapus percobaan transiennya.
   */
  recordServed(req: RouterRequest, candidate: Candidate): void {
    this.recordSuccess(candidate.providerId);
    this.realizedLoss += this.lossOf(req, candidate);
    this.remaining = this.remaining.filter((r) => r.id !== req.id);
    this.attempts.delete(req.id);
  }

  // -- state transien per request --------------------------------------------

  /** Menandai satu request mulai berjalan (nomor kedatangan global +1). */
  beginRequest(requestId: string): void {
    if (typeof requestId !== "string" || requestId.length === 0) {
      throw new TypeError("Router: requestId harus string non-kosong");
    }
    this.arrivals += 1;
    if (!this.attempts.has(requestId)) this.attempts.set(requestId, new Map());
  }

  /** Mencatat satu percobaan ke provider untuk request yang sedang berjalan. */
  noteAttempt(requestId: string, providerId: string): void {
    this.requireHealth(providerId);
    let per = this.attempts.get(requestId);
    if (!per) {
      per = new Map<string, number>();
      this.attempts.set(requestId, per);
    }
    per.set(providerId, (per.get(providerId) ?? 0) + 1);
  }

  /** Jumlah percobaan ke provider untuk request yang sedang berjalan. */
  attemptsFor(requestId: string, providerId: string): number {
    return this.attempts.get(requestId)?.get(providerId) ?? 0;
  }

  // -- privat ----------------------------------------------------------------

  private requireHealth(providerId: string): ProviderHealth {
    const h = this.health.get(providerId);
    if (!h) {
      throw new RangeError(`Router: provider tidak dikenal: ${providerId}`);
    }
    return h;
  }

  private budgetRangeOf(providerId: string): [number, number] {
    const r = this.budgets.get(providerId);
    if (!r) {
      throw new RangeError(`Router: provider tidak dikenal: ${providerId}`);
    }
    return r;
  }

  private budgetMaxOf(providerId: string): number {
    return this.budgetRangeOf(providerId)[1];
  }

  private isAliveInView(
    providerId: string,
    view: Map<string, ProviderHealth>,
  ): boolean {
    const hv = view.get(providerId);
    if (!hv) return false;
    return !hv.dead && hv.callsMade < this.budgetMaxOf(providerId);
  }
}
