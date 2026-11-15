/**
 * lib/catalog.ts — provider catalog for the free-model aggregator.
 *
 * This module owns everything about *which* models exist and which of them
 * are free:
 *
 *   - probing each provider's model list (`GET <base>/v1/models`, or
 *     `/v1beta/models` on the Gemini wire),
 *   - merging those live listings with the brief's sealed rating table,
 *     where the table always wins for a model's `free` flag and its
 *     `context_length`,
 *   - and the catalog document itself: the exact JSON object the server
 *     registers with the env and prints with `--list`, byte for byte.
 *
 * The catalog's `base_url` is the provider's own directory path from the
 * brief (e.g. `/p/groq`), verbatim — never joined to the env origin. The
 * env origin is joined only to dial a provider.
 *
 * Deterministic: no wall clock, no randomness. Every choice here is a pure
 * function of the brief and the providers' answers.
 */

export type Wire = "openai" | "anthropic" | "gemini";

export type AuthScheme = "bearer" | "x-api-key" | "x-goog-api-key" | "none";

/** One model in the catalog document. */
export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

/** One provider in the catalog document. */
export interface CatalogProvider {
  id: string;
  /** Directory path from the brief, verbatim (e.g. "/p/groq"). */
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

/** The catalog document: registered with the env and printed by `--list`. */
export interface Catalog {
  providers: CatalogProvider[];
}

/** The catalog with no brief: what `--list` prints outside a trial. */
export const EMPTY_CATALOG: Catalog = { providers: [] };

/**
 * One provider as the brief's directory lists it. `path` is the directory
 * path (e.g. "/p/groq"); `base_url` is only used when the caller already
 * resolved an absolute dial URL (GET /instance) and no `path` is present.
 */
export interface BriefProvider {
  id: string;
  name?: string;
  wire: Wire;
  auth: AuthScheme;
  key_env: string;
  key: string;
  path?: string;
  base_url?: string;
}

/** One row of the brief's sealed rating table (free models only). */
export interface RatingRow {
  model: string;
  provider: string;
  rating_milli?: number;
  context_length?: number | null;
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  free?: boolean;
}

/** The brief, as parsed from MAKEFUN_PROMPT_FILE: directory + rating table. */
export interface Brief {
  providers: BriefProvider[];
  ratings: RatingRow[];
}

/** A provider with the full URL its list route is dialled at. */
export interface DialProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  auth: AuthScheme;
  key: string;
}

/** Timeout for a provider's model-list probe. */
const PROBE_TIMEOUT_MS = 5000;

/** The anthropic-version header sent on the Anthropic wire. */
const ANTHROPIC_VERSION = "2023-06-01";

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

/** A context length is a finite number, else unknown (`null`). */
function toContextLength(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

/**
 * Authorization headers for one provider call.
 *
 * The auth scheme names the header — never the wire: `bearer` sends
 * `Authorization: Bearer <key>`, `x-api-key` sends `x-api-key: <key>`,
 * `x-goog-api-key` sends `x-goog-api-key: <key>`, and `none` sends no key.
 * The Anthropic wire additionally sends `anthropic-version: 2023-06-01`.
 */
export function authHeaders(
  auth: AuthScheme | string,
  key: string,
  wire: Wire | string,
): Record<string, string> {
  const headers: Record<string, string> = {};
  if (auth === "bearer") {
    headers["Authorization"] = `Bearer ${key}`;
  } else if (auth === "x-api-key") {
    headers["x-api-key"] = key;
  } else if (auth === "x-goog-api-key") {
    headers["x-goog-api-key"] = key;
  }
  if (wire === "anthropic") {
    headers["anthropic-version"] = ANTHROPIC_VERSION;
  }
  return headers;
}

/**
 * The model-list URL for a provider: `<base>/v1/models`, or
 * `<base>/v1beta/models` on the Gemini wire.
 */
export function listUrl(provider: DialProvider): string {
  const base = provider.base_url.replace(/\/+$/, "");
  return provider.wire === "gemini" ? `${base}/v1beta/models` : `${base}/v1/models`;
}

/**
 * Normalize one raw model-list entry to a CatalogModel, or null when the
 * entry names no model.
 *
 * - id: `id`, else a Gemini-style `name` (`models/<id>`).
 * - free: the entry's `free` flag when present; else free when the entry's
 *   `pricing` lists only zero prices; else free (a listing with no price
 *   information is treated as free).
 * - context_length: the entry's `context_length`, else unknown.
 */
function normalizeModelEntry(entry: unknown): CatalogModel | null {
  if (!isRecord(entry)) return null;
  let id: string | null = null;
  if (typeof entry.id === "string" && entry.id.length > 0) {
    id = entry.id;
  } else if (typeof entry.name === "string" && entry.name.length > 0) {
    id = entry.name.startsWith("models/") ? entry.name.slice("models/".length) : entry.name;
  }
  if (id === null || id.length === 0) return null;

  let free: boolean | null = null;
  if (typeof entry.free === "boolean") {
    free = entry.free;
  } else if (isRecord(entry.pricing)) {
    const prices = Object.values(entry.pricing).filter(
      (v): v is number => typeof v === "number",
    );
    if (prices.length > 0) free = prices.every((p) => p === 0);
  }
  if (free === null) free = true;

  return {
    id,
    free,
    context_length: toContextLength(entry.context_length ?? entry.contextLength),
  };
}

/**
 * Normalize a model-list body: the OpenAI/Anthropic `{data: [...]}` shape
 * or the Gemini `{models: [...]}` shape. Anything else is not a listing.
 * Duplicate ids keep their first occurrence.
 */
function normalizeListing(body: unknown): CatalogModel[] {
  if (!isRecord(body)) return [];
  const raw = Array.isArray(body.data) ? body.data : body.models;
  if (!Array.isArray(raw)) return [];
  const seen = new Set<string>();
  const models: CatalogModel[] = [];
  for (const entry of raw) {
    const model = normalizeModelEntry(entry);
    if (model === null || seen.has(model.id)) continue;
    seen.add(model.id);
    models.push(model);
  }
  return models;
}

/**
 * GET one provider's model list (5 s timeout) and normalize it. Any
 * failure — a refused connection, a timeout, a non-2xx status, or a body
 * that is not the expected JSON — yields an empty list, never a throw, so
 * one dead provider cannot take the catalog down with it.
 */
export async function probeProvider(provider: DialProvider): Promise<CatalogModel[]> {
  // No dial URL, no probe: a provider without a base URL cannot be listed.
  // This also keeps a malformed brief from turning into a fetch of "".
  if (typeof provider.base_url !== "string" || provider.base_url.length === 0) return [];
  try {
    const res = await fetch(listUrl(provider), {
      method: "GET",
      headers: authHeaders(provider.auth, provider.key, provider.wire),
      signal: AbortSignal.timeout(PROBE_TIMEOUT_MS),
    });
    if (!res.ok) return [];
    return normalizeListing(await res.json());
  } catch {
    return [];
  }
}

/** Directory path a catalog reports for a provider, verbatim from the brief. */
function catalogBaseUrl(provider: BriefProvider): string {
  if (typeof provider.path === "string" && provider.path.length > 0) return provider.path;
  if (typeof provider.base_url === "string" && provider.base_url.length > 0) return provider.base_url;
  return "";
}

/** Full URL a provider is dialled at: env origin + directory path. */
function dialBaseUrl(envUrl: string, provider: BriefProvider): string {
  const env = envUrl.replace(/\/+$/, "");
  if (typeof provider.base_url === "string" && /^https?:\/\//.test(provider.base_url)) {
    return provider.base_url.replace(/\/+$/, "");
  }
  const path = catalogBaseUrl(provider);
  if (path.length === 0) return env;
  return env + (path.startsWith("/") ? path : `/${path}`);
}

function ratingModelId(row: RatingRow): string | null {
  if (typeof row.model === "string" && row.model.length > 0) return row.model;
  const alt = (row as { id?: unknown }).id;
  return typeof alt === "string" && alt.length > 0 ? alt : null;
}

/** Alphabetical order by model id (plain code-unit order). */
function byModelId(a: CatalogModel, b: CatalogModel): number {
  if (a.id < b.id) return -1;
  if (a.id > b.id) return 1;
  return 0;
}

/**
 * Build the catalog from a brief: probe every provider's list route in
 * parallel, then merge each listing with the brief's rating table.
 *
 * The rating table wins for a model's `free` flag and `context_length`
 * wherever it knows the model. A model the table knows but the listing
 * does not mention is still included (with the table's values); a model
 * the listing names but the table does not know is included with the
 * listing's values. Models are sorted alphabetically per provider, and
 * providers keep the brief's order.
 */
export async function buildCatalog(brief: Brief, envUrl: string): Promise<Catalog> {
  const directory: BriefProvider[] = Array.isArray(brief?.providers) ? brief.providers : [];
  const ratings: RatingRow[] = Array.isArray(brief?.ratings) ? brief.ratings : [];

  // Rating rows indexed by provider, then by model id.
  const table = new Map<string, Map<string, RatingRow>>();
  for (const row of ratings) {
    if (!isRecord(row)) continue;
    const modelId = ratingModelId(row as RatingRow);
    if (modelId === null || typeof row.provider !== "string") continue;
    let byModel = table.get(row.provider);
    if (byModel === undefined) {
      byModel = new Map<string, RatingRow>();
      table.set(row.provider, byModel);
    }
    byModel.set(modelId, row as RatingRow);
  }

  const dialProviders: DialProvider[] = directory.map((provider) => ({
    id: provider.id,
    base_url: dialBaseUrl(envUrl, provider),
    key_env: typeof provider.key_env === "string" ? provider.key_env : "",
    wire: provider.wire,
    auth: typeof provider.auth === "string" ? provider.auth : "none",
    key: typeof provider.key === "string" ? provider.key : "",
  }));
  const listings = await Promise.all(dialProviders.map((p) => probeProvider(p)));

  const providers: CatalogProvider[] = directory.map((provider, i) => {
    // Merge order matters. Start from the live listing (it is the source
    // for which ids the provider actually serves), then overlay the rating
    // table: the table is the sealed truth for `free` and `context_length`,
    // so wherever it knows a model its values replace the listing's. A
    // table row with no listing counterpart is still added — the table is
    // what the scorer compares the catalog against — while a listing model
    // the table never heard of keeps the listing's own values.
    const merged = new Map<string, CatalogModel>();
    for (const model of listings[i]) merged.set(model.id, { ...model });
    const rows = table.get(provider.id);
    if (rows !== undefined) {
      for (const [modelId, row] of rows) {
        const listed = merged.get(modelId);
        merged.set(modelId, {
          id: modelId,
          free: typeof row.free === "boolean" ? row.free : (listed?.free ?? true),
          context_length:
            row.context_length !== undefined
              ? toContextLength(row.context_length)
              : (listed?.context_length ?? null),
        });
      }
    }
    return {
      id: provider.id,
      base_url: catalogBaseUrl(provider),
      key_env: typeof provider.key_env === "string" ? provider.key_env : "",
      wire: provider.wire,
      models: [...merged.values()].sort(byModelId),
    };
  });
  return { providers };
}

/**
 * A provider entry's free models, in catalog order.
 */
export function freeModelsOf(entry: CatalogProvider | undefined | null): CatalogModel[] {
  if (entry === undefined || entry === null || !Array.isArray(entry.models)) return [];
  return entry.models.filter((model) => model.free === true);
}

/**
 * Every free model id in the catalog, deduplicated, in catalog order
 * (provider order, then model order). This is the id set the server's
 * `GET /v1/models` lists, and the set the registration carries.
 */
export function servedFreeModelIds(catalog: Catalog | undefined | null): string[] {
  if (catalog === undefined || catalog === null || !Array.isArray(catalog.providers)) return [];
  const seen = new Set<string>();
  const ids: string[] = [];
  for (const provider of catalog.providers) {
    for (const model of freeModelsOf(provider)) {
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      ids.push(model.id);
    }
  }
  return ids;
}

/** The catalog entry for one provider id, or undefined when absent. */
export function catalogEntry(
  catalog: Catalog | undefined | null,
  providerId: string,
): CatalogProvider | undefined {
  if (catalog === undefined || catalog === null || !Array.isArray(catalog.providers)) return undefined;
  return catalog.providers.find((provider) => provider.id === providerId);
}
