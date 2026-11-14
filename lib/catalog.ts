// catalog.ts — probe every provider's model list at startup and build the
// catalog document that is registered with the env and answered by
// `node main.ts --list` and GET /v1/models.
//
// The catalog is the directory's truth about which models are free and how
// much context each one holds: the sealed ratings table is the authority
// for both, with the live listing filling in anything the table lacks.
// Every provider's listing is fetched once, up front, so no catalog answer
// ever depends on a provider answering mid-trial.
import type {
  AuthScheme,
  Brief,
  ProviderEntry,
  RatingRow,
  Wire,
} from "./brief.ts";
import { ratingRow } from "./brief.ts";

export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number;
}

export interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

export interface Catalog {
  providers: CatalogProvider[];
}

export const EMPTY_CATALOG: Catalog = { providers: [] };

const PROBE_TIMEOUT_MS = 5_000;

/** Auth headers for a probe, from the brief's own auth scheme. */
export function authHeaders(
  auth: AuthScheme,
  key: string,
  wire: Wire,
): Record<string, string> {
  const headers: Record<string, string> = {};
  if (auth === "bearer") headers.Authorization = `Bearer ${key}`;
  else if (auth === "x-api-key") headers["x-api-key"] = key;
  else if (auth === "x-goog-api-key") headers["x-goog-api-key"] = key;
  if (wire === "anthropic") headers["anthropic-version"] = "2023-06-01";
  return headers;
}

/** The URL of a provider's model list. Gemini lists under /v1beta. */
export function listUrl(provider: ProviderEntry): string {
  const base = provider.path;
  if (provider.wire === "gemini") {
    return `${base}/v1beta/models`;
  }
  return `${base}/v1/models`;
}

/** One model as a listing reports it, normalized to id/free/context. */
interface ListedModel {
  id: string;
  free: boolean;
  contextLength: number;
}

/** Fetch a provider's model list, normalized. Returns [] on any failure. */
export async function probeProvider(provider: ProviderEntry): Promise<ListedModel[]> {
  const url = listUrl(provider);
  try {
    const res = await fetch(url, {
      method: "GET",
      headers: authHeaders(provider.auth, provider.key, provider.wire),
      signal: AbortSignal.timeout(PROBE_TIMEOUT_MS),
    });
    if (!res.ok) return [];
    const body: unknown = await res.json();
    if (body === null || typeof body !== "object") return [];
    const raw = (body as Record<string, unknown>).data;
    if (!Array.isArray(raw)) return [];
    const models: ListedModel[] = [];
    for (const entry of raw) {
      if (entry === null || typeof entry !== "object") continue;
      const rec = entry as Record<string, unknown>;
      const id = typeof rec.id === "string" ? rec.id : "";
      if (!id) continue;
      const free =
        typeof rec.free === "boolean"
          ? rec.free
          : Array.isArray(rec.pricing)
            ? rec.pricing.every((p: unknown) => typeof p === "number" && p === 0)
            : true;
      const contextLength =
        typeof rec.context_length === "number" ? rec.context_length : 0;
      models.push({ id, free, contextLength });
    }
    return models;
  } catch {
    return [];
  }
}

/**
 * Build the catalog: probe every provider (in brief order, in parallel),
 * then merge the live listing with the ratings table. A model is free when
 * the ratings table says so; the listing only fills gaps. Models the table
 * knows but the listing omits are still catalogued — the table is sealed
 * truth, the listing is a probe.
 */
export async function buildCatalog(
  brief: Brief,
  envUrl: string,
): Promise<Catalog> {
  const catalog: Catalog = { providers: [] };
  const probes = await Promise.all(
    brief.providers.map(async (provider) => ({
      provider,
      listed: await probeProvider({
        ...provider,
        path: envUrl + provider.path,
      }),
    })),
  );
  for (const { provider, listed } of probes) {
    const byId = new Map<string, CatalogModel>();
    for (const entry of listed) {
      const row = ratingRow(brief, provider.id, entry.id);
      byId.set(entry.id, {
        id: entry.id,
        free: row !== undefined ? row.free : entry.free,
        context_length:
          row !== undefined && row.contextLength > 0
            ? row.contextLength
            : entry.contextLength,
      });
    }
    for (const row of brief.ratings) {
      if (row.provider !== provider.id) continue;
      if (byId.has(row.model)) continue;
      byId.set(row.model, {
        id: row.model,
        free: row.free,
        context_length: row.contextLength,
      });
    }
    const models = [...byId.values()].sort((a, b) =>
      a.id < b.id ? -1 : a.id > b.id ? 1 : 0,
    );
    catalog.providers.push({
      id: provider.id,
      base_url: provider.path,
      key_env: provider.keyEnv,
      wire: provider.wire,
      models,
    });
  }
  return catalog;
}

/** The free model ids a provider serves, from its catalog entry. */
export function freeModelsOf(entry: CatalogProvider): string[] {
  return entry.models.filter((m) => m.free).map((m) => m.id);
}

/** Every free model id in the catalog, de-duplicated, in catalog order. */
export function servedFreeModelIds(catalog: Catalog): string[] {
  const seen = new Set<string>();
  const ids: string[] = [];
  for (const provider of catalog.providers) {
    for (const model of provider.models) {
      if (!model.free) continue;
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      ids.push(model.id);
    }
  }
  return ids;
}

/** The catalog entry for a provider id, or undefined. */
export function catalogEntry(
  catalog: Catalog,
  providerId: string,
): CatalogProvider | undefined {
  return catalog.providers.find((p) => p.id === providerId);
}
