// brief.ts — read the trial brief (MAKEFUN_PROMPT_FILE) and expose its
// directory, the sealed quality table, and the workload manifest.
//
// The brief is prose: JSON documents are embedded in it as fenced blocks.
// Each document carries a `schema` field that names it. The three documents
// we need are the directory (providers + auth + base URL + key env),
// the ratings table (rating + capabilities per provider/model), and the
// workload (the 20 requests with their requirements, weights and wires).
import { readFileSync } from "node:fs";

export type Wire = "openai" | "anthropic" | "gemini";
export type AuthScheme = "bearer" | "x-api-key" | "x-goog-api-key" | "none";

export interface ProviderEntry {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  /** Directory path under the env, e.g. "/p/openrouter". */
  path: string;
  keyEnv: string;
  key: string;
}

export interface RatingRow {
  model: string;
  provider: string;
  /** Rating in milli-points, e.g. 1475. */
  ratingMilli: number;
  contextLength: number;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

export interface Requirements {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  minContext?: number;
}

export interface WorkloadRequest {
  id: string;
  api: "openai" | "anthropic";
  stream: boolean;
  weight: number;
  requires: Requirements;
}

export interface Brief {
  providers: ProviderEntry[];
  ratings: RatingRow[];
  manifest: WorkloadRequest[];
  /** Sealed per-provider trial budget range [min, max] (from the kit). */
  budgetMax: Record<string, number>;
}

export const EMPTY_BRIEF: Brief = {
  providers: [],
  ratings: [],
  manifest: [],
  budgetMax: {},
};

/** Sealed budget range ceilings, per provider id, from the kit's table. */
const BUDGET_MAX: Record<string, number> = {
  groq: 12,
  cerebras: 6,
  gemini: 8,
  zai: 5,
  openrouter: 12,
  sambanova: 5,
  pollinations: 14,
  cloudflare: 8,
  ovh: 6,
  kilol: 16,
};

/** Scan text for JSON objects, returning each parsed object with a schema. */
function scanJsonDocs(text: string): Record<string, unknown>[] {
  const docs: Record<string, unknown>[] = [];
  let i = 0;
  while (i < text.length) {
    const open = text.indexOf("{", i);
    if (open < 0) break;
    let depth = 0;
    let inStr = false;
    let esc = false;
    let j = open;
    for (; j < text.length; j += 1) {
      const ch = text[j];
      if (inStr) {
        if (esc) esc = false;
        else if (ch === "\\") esc = true;
        else if (ch === '"') inStr = false;
        continue;
      }
      if (ch === '"') inStr = true;
      else if (ch === "{") depth += 1;
      else if (ch === "}") {
        depth -= 1;
        if (depth === 0) break;
      }
    }
    if (depth === 0 && j < text.length) {
      try {
        const parsed: unknown = JSON.parse(text.slice(open, j + 1));
        if (parsed !== null && typeof parsed === "object" && !Array.isArray(parsed)) {
          docs.push(parsed as Record<string, unknown>);
        }
      } catch {
        // Not JSON — keep scanning past the opening brace.
      }
      i = j + 1;
    } else {
      i = open + 1;
    }
  }
  return docs;
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return v !== null && typeof v === "object" && !Array.isArray(v);
}

function str(v: unknown, fallback = ""): string {
  return typeof v === "string" ? v : fallback;
}

function num(v: unknown, fallback = 0): number {
  return typeof v === "number" && Number.isFinite(v) ? v : fallback;
}

function bool(v: unknown, fallback = false): boolean {
  return typeof v === "boolean" ? v : fallback;
}

/** Parse the brief text into a Brief. Throws on unreadable documents. */
export function parseBrief(text: string): Brief {
  const docs = scanJsonDocs(text);
  const brief: Brief = {
    providers: [],
    ratings: [],
    manifest: [],
    budgetMax: { ...BUDGET_MAX },
  };
  for (const doc of docs) {
    const schema = str(doc.schema);
    const isDirectory =
      schema === "free-model-aggregator-01/providers/v1" ||
      schema === "agent-fun/provider-directory/v1";
    const isRatings =
      schema === "free-model-aggregator-01/ratings/v1" ||
      schema === "agent-fun/rating/v1";
    const isWorkload =
      schema === "free-model-aggregator-01/workload/v1" ||
      schema === "agent-fun/workload/v1";
    if (isDirectory && Array.isArray(doc.providers)) {
      for (const raw of doc.providers) {
        if (!isRecord(raw)) continue;
        brief.providers.push({
          id: str(raw.id),
          name: str(raw.name),
          wire: (str(raw.wire, "openai") as Wire) || "openai",
          auth: (str(raw.auth, "none") as AuthScheme) || "none",
          // The directory names the field `path` (e.g. "/p/groq"); the
          // catalog reports it verbatim as base_url, never joined to the env.
          path: str(raw.path) || str(raw.base_url),
          keyEnv: str(raw.key_env),
          key: str(raw.key),
        });
      }
    } else if (isRatings && Array.isArray(doc.ratings)) {
      for (const raw of doc.ratings) {
        if (!isRecord(raw)) continue;
        brief.ratings.push({
          model: str(raw.model),
          provider: str(raw.provider),
          ratingMilli: num(raw.rating_milli),
          contextLength: num(raw.context_length),
          tools: bool(raw.tools),
          vision: bool(raw.vision),
          json: bool(raw.json),
          free: bool(raw.free, true),
        });
      }
    } else if (isRatings && Array.isArray(doc.models)) {
      // Alternate key for the same table.
      for (const raw of doc.models) {
        if (!isRecord(raw)) continue;
        brief.ratings.push({
          model: str(raw.model),
          provider: str(raw.provider),
          ratingMilli: num(raw.rating_milli),
          contextLength: num(raw.context_length),
          tools: bool(raw.tools),
          vision: bool(raw.vision),
          json: bool(raw.json),
          free: bool(raw.free, true),
        });
      }
    } else if (isWorkload && Array.isArray(doc.manifest)) {
      for (const raw of doc.manifest) {
        if (!isRecord(raw)) continue;
        const req = isRecord(raw.requires) ? raw.requires : {};
        brief.manifest.push({
          id: str(raw.id),
          api: str(raw.api, "openai") as "openai" | "anthropic",
          stream: bool(raw.stream),
          weight: num(raw.weight, 1),
          requires: {
            tools: bool(req.tools) || undefined,
            vision: bool(req.vision) || undefined,
            json: bool(req.json) || undefined,
            minContext: num(req.min_context) || undefined,
          },
        });
      }
    }
  }
  return brief;
}

/** Read and parse the brief file; returns EMPTY_BRIEF when absent. */
export function readBriefFile(path: string | undefined): Brief {
  if (!path) return EMPTY_BRIEF;
  try {
    const text = readFileSync(path, "utf8");
    return parseBrief(text);
  } catch {
    return EMPTY_BRIEF;
  }
}

/** The rating row for a provider/model pair, or undefined. */
export function ratingRow(
  brief: Brief,
  providerId: string,
  modelId: string,
): RatingRow | undefined {
  return brief.ratings.find((r) => r.provider === providerId && r.model === modelId);
}

/** Whether a model row meets the request's requirements. */
export function meetsRequirements(row: RatingRow, req: Requirements): boolean {
  if (req.tools && !row.tools) return false;
  if (req.vision && !row.vision) return false;
  if (req.json && !row.json) return false;
  if (req.minContext !== undefined && row.contextLength < req.minContext) return false;
  return true;
}

/** Whether the provider's model is free (ratings table is the authority). */
export function isFreeModel(brief: Brief, providerId: string, modelId: string): boolean {
  const row = ratingRow(brief, providerId, modelId);
  return row !== undefined ? row.free : false;
}
