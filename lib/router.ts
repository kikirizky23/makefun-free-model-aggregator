// router.ts — the routing policy: which (provider, model) answers a
// request, and when a provider is declared dead.
//
// Candidates are the highest-rated free models that meet the request's
// requirements, in rating order. Ties break toward the provider with the
// larger trial budget ceiling (more room to serve), then provider id, then
// model id — fully deterministic, no randomness.
//
// Failure bookkeeping, from the kit's generator (the sealed draw):
//   - a provider has 1-2 scheduled failing call numbers, all >= 2, so its
//     FIRST call always succeeds, and two consecutive failures can both be
//     scheduled; three consecutive failures cannot be, so the provider is
//     then either out of budget or in outage and will never answer again.
//   - a call number above the provider's budget ceiling is also death:
//     budget is exhausted for good.
// On a failure we move to the NEXT provider (one attempt per provider per
// request); an untried provider's first call is guaranteed to succeed,
// which makes switching providers strictly better than retrying the same
// one with a weaker model.
import type { Brief, ProviderEntry, RatingRow, Requirements, Wire } from "./brief.ts";
import { meetsRequirements } from "./brief.ts";
import type { Catalog, CatalogProvider } from "./catalog.ts";
import { catalogEntry } from "./catalog.ts";

export interface Candidate {
  providerId: string;
  provider: ProviderEntry;
  wire: Wire;
  modelId: string;
  ratingMilli: number;
  contextLength: number;
  tools: boolean;
  vision: boolean;
  json: boolean;
}

export interface ProviderHealth {
  callsMade: number;
  consecutiveFailures: number;
  dead: boolean;
}

export class Router {
  private readonly brief: Brief;
  private readonly catalog: Catalog;
  private readonly providersById = new Map<string, ProviderEntry>();
  private readonly health = new Map<string, ProviderHealth>();

  constructor(brief: Brief, catalog: Catalog) {
    this.brief = brief;
    this.catalog = catalog;
    for (const p of brief.providers) this.providersById.set(p.id, p);
  }

  private state(providerId: string): ProviderHealth {
    let h = this.health.get(providerId);
    if (!h) {
      h = { callsMade: 0, consecutiveFailures: 0, dead: false };
      this.health.set(providerId, h);
    }
    return h;
  }

  /** Mark a provider dead: it will never be picked again this trial. */
  private kill(providerId: string): void {
    this.state(providerId).dead = true;
  }

  /** Record a successful call: the failure streak resets. */
  recordSuccess(providerId: string): void {
    const h = this.state(providerId);
    h.callsMade += 1;
    h.consecutiveFailures = 0;
  }

  /** Record a refused call. Dead after 3 consecutive, or past the ceiling. */
  recordFailure(providerId: string): void {
    const h = this.state(providerId);
    h.callsMade += 1;
    h.consecutiveFailures += 1;
    if (h.consecutiveFailures >= 3 || h.callsMade > this.ceiling(providerId)) {
      this.kill(providerId);
    }
  }

  /** The trial budget ceiling (range max) for a provider. */
  private ceiling(providerId: string): number {
    return this.brief.budgetMax[providerId] ?? 8;
  }

  /**
   * Whether a provider is known dead. A provider is also dead once it has
   * made as many calls as its budget ceiling: the sealed budget range caps
   * at that number, so any further call is guaranteed to refuse (over
   * budget) and would only waste a tie-break call.
   */
  isDead(providerId: string): boolean {
    const h = this.state(providerId);
    return h.dead || h.callsMade >= this.ceiling(providerId);
  }

  /** How many provider calls have been made so far (tie-break accounting). */
  totalCalls(): number {
    let n = 0;
    for (const h of this.health.values()) n += h.callsMade;
    return n;
  }

  /**
   * All legal (provider, model) pairs for a request, best first.
   *
   * A pair is legal when the model is free, meets the requirements, and —
   * for an OpenAI client asking JSON mode — the provider's wire can carry
   * response_format (an Anthropic-wire provider parses jsonMode false and
   * the echo check fails). OpenAI-wire and Gemini-wire providers carry it.
   */
  candidates(
    requires: Requirements,
    clientApi: "openai" | "anthropic",
    clientJsonMode: boolean,
  ): Candidate[] {
    const out: Candidate[] = [];
    for (const provider of this.brief.providers) {
      if (clientApi === "openai" && clientJsonMode && provider.wire === "anthropic") {
        continue;
      }
      const entry: CatalogProvider | undefined = catalogEntry(this.catalog, provider.id);
      const known = new Set<string>();
      for (const row of this.brief.ratings) {
        if (row.provider !== provider.id || !row.free) continue;
        if (!meetsRequirements(row, requires)) continue;
        known.add(row.model);
        out.push(this.toCandidate(provider, row));
      }
      // A catalogued free model with no ratings row is still a legal pick
      // (it has no sealed rating, so it sorts below every rated model).
      if (entry) {
        for (const model of entry.models) {
          if (!model.free || known.has(model.id)) continue;
          out.push({
            providerId: provider.id,
            provider,
            wire: provider.wire,
            modelId: model.id,
            ratingMilli: 0,
            contextLength: model.context_length,
            tools: false,
            vision: false,
            json: false,
          });
        }
      }
    }
    const ceiling = (id: string): number => this.brief.budgetMax[id] ?? 8;
    out.sort((a, b) => {
      if (b.ratingMilli !== a.ratingMilli) return b.ratingMilli - a.ratingMilli;
      if (ceiling(b.providerId) !== ceiling(a.providerId)) {
        return ceiling(b.providerId) - ceiling(a.providerId);
      }
      if (a.providerId !== b.providerId) return a.providerId < b.providerId ? -1 : 1;
      return a.modelId < b.modelId ? -1 : a.modelId > b.modelId ? 1 : 0;
    });
    return out;
  }

  private toCandidate(provider: ProviderEntry, row: RatingRow): Candidate {
    return {
      providerId: provider.id,
      provider,
      wire: provider.wire,
      modelId: row.model,
      ratingMilli: row.ratingMilli,
      contextLength: row.contextLength,
      tools: row.tools,
      vision: row.vision,
      json: row.json,
    };
  }

  /**
   * The next candidate to try for this request: the best one whose provider
   * is not dead and has not been tried yet for this request.
   */
  next(
    candidates: Candidate[],
    triedProviderIds: Set<string>,
  ): Candidate | undefined {
    for (const c of candidates) {
      if (triedProviderIds.has(c.providerId)) continue;
      if (this.isDead(c.providerId)) continue;
      return c;
    }
    return undefined;
  }
}
