// main.ts — the free-model aggregator: a proxy that answers every request
// with the highest-rated free model that meets the request's requirements
// and still has provider budget, speaking that provider's wire, and
// returns the answer in the wire the client used.
//
// Flow at startup:
//   1. Read the brief (MAKEFUN_PROMPT_FILE): directory, ratings, manifest.
//   2. Probe every provider's model list; build the catalog.
//   3. Listen on a port; POST $MAKEFUN_ENV_URL/register; poll /work until
//      done:true, then exit 0.
//   `node main.ts --list` prints the catalog byte-for-byte as registered
//   and exits 0 ({"providers":[]} outside a trial).
//
// Per request the router picks the best legal (provider, model); a refusal
// (402/429/5xx) moves to the next-best provider — never a second call to
// the same provider for the same request, and never two concurrent calls
// to the same provider. A provider that fails three times in a row, or
// past its budget ceiling, is dead for the rest of the trial.
import { createServer } from "node:http";
import type { IncomingMessage, ServerResponse } from "node:http";
import { readBriefFile } from "./lib/brief.ts";
import type { Brief, Requirements, WorkloadRequest } from "./lib/brief.ts";
import { buildCatalog, servedFreeModelIds } from "./lib/catalog.ts";
import type { Catalog } from "./lib/catalog.ts";
import { authHeaders } from "./lib/catalog.ts";
import { Router } from "./lib/router.ts";
import type { Candidate } from "./lib/router.ts";
import {
  anthropicToOpenAI,
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
} from "./lib/translate.ts";
import type { JsonRecord, ProviderAnswer } from "./lib/translate.ts";

const UPSTREAM_TIMEOUT_MS = 10_000;
const POLL_INTERVAL_MS = 100;

function env(name: string): string | undefined {
  const value = process.env[name];
  return value === undefined || value === "" ? undefined : value;
}

/** The requirements of a request, from the manifest when known. */
function requirementsFor(
  brief: Brief,
  body: JsonRecord,
  arrivalIndex: number,
): { requires: Requirements; manifest: WorkloadRequest | undefined } {
  const manifest = brief.manifest[arrivalIndex];
  if (manifest) return { requires: manifest.requires, manifest };
  // Fallback when the manifest is absent: infer from the body itself.
  const requires: Requirements = {};
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let hasTools = false;
  let hasVision = false;
  for (const raw of messages) {
    if (raw === null || typeof raw !== "object") continue;
    const content = (raw as JsonRecord).content;
    if (Array.isArray(content)) {
      for (const part of content) {
        if (part !== null && typeof part === "object") {
          const rec = part as JsonRecord;
          if (rec.type === "image_url" || rec.type === "image") hasVision = true;
        }
      }
    }
  }
  if (Array.isArray(body.tools) && body.tools.length > 0) hasTools = true;
  if (hasTools) requires.tools = true;
  if (hasVision) requires.vision = true;
  const fmt = body.response_format;
  if (
    fmt !== null &&
    typeof fmt === "object" &&
    ((fmt as JsonRecord).type === "json_object" ||
      (fmt as JsonRecord).type === "json_schema")
  ) {
    requires.json = true;
  }
  return { requires, manifest: undefined };
}

/** Whether the client asked for JSON mode (OpenAI response_format only). */
function clientJsonMode(api: "openai" | "anthropic", body: JsonRecord): boolean {
  if (api !== "openai") return false;
  const fmt = body.response_format;
  return (
    fmt !== null &&
    typeof fmt === "object" &&
    ((fmt as JsonRecord).type === "json_object" ||
      (fmt as JsonRecord).type === "json_schema")
  );
}

/** The body to send upstream for a candidate, in the provider's wire. */
function upstreamRequest(
  candidate: Candidate,
  clientApi: "openai" | "anthropic",
  body: JsonRecord,
  stream: boolean,
): { path: string; body: JsonRecord } {
  const wire = candidate.wire;
  const model = candidate.modelId;
  if (wire === "openai") {
    const upstream = clientApi === "openai" ? { ...body, model } : anthropicToOpenAI(body, model);
    upstream.model = model;
    if (stream) {
      upstream.stream = true;
      upstream.stream_options = { include_usage: true };
    } else {
      delete upstream.stream;
      delete upstream.stream_options;
    }
    return { path: "/v1/chat/completions", body: upstream };
  }
  if (wire === "anthropic") {
    const upstream = clientApi === "anthropic" ? { ...body, model } : openAIToAnthropic(body, model);
    upstream.model = model;
    if (stream) upstream.stream = true;
    else delete upstream.stream;
    return { path: "/v1/messages", body: upstream };
  }
  // gemini
  return toGemini(clientApi, body, model, stream);
}

/** A per-provider mutex: one in-flight call per provider at a time. */
class ProviderLocks {
  private readonly chains = new Map<string, Promise<void>>();

  async run<T>(providerId: string, fn: () => Promise<T>): Promise<T> {
    const previous = this.chains.get(providerId) ?? Promise.resolve();
    let release: () => void = () => {};
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    this.chains.set(providerId, previous.then(() => gate));
    await previous;
    try {
      return await fn();
    } finally {
      release();
    }
  }
}

/** One upstream call. Returns the parsed answer, or null on refusal. */
async function callUpstream(
  candidate: Candidate,
  clientApi: "openai" | "anthropic",
  body: JsonRecord,
  stream: boolean,
  envUrl: string,
): Promise<ProviderAnswer | null> {
  const { path, body: upstreamBody } = upstreamRequest(
    candidate,
    clientApi,
    body,
    stream,
  );
  const url = envUrl + candidate.provider.path + path;
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...authHeaders(candidate.provider.auth, candidate.provider.key, candidate.wire),
  };
  let res: Response;
  try {
    res = await fetch(url, {
      method: "POST",
      headers,
      body: JSON.stringify(upstreamBody),
      signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
    });
  } catch {
    return null;
  }
  if (!res.ok) return null;
  const text = await res.text();
  if (stream) {
    return parseProviderSse(candidate.wire, text);
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    return null;
  }
  if (parsed === null || typeof parsed !== "object") return null;
  const record = parsed as JsonRecord;
  if (candidate.wire === "openai") return fromOpenAI(record);
  if (candidate.wire === "anthropic") return fromAnthropic(record);
  return fromGemini(record);
}

/** Read a request body as JSON (cap 32 MiB). */
function readBody(req: IncomingMessage): Promise<JsonRecord | null> {
  return new Promise((resolve) => {
    const chunks: Buffer[] = [];
    let size = 0;
    req.on("data", (chunk: Buffer) => {
      size += chunk.length;
      if (size > 32 * 1024 * 1024) {
        resolve(null);
        return;
      }
      chunks.push(chunk);
    });
    req.on("end", () => {
      try {
        const parsed: unknown = JSON.parse(Buffer.concat(chunks).toString("utf8"));
        resolve(
          parsed !== null && typeof parsed === "object" && !Array.isArray(parsed)
            ? (parsed as JsonRecord)
            : null,
        );
      } catch {
        resolve(null);
      }
    });
    req.on("error", () => resolve(null));
  });
}

/** The request-serving core: route one client request through providers. */
class Aggregator {
  private readonly brief: Brief;
  private readonly catalog: Catalog;
  private readonly router: Router;
  private readonly locks = new ProviderLocks();
  private readonly envUrl: string;
  private arrivalIndex = 0;

  constructor(brief: Brief, catalog: Catalog, envUrl: string) {
    this.brief = brief;
    this.catalog = catalog;
    this.router = new Router(brief, catalog);
    this.envUrl = envUrl;
  }

  /** Answer one request, returning {status, contentType, body}. */
  async handle(
    api: "openai" | "anthropic",
    body: JsonRecord,
  ): Promise<{ status: number; contentType: string; body: string }> {
    const index = this.arrivalIndex;
    this.arrivalIndex += 1;
    const stream = body.stream === true;
    const { requires } = requirementsFor(this.brief, body, index);
    const jsonMode = clientJsonMode(api, body);
    const candidates = this.router.candidates(requires, api, jsonMode);
    const tried = new Set<string>();
    while (true) {
      const candidate = this.router.next(candidates, tried);
      if (!candidate) break;
      tried.add(candidate.providerId);
      const answer = await this.locks.run(candidate.providerId, () =>
        callUpstream(candidate, api, body, stream, this.envUrl),
      );
      if (answer === null) {
        this.router.recordFailure(candidate.providerId);
        continue;
      }
      this.router.recordSuccess(candidate.providerId);
      if (api === "openai") {
        if (stream) {
          return {
            status: 200,
            contentType: "text/event-stream",
            body: renderOpenAIStream(candidate.modelId, answer),
          };
        }
        return {
          status: 200,
          contentType: "application/json",
          body: JSON.stringify(renderOpenAIJson(candidate.modelId, answer)),
        };
      }
      if (stream) {
        return {
          status: 200,
          contentType: "text/event-stream",
          body: renderAnthropicStream(candidate.modelId, answer),
        };
      }
      return {
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(renderAnthropicJson(candidate.modelId, answer)),
      };
    }
    // No provider could answer: a well-formed error in the client's wire.
    // (The capacity math makes this near-impossible: ~78 calls for 20
    // requests, and the first call to any provider always succeeds.)
    if (api === "openai") {
      return {
        status: 503,
        contentType: "application/json",
        body: JSON.stringify({
          error: { message: "all providers refused", type: "server_error" },
        }),
      };
    }
    return {
      status: 503,
      contentType: "application/json",
      body: JSON.stringify({ type: "error", error: { type: "overloaded_error", message: "all providers refused" } }),
    };
  }

  /** GET /v1/models: the free model ids in the registered catalog. */
  modelsList(auth: { bearer: boolean; xApiKey: boolean }): JsonRecord {
    const ids = servedFreeModelIds(this.catalog);
    const now = "2026-01-01T00:00:00Z";
    if (auth.bearer) {
      return {
        object: "list",
        data: ids.map((id) => ({
          id,
          object: "model",
          created: 1767225600,
          owned_by: "aggregator",
        })),
      };
    }
    if (auth.xApiKey) {
      return {
        object: "list",
        data: ids.map((id) => ({
          id,
          created_at: now,
          display_name: id,
          type: "model",
        })),
        first_id: ids[0] ?? null,
        last_id: ids[ids.length - 1] ?? null,
        has_more: false,
      };
    }
    return { object: "list", data: [] };
  }
}

async function main(): Promise<number> {
  const brief = readBriefFile(env("MAKEFUN_PROMPT_FILE"));
  const envUrl = env("MAKEFUN_ENV_URL") ?? "";
  const catalog = await buildCatalog(brief, envUrl);

  if (process.argv.includes("--list")) {
    process.stdout.write(`${JSON.stringify(catalog)}\n`);
    return 0;
  }

  const aggregator = new Aggregator(brief, catalog, envUrl);
  const server = createServer(async (req: IncomingMessage, res: ServerResponse) => {
    try {
      const url = new URL(req.url ?? "/", "http://127.0.0.1");
      if (req.method === "GET" && url.pathname === "/v1/models") {
        const hasBearer = (req.headers.authorization ?? "").startsWith("Bearer ");
        const hasXApiKey = req.headers["x-api-key"] !== undefined;
        const list = aggregator.modelsList({ bearer: hasBearer, xApiKey: hasXApiKey });
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify(list));
        return;
      }
      if (
        req.method === "POST" &&
        (url.pathname === "/v1/chat/completions" || url.pathname === "/v1/messages")
      ) {
        const body = await readBody(req);
        if (body === null) {
          res.writeHead(400, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: { message: "invalid JSON body" } }));
          return;
        }
        const api = url.pathname === "/v1/messages" ? "anthropic" : "openai";
        const result = await aggregator.handle(api, body);
        res.writeHead(result.status, { "Content-Type": result.contentType });
        res.end(result.body);
        return;
      }
      res.writeHead(404, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: { message: "not found" } }));
    } catch {
      if (!res.headersSent) {
        res.writeHead(500, { "Content-Type": "application/json" });
      }
      res.end(JSON.stringify({ error: { message: "internal error" } }));
    }
  });

  const port = await new Promise<number>((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "0.0.0.0", () => {
      const address = server.address();
      if (address !== null && typeof address === "object") resolve(address.port);
      else reject(new Error("could not listen"));
    });
  });

  if (!envUrl) {
    // No env to register with: stay up until killed (dev convenience).
    return 0;
  }

  try {
    const reg = await fetch(`${envUrl}/register`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ protocol_version: 1, port, catalog }),
      signal: AbortSignal.timeout(5_000),
    });
    if (!reg.ok) {
      server.close();
      return 1;
    }
  } catch {
    server.close();
    return 1;
  }

  for (;;) {
    try {
      const work = await fetch(`${envUrl}/work`, {
        signal: AbortSignal.timeout(5_000),
      });
      if (work.ok) {
        const state: unknown = await work.json();
        if (
          state !== null &&
          typeof state === "object" &&
          (state as JsonRecord).done === true
        ) {
          break;
        }
      }
    } catch {
      // A transient poll failure is not a reason to stop.
    }
    await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));
  }
  server.close();
  return 0;
}

main()
  .then((code) => process.exit(code))
  .catch(() => process.exit(1));
