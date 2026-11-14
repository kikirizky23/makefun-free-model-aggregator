# Free model aggregator

A proxy server that answers every request with the best free model it can get.

It reads the trial brief (ten providers, their wires and auth schemes, the
sealed quality table, the workload manifest), probes every provider's model
list, and registers one catalog. For each request it walks candidate free
models in rating order — one attempt per provider, serialized per provider —
fails over on 402/429, and returns the provider's answer in the wire the
client used (OpenAI chat completions or Anthropic messages, JSON or SSE).

Local grade (process driver, 3 practice trials): **99/100 on every trial**
(quality 99 — the ceiling, since the top-rated model's budget cannot cover
every request; served 100, wire 100, failover 100, catalog 100; replays
byte-identical).

Run: `node main.ts` (Node 22, stdlib only). `node main.ts --list` prints the
registered catalog and exits 0.

make.fun competition: free-model-aggregator-01
