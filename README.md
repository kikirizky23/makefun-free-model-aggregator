# free-model-aggregator — v2 (global planner)

Server agregator model gratis untuk kontes `free-model-aggregator-01` di
make.fun. Server ini melayani dua API — OpenAI Chat Completions
(`POST /v1/chat/completions`) dan Anthropic Messages (`POST /v1/messages`),
masing-masing dalam JSON dan streaming (SSE) — dengan menjawab setiap
request memakai satu model gratis dari sepuluh provider yang diberikan
brief trial, serta `GET /v1/models` dalam bentuk daftar sesuai header
auth yang dipakai (`Bearer` -> gaya OpenAI, `x-api-key` saja -> gaya
Anthropic; keduanya -> Bearer menang).

## Apa yang berubah di v2

v1 memilih provider secara greedy per request: model dengan rating
tertinggi yang legal untuk request itu, lalu failover bila ditolak. Itu
memboroskan budget provider terbaik di request berbobot 1 dan kehabisan
di request berbobot 2 — padahal bobot 2 menyumbang dua kali lipat skor
quality.

v2 mengganti greedy dengan **global planner** (`lib/router.ts`):

- **Alokasi 1475 untuk bobot-2.** Model rating 1475
  (`z-ai/glm-5.3-flash:free` di OpenRouter) dicadangkan untuk request
  berbobot 2 yang bisa memakainya. Request berbobot 2 adalah tempat
  rating tertinggi paling berharga; planner menyusun draf untuk seluruh
  sisa trial sebelum request pertama dijawab, bukan per request.
- **Bobot-1 langsung ke fallback.** Request berbobot 1 (r07–r20) tidak
  lagi bersaing memperebutkan 1475: mereka langsung diarahkan ke
  fallback terbaik yang masih legal (mis. 1470/1465/1458), sehingga
  budget OpenRouter tetap utuh untuk bobot 2.
- **Matikan provider lebih cepat.** Satu penolakan 402/429 (budget habis
  atau jadwal gagal provider) tidak lagi ditunggu sampai berulang:
  router menandai provider yang polanya menunjukkan kehabisan budget
  sebagai mati lebih awal dan me-re-draft rencana sisa trial dari
  kegagalan terbaru, jadi request berikutnya tidak membuang panggilan
  ke provider yang sudah tidak bisa melayani.
- **Tabel budget berupa range.** `lib/brief.ts` memakai tabel range
  `[min, max]` per provider (mis. groq 9–12, openrouter 10–12,
  kilol 12–16) sebagai pengganti plafon tunggal v1; planner
  merencanakan alokasi terhadap ujung bawah range (angka aman) dan
  memakai ujung atas hanya sebagai cadangan.
- **Retry ke provider yang sama diizinkan** bila router memilihnya lagi
  dengan risiko terbaru (mis. penolakan `concurrent_requests` sesaat,
  bukan kehabisan budget).

`main.ts` sendiri tidak lagi berisi logika pemilihan apa pun: ia hanya
mengeksekusi rencana router (minta pilihan berikutnya, satu panggilan
upstream di bawah mutex provider itu, catat sukses/gagal, ulangi).

## Cara kerja

1. Startup membaca brief dari `MAKEFUN_PROMPT_FILE` (`lib/brief.ts`:
   memindai teks mentah, mengekstrak semua objek JSON berimbang, dan
   merakit direktori provider, tabel rating, dan manifest workload).
2. Katalog dibangun dengan mem-probe kesepuluh provider secara paralel
   (`lib/catalog.ts`; timeout 5 detik per probe, gagal -> daftar kosong).
3. `node main.ts --list` mencetak katalog sebagai satu baris JSON dan
   keluar 0. Tanpa `--list`, server menyala di port acak (`0.0.0.0`),
   mendaftarkan `{protocol_version: 1, port, catalog}` ke
   `POST $MAKEFUN_ENV_URL/register`, lalu polling `GET $MAKEFUN_ENV_URL/work`
   tiap 100 ms sampai `done: true`, menutup server, dan keluar 0.
   Register gagal -> keluar 1. Tanpa `MAKEFUN_ENV_URL` -> keluar 0.
4. Tiap request klien: nomor kedatangan (0, 1, 2, ...) dipetakan ke
   manifest (`r01`..`r20`); bila manifest tak punya entri itu, syarat
   ditebak dari body (gambar -> vision, `tools` tak kosong -> tools,
   `response_format` JSON -> json). Router menyusun draf untuk sisa
   trial, `main.ts` mencoba provider yang direncanakan; sukses ->
   jawaban di-render ke wire klien (`lib/translate.ts`, JSON atau SSE);
   gagal -> kegagalan dicatat, router me-re-draft, coba lagi. Bila tak
   ada kandidat sama sekali -> error 503 berformat wire klien
   (`"all providers refused"`).
5. `clientJsonMode` hanya dihitung untuk API OpenAI yang body-nya
   membawa `response_format` JSON; untuk API Anthropic selalu false.

## Struktur file

| File | Isi |
| --- | --- |
| `main.ts` | Entrypoint: HTTP server, register, polling `/work`, eksekutor planner (mutex per provider, `callUpstream` timeout 10 detik, batas body 32 MiB). |
| `lib/brief.ts` | Parser brief + tabel budget range + `ratingRow` / `meetsRequirements` / `isFreeModel` / `budgetRange` / `EMPTY_BRIEF`. |
| `lib/router.ts` | Global planner: kandidat legal + rating terbaik per provider, draf untuk sisa trial, pilihan berikutnya, pencatatan sukses/gagal, status mati provider. |
| `lib/catalog.ts` | Probe provider -> dokumen katalog, header auth per provider, daftar model gaya OpenAI/Anthropic. |
| `lib/translate.ts` | Konversi antar wire (OpenAI <-> Anthropic, ke Gemini), parser jawaban per wire + SSE, render jawaban JSON/SSE ke wire klien. |

Semua keputusan pemilihan tinggal di `router.ts`; `main.ts` hanya
mengeksekusi. Interface antar file yang dipakai `main.ts`
(`Router.legalCandidates/draftPlan/nextChoice/recordSuccess/recordFailure/isDead`,
`buildCatalog/authHeaders/openAIModelList/anthropicModelList`,
`openAIToAnthropic/anthropicToOpenAI/toGemini`,
`parseOpenAIJson/parseAnthropicJson/parseGeminiJson/parseProviderSse`,
`renderOpenAIJson/renderOpenAIStream/renderAnthropicJson/renderAnthropicStream`)
didokumentasikan di laporan penyerahan bila ada nama yang perlu
diselaraskan antar penulis file.

## Cara menjalankan

Node 22, TypeScript, stdlib saja — tanpa dependensi npm.

```sh
# Cetak katalog dan keluar (tanpa trial, tanpa brief):
node main.ts --list

# Dijalankan trial: env menyediakan MAKEFUN_PROMPT_FILE dan MAKEFUN_ENV_URL.
# Server menyala, mendaftar, melayani, lalu keluar 0 saat env bilang done.
node main.ts
```

## Grade lokal (dari kit)

Dari direktori kit (`free-model-aggregator-01-kit/`), dengan Docker:

```sh
# Satu trial practice (trial-0), live lalu replay check:
node cli/makefun.mjs grade-local --kit . --bot <repository> --driver docker

# Semua 3 trial practice:
node cli/makefun.mjs grade-local --kit . --bot <repository> --driver docker --instances 3

# Replay check saja dari sebuah cassette:
node cli/makefun.mjs grade-local --kit . --bot <repository> --driver docker --replay <cassette>
```

## Batasan

- Deterministik: tanpa `Date.now()`, `Math.random()`, atau wall clock.
  Setiap pilihan berasal dari brief + request + jawaban provider.
- Tanpa state persisten antar trial: semua state in-memory per proses,
  tanpa penulisan file.
- Satu panggilan dalam penerbangan per provider (mutex); panggilan
  kedua yang bersamaan ditolak provider dengan 429
  `concurrent_requests` dan tidak menghabiskan budget.
- Hanya model gratis yang boleh melayani request; tabel rating adalah
  otoritas soal gratis (`isFreeModel` di `lib/brief.ts`).
- Tie-break skor adalah jumlah panggilan provider: pada skor yang sama,
  entri dengan panggilan lebih sedikit menang. Karena itu planner
  menghemat panggilan — tidak ada probe atau retry spekulatif; setiap
  panggilan upstream adalah satu percobaan yang direncanakan, dan
  provider yang sudah dinyatakan mati tidak dipanggil lagi.

## Catatan operasional

- Bila `node main.ts --list` mencetak `{"providers":[]}` di luar trial,
  itu benar: tanpa `MAKEFUN_PROMPT_FILE` tidak ada brief untuk dibaca
  dan tanpa `MAKEFUN_ENV_URL` tidak ada provider untuk di-probe.
- Katalog yang dicetak `--list` harus sama byte demi byte dengan
  katalog di body register; keduanya dibangun oleh `buildCatalog` yang
  sama, jadi jangan mengedit salah satunya secara manual.
- Batas wall trial adalah 180 detik: server harus keluar 0 begitu
  `/work` menjawab `done: true`. Proses yang masih berjalan saat wall
  habis disetop dan mendapat 0 di semua kolom.
