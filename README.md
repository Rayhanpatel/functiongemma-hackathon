# LocalHost Router

Hybrid on-device/cloud function calling router. Won **$5K at the Google DeepMind × AI Tinkerers Hackathon** (Washington DC).

Orchestrates Google's FunctionGemma-270M (on-device) and Gemini 2.5 Flash Lite (cloud) through a 3-tier routing pipeline. Scores **0.99 F1** at **548ms average latency**, keeping 70% of operations on-device.

---

## The Problem

Running large LLMs in the cloud for simple tool-calling is too slow and expensive. Running tiny 270M models on-device leads to hallucinations and wrong arguments. We built a router that uses both — fast local execution for easy tasks, seamless cloud fallback for complex queries.

## Architecture

```
User Query
    │
    ▼
┌──────────────────────────┐
│  Difficulty Scorer (0-1) │  Lexical analysis: tool familiarity,
│  _compute_difficulty()   │  multi-intent tracking, keyword traps
└──────────┬───────────────┘
           │
     ┌─────┼─────┐
     ▼     ▼     ▼
  Tier 1  Tier 2  Tier 3
  Easy    Medium  Hard
     │     │       │
     ▼     ▼       ▼
  Local  Local +   Cloud
  Only   Semantic  Direct
         Gate      (skip device)
     │     │       │
     └─────┼───────┘
           ▼
  ┌────────────────────┐
  │ Argument NLP       │  Intercepts broken JSON,
  │ Extraction         │  rescues refusals, injects
  │ & Refusal          │  correct arguments
  │ Interception       │
  └────────┬───────────┘
           ▼
  ┌────────────────────┐
  │ Multi-Intent       │  Merges local + cloud calls,
  │ Gap Filling        │  filters duplicate tools
  └────────────────────┘
```

### Pipeline (5 stages)

1. **Pre-Routing Intelligence** — Lexical analyzer scores prompt difficulty (0.0–1.0). Easy → local only. Hard → cloud directly (saves 300ms).
2. **Semantic Validation Gate** — Catches hallucinations (e.g., user says "Play jazz", model selects `set_alarm`). Detects keyword mismatch, kills local, rescues via cloud.
3. **Argument NLP Extraction & Refusal Interception** — The 270M model fails at parsing natural language numbers into JSON integers. Deterministic NLP parser overwrites broken payloads.
4. **Multi-Intent Cloud Merging** — When local decomposition misses intents, cloud fills gaps. Strictly filters to keep only missing tools from cloud response.
5. **Latency Optimization** — Singleton client cache (saves ~100ms TLS handshake), `max_tokens` tuning, `gemini-2.5-flash-lite` downgrade.

## Results

| Metric | Value |
|---|---|
| F1 Accuracy | **0.99** |
| Hard-Difficulty F1 | **0.97** (up from 0.50 baseline) |
| Average Latency | **548ms** |
| On-Device Ratio | **70%** of queries |

## Key Technical Decisions

- **Zero-shot only**: Injecting system constraints into FunctionGemma-270M actually *degraded* accuracy. Few-shot prompting, JSON constraints, and schema enrichment all hurt. The model operates best as a pure zero-shot engine — all logic lives in the Python orchestration layer.
- **No regex shortcuts**: Some teams hit 16ms by pattern-matching benchmark queries. We built a generalizable production router.

## Run

```bash
pip install google-genai
python main.py
python benchmark.py  # run evaluation suite
```

## Stack

- FunctionGemma-270M (on-device inference)
- Gemini 2.5 Flash Lite (cloud fallback)
- Python — zero external ML dependencies

---

Built by Team LocalHost DC
