# LocalHost Router (Team: LocalHost DC)

## Elevator Pitch / Tagline

A production-ready, ultra-fast Hybrid AI Router that orchestrates Google's tiny FunctionGemma-270M model and Gemini 2.5 Flash Lite to achieve 99% function-calling accuracy under 550ms.

## What it does (The Core Problem)

Running large LLMs in the cloud for simple tool-calling (like setting alarms or checking the weather) is too slow and expensive. However, running tiny 270M parameter models entirely on-device leads to hallucinations, wrong arguments, and failures on complex multi-intent requests.

We built a **proactive 3-Tier Hybrid Router**. Instead of blindly sending every query to the weak local model, our router scores the linguistic difficulty of a user's prompt *before* inference. It guarantees fast, free local execution for easy tasks, and seamlessly falls back to the cloud for complex multi-tool orchestration—all while masking the local model's hallucinations from the user.

## How we built it (The Architecture)

We engineered a 5-step pipeline that pushed the baseline score from ~50% to **80.9%** (with an incredible **F1 Accuracy of 0.99**), keeping 70% of operations entirely on-device and dropping average latency to 548ms.

1. **Pre-Routing Intelligence (`_compute_difficulty`):** We built a lexical analyzer that scores a prompt from 0.0 to 1.0 based on tool familiarity, multi-intent tracking, and keyword trapping.
    * **Tier 1 (Easy):** Handled purely on-device.
    * **Tier 2 (Medium):** Handled on-device, but audited by a semantic gate.
    * **Tier 3 (Hard):** Bypasses the device entirely to save 300ms, routing straight to Cloud.
2. **The Semantic Validation Gate:** We built a lexical firewall. If the weak local model hallucinates (e.g., the user says "Play jazz" and the model selects `set_alarm`), our gate detects the keyword mismatch, kills the local execution, and rescues the call via the cloud.
3. **Argument Extraction (The Hallucination Fix):** The 270M model fundamentally failed at parsing natural language numbers into JSON integers. We built a deterministic NLP parser that intercepts the model's broken JSON payload and overwrites it with safe, accurately extracted integers directly from the user's text.
4. **Multi-Intent Cloud Merging:** When a user asked for multiple things ("Set a timer *and* send a message"), the local model would often only get one right. Instead of throwing out the valid local call and wasting latency, our router **merges** them. We keep the valid local call, ask Gemini to fulfill *only* the missing one, and combine the payloads.
5. **Aggressive Latency Slicing:** To hit the ultra-low latency benchmark (capped at 500ms), we wrapped the `google.genai` client in a Singleton cache to avoid TLS handshake taxes (saving ~100ms per call), downgraded to `gemini-2.5-flash-lite`, and dropped the on-device `max_tokens` to speed up local loops.

## Challenges we ran into

The biggest challenge was the structural limitation of the `FunctionGemma-270m` model.

1. **Prompt Brittleness:** We tried advanced prompt engineering (Few-Shot Prompting, Dynamic Schema Enrichment, Strict JSON constraints). We discovered mathematically that injecting *any* system constraints into this specific 270M model actually degraded its baseline accuracy. It operates best as a pure zero-shot engine, forcing us to build the logic entirely into the Python orchestration layer instead of the prompt.
2. **Leaderboard Regex:** We noticed top-ranking teams were hitting 16ms latencies by bypassing the AI model entirely and explicitly parsing exact benchmark query strings. We chose instead to build a legitimate, production-ready AI hybrid router that generalizes beyond the scope of this hackathon's hidden evals.

## Accomplishments that we're proud of

* Achieving a near-perfect **0.99 F1 Accuracy Score** entirely algorithmically.
* Spiking our **Hard-Difficulty F1 from 0.50 to 0.97** by inventing the "Multi-Intent Gap Filling" merging algorithm.
* Building a truly robust **Semantic Validation Gate** that catches AI hallucinations before they are executed.
* Successfully deploying a 270M parameter model to handle 70% of user traffic, cleanly abstracting away its inherent mathematical weaknesses.

## What's next for LocalHost Router

Our next step is integrating `cactus_transcribe` (Whisper-small) to wrap our 3-Tier Router into a low-latency, fully functional Voice-to-Action terminal application that executes real actions on the device.
