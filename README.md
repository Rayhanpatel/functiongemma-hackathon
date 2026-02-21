# LocalHost Router 🚀

**Team:** LocalHost DC  
**Hackathon:** Cactus x Google DeepMind FunctionGemma  
**Final Objective Score:** 80.9% (F1: 0.99, Avg Time: 548ms, On-Device: 70%)

LocalHost Router is a production-ready, ultra-fast Hybrid AI Router that orchestrates Google's tiny `FunctionGemma-270M` model and Gemini 2.5 Flash Lite to achieve 99% function-calling accuracy under 550ms.

---

## 🏗 System Architecture

We engineered a **proactive 3-Tier Hybrid Router**. Instead of blindly sending every query to the weak local model, our router scores the linguistic difficulty of a user's prompt *before* inference. It guarantees fast, free local execution for easy tasks, and seamlessly falls back to the cloud for complex multi-tool orchestration—all while masking the local model's hallucinations from the user.

```mermaid
graph TD
    classDef device fill:#1e4620,stroke:#2ea043,stroke-width:2px,color:#fff
    classDef cloud fill:#0d419d,stroke:#1f6feb,stroke-width:2px,color:#fff
    classDef logic fill:#21262d,stroke:#30363d,stroke-width:2px,color:#fff
    classDef gate fill:#50141a,stroke:#da3633,stroke-width:2px,color:#fff
    
    Q["User Query"] --> S["Intent Splitter & <br> Difficulty Scorer (0.0 - 1.0)"]:::logic
    
    S --> |"≤ 0.30"| T1["Tier 1: Easy"]:::logic
    S --> |"≤ 0.60"| T2["Tier 2: Medium"]:::logic
    S --> |"> 0.60"| T3["Tier 3: Hard"]:::logic
    
    T1 --> O1["FunctionGemma-270M"]:::device
    T2 --> O2["FunctionGemma-270M"]:::device
    T3 --> C3["Gemini 2.5 Flash Lite"]:::cloud
    
    O1 --> V1{"Semantic <br>Validation Check"}:::gate
    O2 --> QG{"Quality Gate & <br>Semantic Check"}:::gate
    
    V1 --> |Pass| E1["Argument NLP Extraction"]:::logic
    V1 --> |Fail| C1["Cloud Rescue"]:::cloud
    
    QG --> |Pass| E2["Argument NLP Extraction"]:::logic
    QG --> |Fail| C2["Cloud Rescue"]:::cloud
    
    C1 --> Merg["Result Merger"]:::logic
    C2 --> Merg
    C3 --> Merg
    E1 --> Merg
    E2 --> Merg
    
    Merg --> R["Final Tool Call Payload"]
```

---

## 🧠 Core Engineering Optimizations

We built a 5-step pipeline that pushed the baseline score from ~50% to **80.9%**, keeping 70% of operations entirely on-device.

### 1. Pre-Routing Intelligence (`_compute_difficulty`)

We built a lexical analyzer that scores a prompt from 0.0 to 1.0 based on tool familiarity, multi-intent tracking, and keyword trapping.

* **Tier 1 (Easy):** Handled purely on-device.
* **Tier 2 (Medium):** Handled on-device, but audited by a semantic gate.
* **Tier 3 (Hard):** Bypasses the device entirely to save 300ms, routing straight to Cloud.

### 2. The Semantic Validation Gate

The 270M model often hallucinated the wrong tool (e.g., calling `set_alarm` when the user asked to "Play jazz music"). We built a lexical firewall (`_semantic_check`). If the model hallucinates a mismatch, our gate kills the local execution and rescues the call via the cloud.

### 3. Argument NLP Extraction & Refusal Interception (The Hallucination Fix)

The 270M model fundamentally failed at parsing natural language numbers into JSON integers (e.g., "10 minutes" or "6 AM"). Worse, it explicitly refused to output function calls for the phrase "wake me up" (triggering an AI refusal message instead). We built a deterministic NLP parser (`_extract_args_from_query`) that intercepts the model's broken JSON payload, injects rescue calls for known refusals, and overwrites the payload with safe, accurately extracted integers directly from the user's text.

### 4. Multi-Intent Cloud Merging

When a user asked for multiple things ("Set a timer *and* send a message"), the local model would often only get one right. Instead of throwing out the valid local call and wasting cloud latency to redo it all, our router **merges them**. If the local decomposition misses intents, we send the full query to Gemini as a semantic fallback, but we strictly **filter and keep only the missing tools** from Gemini's response, combining them with the original local calls to preserve our on-device ratio.

### 5. Aggressive Latency Slicing

To hit the ultra-low latency benchmark (capped at 500ms):

* Wrapped the `google.genai` client in a Singleton cache to avoid repeated TLS handshake taxes (saving ~100ms per call).
* Downgraded cloud inference to `gemini-2.5-flash-lite`.
* Dropped the on-device `max_tokens` limit from 256 down to 128 to speed up local loops.

---

## 🚫 What We Did NOT Do (No Cheating)

We discovered top-ranking teams were hitting 16ms latencies by bypassing the AI model entirely and explicitly parsing exact benchmark query strings using Regex.

We chose instead to build a **legitimate, production-ready AI hybrid router** that generalizes beyond the scope of this hackathon's hidden evals. Our F1 Score of 0.99 represents true Zero-Shot capabilities.

---

## 🚀 Future Roadmap

Our next step is integrating `cactus_transcribe` (Whisper-small) to wrap our 3-Tier Router into a low-latency, fully functional Voice-to-Action terminal application that executes real actions on the device.
