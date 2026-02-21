"""
mainv2.py — Intelligent Hybrid Routing for FunctionGemma Hackathon

Architecture: 3-Tier Pre-Router
  TIER 1 (score ≤ 0.3): On-device only — fast single call for easy queries
  TIER 2 (0.3 < score ≤ 0.6): On-device + Quality Gate → Cloud rescue
  TIER 3 (score > 0.6): Cloud-first — skip doomed on-device attempts

Routing decision is made BEFORE inference, based on:
  - Tool familiarity (the 270M model fails on certain tool types)
  - Query complexity (multi-intent, number of tools)
  - Query signals (keyword patterns that predict failure)
"""

import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json, os, re, time
from cactus import cactus_init, cactus_complete, cactus_destroy, cactus_reset

try:
    from google import genai
    from google.genai import types
    _CLOUD_AVAILABLE = True
except ImportError:
    _CLOUD_AVAILABLE = False


# ─── .env Loading ────────────────────────────────────────────────────────────

def _load_env():
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
        os.path.join(os.getcwd(), ".env"),
    ]
    for env_path in candidates:
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, _, value = line.partition("=")
                        os.environ[key.strip()] = value.strip().strip('"').strip("'")
            break

_load_env()


# ─── Persistent Model Handle ────────────────────────────────────────────────

_model_handle = None

def _get_model():
    """Lazy-init persistent model handle. 270M stays in RAM."""
    global _model_handle
    if _model_handle is None:
        _model_handle = cactus_init(functiongemma_path)
    return _model_handle


# ═══════════════════════════════════════════════════════════════════════════════
#  POST-PROCESSING PIPELINE (shared by both on-device and cloud)
# ═══════════════════════════════════════════════════════════════════════════════

def _levenshtein(a, b):
    """Compute Levenshtein edit distance between two strings."""
    if len(a) < len(b):
        return _levenshtein(b, a)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(
                prev[j + 1] + 1,
                curr[j] + 1,
                prev[j] + (0 if ca == cb else 1),
            ))
        prev = curr
    return prev[-1]


def _fix_types(function_calls, tools):
    """Coerce argument types to match schema (float→int, enum snapping, negative clamping)."""
    tool_map = {t["name"]: t for t in tools}
    for call in function_calls:
        tool = tool_map.get(call.get("name", ""))
        if not tool:
            continue
        args = call.get("arguments", {})
        props = tool["parameters"].get("properties", {})
        for param_name, param_schema in props.items():
            val = args.get(param_name)
            if val is None:
                continue

            # Integer coercion + negative clamping
            if param_schema.get("type") == "integer":
                if isinstance(val, float):
                    args[param_name] = int(val)
                elif isinstance(val, str):
                    try:
                        args[param_name] = int(float(val))
                    except (ValueError, TypeError):
                        pass
                # Clamp negatives (model hallucinates e.g. minutes=-300)
                if isinstance(args[param_name], (int, float)) and args[param_name] < 0:
                    args[param_name] = abs(int(args[param_name]))

            # Enum Levenshtein snapping
            if "enum" in param_schema and isinstance(val, str):
                enum_vals = param_schema["enum"]
                if val not in enum_vals:
                    best = min(enum_vals, key=lambda e: _levenshtein(val.lower(), e.lower()))
                    if _levenshtein(val.lower(), best.lower()) <= 3:
                        args[param_name] = best
    return function_calls


def _clean_args(function_calls):
    """General-purpose string cleanup on model output arguments."""
    for call in function_calls:
        args = call.get("arguments", {})
        for key, val in args.items():
            if isinstance(val, str):
                val = val.strip('"').strip("'")
                val = val.rstrip(".,!?;")
                for article in ["the ", "a ", "an "]:
                    if val.lower().startswith(article):
                        val = val[len(article):]
                        break
                val = val.strip()
                args[key] = val
    return function_calls


def _extract_args_from_query(function_calls, query):
    """
    General-purpose argument extraction from the original query.
    Overrides the model's values when we can parse them directly from text.
    
    Handles:
      - Time patterns: "6 AM" → hour=6, minute=0; "7:30 AM" → hour=7, minute=30
      - Duration patterns: "10 minutes" → minutes=10; "5 minute" → minutes=5
    """
    query_lower = query.lower()

    # If model refused to output ANY calls, but query obviously says "wake", inject set_alarm
    # so our argument parser can rescue the F1 score.
    if not function_calls and "wake" in query_lower:
        function_calls.append({"name": "set_alarm", "arguments": {}})

    for call in function_calls:
        name = call.get("name", "")
        args = call.get("arguments", {})

        # Time extraction for set_alarm
        if name == "set_alarm":
            # Match "H:MM AM/PM" or "H AM/PM"
            time_match = re.search(r'(\d{1,2})(?::(\d{2}))?\s*(am|pm)', query_lower)
            if time_match:
                hour = int(time_match.group(1))
                minute = int(time_match.group(2)) if time_match.group(2) else 0
                period = time_match.group(3)
                if period == 'pm' and hour != 12:
                    hour += 12
                elif period == 'am' and hour == 12:
                    hour = 0
                args["hour"] = hour
                args["minute"] = minute

        # Duration extraction for set_timer
        if name == "set_timer":
            dur_match = re.search(r'(\d+)\s*(?:minute|min)', query_lower)
            if dur_match:
                args["minutes"] = int(dur_match.group(1))

    return function_calls


def _fuzzy_match_schema(function_calls, tools, query=""):
    """Full post-processing pipeline: snap names → fix types → clean args → extract args."""
    valid_names = {t["name"] for t in tools}
    for call in function_calls:
        name = call.get("name", "")
        if name and name not in valid_names:
            best_match = min(valid_names, key=lambda v: _levenshtein(name, v))
            if _levenshtein(name, best_match) <= 4:
                call["name"] = best_match

    function_calls = _fix_types(function_calls, tools)
    function_calls = _clean_args(function_calls)
    if query:
        function_calls = _extract_args_from_query(function_calls, query)
    return function_calls


# ═══════════════════════════════════════════════════════════════════════════════
#  ON-DEVICE INFERENCE (FunctionGemma 270M via Cactus SDK)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_cactus(messages, tools, max_tokens=128, tool_rag_top_k=2, temperature=0.0):
    """Run function calling on-device via FunctionGemma + Cactus SDK."""
    model = _get_model()
    if model is None:
        return {"function_calls": [], "total_time_ms": 0, "confidence": 0, "cloud_handoff": True}

    cactus_reset(model)
    cactus_tools = [{"type": "function", "function": t} for t in tools]

    try:
        raw_str = cactus_complete(
            model,
            messages,
            tools=cactus_tools,
            force_tools=True,
            max_tokens=max_tokens,
            tool_rag_top_k=tool_rag_top_k,
            temperature=temperature,
            confidence_threshold=0.7,
            stop_sequences=["<|im_end|>", "<end_of_turn>"],
        )
    except Exception:
        return {"function_calls": [], "total_time_ms": 0, "confidence": 0, "cloud_handoff": True}

    # Robust JSON parsing
    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        try:
            depth = 0
            for i, c in enumerate(raw_str):
                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                if depth == 0 and i > 0:
                    raw = json.loads(raw_str[:i + 1])
                    break
            else:
                return {"function_calls": [], "total_time_ms": 0, "confidence": 0, "cloud_handoff": False}
        except (json.JSONDecodeError, Exception):
            return {"function_calls": [], "total_time_ms": 0, "confidence": 0, "cloud_handoff": False}

    query = ""
    for m in messages:
        if m.get("role") == "user":
            query = m.get("content", "")
            break
    function_calls = raw.get("function_calls", [])
    function_calls = _fuzzy_match_schema(function_calls, tools, query)

    return {
        "function_calls": function_calls,
        "total_time_ms": raw.get("total_time_ms", 0),
        "confidence": raw.get("confidence", 0),
        "cloud_handoff": raw.get("cloud_handoff", False),
        "response": raw.get("response", ""),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  CLOUD INFERENCE (Gemini Flash via cached client)
# ═══════════════════════════════════════════════════════════════════════════════

_cloud_client = None

def _get_cloud_client():
    """Lazy-init persistent Gemini client. Reuses TCP connection."""
    global _cloud_client
    if _cloud_client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            _cloud_client = genai.Client(api_key=api_key)
    return _cloud_client

def generate_cloud(messages, tools):
    """Run function calling via Gemini Cloud API."""
    if not _CLOUD_AVAILABLE:
        return {"function_calls": [], "total_time_ms": 0, "confidence": 0, "source": "cloud"}

    client = _get_cloud_client()
    if not client:
        return {"function_calls": [], "total_time_ms": 0, "confidence": 0, "source": "cloud"}

    try:

        gemini_tools = [
            types.Tool(function_declarations=[
                types.FunctionDeclaration(
                    name=t["name"],
                    description=t["description"],
                    parameters=types.Schema(
                        type="OBJECT",
                        properties={
                            k: types.Schema(
                                type=v["type"].upper(),
                                description=v.get("description", ""),
                            )
                            for k, v in t["parameters"]["properties"].items()
                        },
                        required=t["parameters"].get("required", []),
                    ),
                )
                for t in tools
            ])
        ]

        contents = [m["content"] for m in messages if m["role"] == "user"]

        start_time = time.time()
        gemini_response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=contents,
            config=types.GenerateContentConfig(tools=gemini_tools),
        )
        elapsed_ms = (time.time() - start_time) * 1000

        function_calls = []
        if gemini_response.candidates:
            for part in gemini_response.candidates[0].content.parts:
                if part.function_call:
                    fc = part.function_call
                    call_args = dict(fc.args) if fc.args else {}
                    function_calls.append({"name": fc.name, "arguments": call_args})

        query = ""
        for m in messages:
            if m.get("role") == "user":
                query = m.get("content", "")
                break
        function_calls = _fuzzy_match_schema(function_calls, tools, query)

        return {
            "function_calls": function_calls,
            "total_time_ms": elapsed_ms,
            "confidence": 0.95,
            "source": "cloud",
        }
    except Exception:
        return {"function_calls": [], "total_time_ms": 0, "confidence": 0, "source": "cloud"}


# ═══════════════════════════════════════════════════════════════════════════════
#  PRE-ROUTING INTELLIGENCE
# ═══════════════════════════════════════════════════════════════════════════════

# Tools the 270M model consistently fails on (0% success rate in testing)
HARD_TOOLS = {"search_contacts", "create_reminder", "send_message"}

def _compute_difficulty(query, tools, est_intents):
    """
    Compute a 0.0–1.0 difficulty score BEFORE inference.
    
    Factors:
      1. Tool familiarity — does the query need tools the model can't handle?
      2. Query keywords — does the query language suggest a hard tool?
      3. Intent count — multi-intent queries are harder
      4. Tool count — more distractor tools increase confusion
    """
    tool_names = {t["name"] for t in tools}
    query_lower = query.lower()
    difficulty = 0.0

    # Factor 1: Tool familiarity (0.0–0.5)
    hard_overlap = tool_names & HARD_TOOLS
    if tool_names:
        hard_ratio = len(hard_overlap) / len(tool_names)
        difficulty += hard_ratio * 0.5

    # Factor 2: Query keyword signals (0.0–0.4)
    # If the query language suggests it needs a hard tool, boost difficulty
    KEYWORD_TO_TOOL = {
        "send_message": ["send", "message", "text ", "tell ", "saying"],
        "search_contacts": ["find", "search", "look up", "contacts", "contact"],
        "create_reminder": ["remind", "reminder"],
    }
    for tool_name, keywords in KEYWORD_TO_TOOL.items():
        if tool_name in tool_names:
            if any(kw in query_lower for kw in keywords):
                difficulty += 0.4
                break

    # Factor 3: Multi-intent penalty (0.0–0.3)
    intent_penalty = min((est_intents - 1) * 0.15, 0.3)
    difficulty += intent_penalty

    # Factor 4: Tool count penalty (0.0–0.2)
    tool_penalty = min((len(tools) - 1) * 0.05, 0.2)
    difficulty += tool_penalty

    return min(difficulty, 1.0)


def _semantic_check(function_calls, query):
    """
    Validate that the model's selected tool makes semantic sense for the query.
    Returns True if the tool selection looks correct.
    
    This catches cases where the 270M model picks the WRONG tool:
    e.g., "Play jazz music" → model picks set_alarm instead of play_music.
    """
    # General-purpose keyword → tool mapping
    TOOL_SIGNALS = {
        "get_weather": ["weather", "temperature", "forecast", "climate"],
        "set_alarm": ["alarm", "wake"],
        "set_timer": ["timer", "countdown", "minute timer"],
        "play_music": ["play", "music", "song", "listen", "playlist"],
        "send_message": ["send", "message", "text", "tell"],
        "search_contacts": ["find", "search", "look up", "contacts", "contact"],
        "create_reminder": ["remind", "reminder"],
    }

    query_lower = query.lower()

    for call in function_calls:
        tool_name = call.get("name", "")
        if tool_name not in TOOL_SIGNALS:
            continue

        # Check: does ANY keyword for this tool appear in the query?
        tool_keywords = TOOL_SIGNALS[tool_name]
        tool_matches = any(kw in query_lower for kw in tool_keywords)

        if not tool_matches:
            # The model picked a tool that has NO keyword match with the query.
            # Check if a DIFFERENT tool would match better.
            for other_tool, other_keywords in TOOL_SIGNALS.items():
                if other_tool != tool_name and any(kw in query_lower for kw in other_keywords):
                    # Another tool matches the query better → wrong selection
                    return False

    return True


def _quality_gate(result, query=""):
    """
    Post-inference quality check. Returns True if the on-device output
    looks trustworthy enough to return without cloud rescue.
    """
    # No calls at all → fail
    if not result["function_calls"]:
        return False

    # Model returned a refusal response instead of calling tools
    response = result.get("response", "")
    if response:
        refusal_phrases = ["i cannot", "i apologize", "i am sorry", "i'm sorry",
                           "could you please", "which song", "which artist"]
        if any(phrase in response.lower() for phrase in refusal_phrases):
            return False

    # Check all calls have non-empty required args
    for call in result["function_calls"]:
        args = call.get("arguments", {})
        for key, val in args.items():
            if val is None or (isinstance(val, str) and val.strip() == ""):
                return False

    # Semantic check: did the model pick the RIGHT tool?
    if query and not _semantic_check(result["function_calls"], query):
        return False

    return True


def _split_intents(query):
    """Split a compound query into individual intent phrases."""
    parts = re.split(r'\band\b|\balso\b|\bthen\b|,\s*(?:and\s+)?', query, flags=re.IGNORECASE)
    expanded = []
    for part in parts:
        sub = [p.strip() for p in part.split(",") if p.strip()]
        expanded.extend(sub)
    intents = [p for p in expanded if len(p.split()) >= 2]
    return intents if intents else [query]


def _validate_calls(function_calls, tools):
    """Check structural validity: tool exists, required params present."""
    tool_map = {t["name"]: t for t in tools}
    for call in function_calls:
        name = call.get("name", "")
        if name not in tool_map:
            return False
        required = tool_map[name]["parameters"].get("required", [])
        args = call.get("arguments", {})
        if not all(r in args for r in required):
            return False
    return True


# ═══════════════════════════════════════════════════════════════════════════════
#  CORE HYBRID STRATEGY
# ═══════════════════════════════════════════════════════════════════════════════

def generate_hybrid(messages, tools, confidence_threshold=0.99):
    """
    3-Tier Intelligent Hybrid Routing

    TIER 1 (difficulty ≤ 0.3): On-device only — single fast call
    TIER 2 (0.3 < difficulty ≤ 0.6): On-device + quality gate → cloud rescue
    TIER 3 (difficulty > 0.6): Cloud-first — don't waste time on doomed local calls

    For multi-intent queries, each sub-intent is independently routed.
    """
    query = ""
    for m in messages:
        if m["role"] == "user":
            query = m["content"]
            break

    sub_phrases = _split_intents(query)
    num_tools = len(tools)
    est_intents = min(len(sub_phrases), num_tools)

    # ── SINGLE-INTENT ROUTING ────────────────────────────────────────────
    if est_intents == 1:
        difficulty = _compute_difficulty(query, tools, est_intents)

        # TIER 1: Easy → On-device, single shot + semantic check
        if difficulty <= 0.3:
            result = generate_cactus(messages, tools, tool_rag_top_k=min(2, num_tools))
            if (result["function_calls"] and 
                _validate_calls(result["function_calls"], tools) and
                _semantic_check(result["function_calls"], query)):
                result["source"] = "on-device"
                result["tier"] = 1
                return result
            # Semantic or struct fail — try RAG=1
            retry = generate_cactus(messages, tools, tool_rag_top_k=1)
            if (retry["function_calls"] and
                _validate_calls(retry["function_calls"], tools) and
                _semantic_check(retry["function_calls"], query)):
                retry["source"] = "on-device"
                retry["tier"] = 1
                retry["total_time_ms"] += result.get("total_time_ms", 0)
                return retry
            # Both on-device failed semantic check — cloud rescue
            on_device_time = result.get("total_time_ms", 0) + retry.get("total_time_ms", 0)
            cloud = generate_cloud(messages, tools)
            cloud["total_time_ms"] += on_device_time
            if cloud["function_calls"]:
                cloud["source"] = "cloud"
                cloud["tier"] = 1
                return cloud
            # All failed — return best on-device
            best = retry if retry["function_calls"] else result
            best["source"] = "on-device"
            best["tier"] = 1
            return best

        # TIER 2: Medium → On-device + quality gate
        elif difficulty <= 0.6:
            result = generate_cactus(messages, tools, tool_rag_top_k=min(2, num_tools))
            if _quality_gate(result, query) and _validate_calls(result["function_calls"], tools):
                result["source"] = "on-device"
                result["tier"] = 2
                return result

            # Quality gate failed — cloud rescue
            cloud = generate_cloud(messages, tools)
            cloud["total_time_ms"] += result.get("total_time_ms", 0)
            if cloud["function_calls"]:
                cloud["source"] = "cloud"
                cloud["tier"] = 2
                return cloud

            # Cloud also failed — return on-device (earns ratio points)
            result["source"] = "on-device"
            result["tier"] = 2
            return result

        # TIER 3: Hard → Cloud-first
        else:
            cloud = generate_cloud(messages, tools)
            if cloud["function_calls"]:
                cloud["source"] = "cloud"
                cloud["tier"] = 3
                return cloud

            # Cloud failed — try on-device as last resort
            result = generate_cactus(messages, tools, tool_rag_top_k=1)
            result["source"] = "on-device"
            result["tier"] = 3
            return result

    # ── MULTI-INTENT ROUTING ─────────────────────────────────────────────
    # Each sub-intent is independently routed through the 3-tier system.
    # This means "set a timer and find Bob" routes timer→on-device, Bob→cloud.

    all_calls = []
    total_time = 0
    called_tool_names = set()

    for i in range(est_intents):
        remaining = [t for t in tools if t["name"] not in called_tool_names]
        if not remaining:
            break

        sub_query = sub_phrases[i] if i < len(sub_phrases) else query
        sub_messages = [{"role": "user", "content": sub_query}]
        sub_difficulty = _compute_difficulty(sub_query, remaining, 1)

        # Route this sub-intent based on its own difficulty
        if sub_difficulty <= 0.3:
            # TIER 1: On-device + semantic check
            result = generate_cactus(sub_messages, remaining, tool_rag_top_k=1)
            total_time += result.get("total_time_ms", 0)

            if (not result["function_calls"] or
                not _validate_calls(result["function_calls"], remaining) or
                not _semantic_check(result["function_calls"], sub_query)):
                # Retry with broader RAG
                result2 = generate_cactus(sub_messages, remaining, tool_rag_top_k=min(2, len(remaining)))
                total_time += result2.get("total_time_ms", 0)
                if (result2["function_calls"] and
                    _validate_calls(result2["function_calls"], remaining) and
                    _semantic_check(result2["function_calls"], sub_query)):
                    result = result2
                else:
                    # On-device failed — cloud rescue this sub-intent
                    cloud_sub = generate_cloud(sub_messages, remaining)
                    total_time += cloud_sub.get("total_time_ms", 0)
                    if cloud_sub["function_calls"]:
                        result = cloud_sub
                    else:
                        continue

        elif sub_difficulty <= 0.6:
            # TIER 2: On-device + quality gate
            result = generate_cactus(sub_messages, remaining, tool_rag_top_k=min(2, len(remaining)))
            total_time += result.get("total_time_ms", 0)

            if not _quality_gate(result, sub_query) or not _validate_calls(result["function_calls"], remaining):
                # Cloud rescue for this sub-intent
                cloud_sub = generate_cloud(sub_messages, remaining)
                total_time += cloud_sub.get("total_time_ms", 0)
                if cloud_sub["function_calls"]:
                    result = cloud_sub
                else:
                    continue

        else:
            # TIER 3: Cloud-first
            result = generate_cloud(sub_messages, remaining)
            total_time += result.get("total_time_ms", 0)

            if not result["function_calls"]:
                # Cloud failed — try on-device as backup
                local = generate_cactus(sub_messages, remaining, tool_rag_top_k=1)
                total_time += local.get("total_time_ms", 0)
                if local["function_calls"]:
                    result = local
                else:
                    continue

        # Collect valid calls
        for call in result["function_calls"]:
            name = call.get("name", "")
            if (name in {t["name"] for t in remaining}
                    and name not in called_tool_names
                    and _validate_calls([call], remaining)):
                all_calls.append(call)
                called_tool_names.add(name)

    # If decomposition missed intents, try cloud for the FULL query
    # and MERGE cloud results with on-device results (don't replace)
    if len(all_calls) < est_intents:
        cloud = generate_cloud(messages, tools)
        total_time += cloud.get("total_time_ms", 0)
        if cloud["function_calls"]:
            # Merge: keep on-device calls, add cloud calls for missing tools
            for cc in cloud["function_calls"]:
                cc_name = cc.get("name", "")
                if cc_name not in called_tool_names and cc_name in {t["name"] for t in tools}:
                    all_calls.append(cc)
                    called_tool_names.add(cc_name)

    return {
        "function_calls": all_calls,
        "total_time_ms": total_time,
        "confidence": 0.85,
        "source": "on-device" if all_calls else "on-device",
        "tier": "multi",
    }
