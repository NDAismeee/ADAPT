# openai_agents.py
# =========================================================
# OpenAI Generator Agent for ADAPT (Agentic Data Generation)
#
# Paper alignment:
#   - Generator is LLM-based and runs on-device (client-side)
#   - Implements a "Hard Prompt Pool" mechanism to simulate
#     regime-specific soft prompts (Cold-start, Exploration, Standard).
#
# Expected usage:
#   - client determines 'regime' based on signals (e.g., seq_len, diversity)
#   - calls generator.run(..., regime="exploration")
# =========================================================

from __future__ import annotations

import json
from typing import Dict, List, Any, Optional

from openai import OpenAI


# -----------------------------
# Shared OpenAI client
# -----------------------------
def get_openai_client() -> OpenAI:
    # Reads OPENAI_API_KEY from env
    return OpenAI()


# -----------------------------
# Helper: decide how many items to generate (cost-aware)
# -----------------------------
def decide_num_items(seq_len: int) -> int:
    """
    Decide number of synthetic items based on sequence length.
    Designed for min_seq_len >= 20 (your project assumption).
    """
    if seq_len < 50:
        return 5
    elif seq_len < 100:
        return 3
    elif seq_len < 150:
        return 3
    else:
        return 1


# -----------------------------
# Helper: robust JSON extraction
# -----------------------------
def _safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    """
    Tries to parse strict JSON. If the model wraps JSON with extra text,
    this will attempt to extract the first JSON object block.
    """
    text = (text or "").strip()
    if not text:
        return None

    # 1) Direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # 2) Try to extract the first {...} block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            return None

    return None


# =========================================================
# Generator Agent (LLM) -- ADAPT
# =========================================================
class OpenAIGeneratorAgent:
    """
    ADAPT generator with Prompt Pool Simulation.
    Supports multiple regimes: 'standard', 'cold_start', 'exploration'.
    """

    def __init__(self, model: str):
        self.model = model

    def _build_prompt(
        self, 
        regime: str, 
        user_profile: Dict[str, Any], 
        model_topk: List[int], 
        max_items: int
    ) -> str:
        """
        Selects the appropriate prompt template from the "pool" based on regime.
        """
        profile_json = json.dumps(user_profile, ensure_ascii=False)
        topk_json = json.dumps(model_topk, ensure_ascii=False)
        
        # Common context
        base_instruction = f"""
You are a generator agent for a federated sequential recommender system.
User profile (JSON):
{profile_json}
The recommender model currently predicts these top items:
{topk_json}
"""

        # --- Prompt Pool Logic ---
        
        if regime == "cold_start":
            # Regime 1: User has very little data.
            # Strategy: Suggest popular, foundational, or broad-appeal items to stabilize profile.
            return f"""{base_instruction}
Task:
- The user has a SHORT history (Cold Start).
- Generate up to {max_items} broadly appealing, popular, or foundational next-item IDs.
- Items should be "safe bets" likely to be interacted with by a new user in this domain.
- Do NOT repeat items in "recent_items" or "model_topk".
- Output STRICT JSON ONLY.

Schema:
{{ "items": [{{ "id": 123 }}, {{ "id": 456 }}] }}
""".strip()

        elif regime == "exploration":
            # Regime 2: User has repetitive history.
            # Strategy: Suggest diverse, serendipitous items to break echo chambers.
            return f"""{base_instruction}
Task:
- The user has a REPETITIVE history (Low Diversity).
- Generate up to {max_items} "serendipitous" next-item IDs.
- Choose items that are DIFFERENT from the user's usual history but still plausible.
- Avoid obvious next-items; aim for novel discovery.
- Do NOT repeat items in "recent_items" or "model_topk".
- Output STRICT JSON ONLY.

Schema:
{{ "items": [{{ "id": 123 }}, {{ "id": 456 }}] }}
""".strip()

        else:
            # Regime 3: Standard / Default.
            # Strategy: Complement model predictions with plausible alternatives.
            return f"""{base_instruction}
Task:
- Generate up to {max_items} ADDITIONAL plausible next-item IDs for this user.
- Prefer items that COMPLEMENT the model (plausible but under-explored).
- Do NOT repeat items in "recent_items".
- Do NOT repeat items already in the model's top predictions ("model_topk").
- Output STRICT JSON ONLY.

Schema:
{{ "items": [{{ "id": 123 }}, {{ "id": 456 }}] }}
""".strip()

    def run(
        self,
        user_profile: Dict[str, Any],
        *,
        regime: str = "standard",
        num_items: Optional[int] = None,
        hard_cap: Optional[int] = None,
    ) -> List[int]:
        """
        Args:
            user_profile: dict with recent_items, seq_len, etc.
            regime: 'standard', 'cold_start', or 'exploration' (Prompt Routing)
            num_items: total items count for safety check
            hard_cap: budget limit from server
        """
        # --- Validate profile ---
        if "recent_items" not in user_profile or "seq_len" not in user_profile:
            raise ValueError("user_profile must contain 'recent_items' and 'seq_len'")

        recent_items = user_profile.get("recent_items", [])
        seq_len = int(user_profile.get("seq_len", len(recent_items)))
        model_topk = user_profile.get("model_topk", [])

        # Cost-aware count
        max_items = decide_num_items(seq_len)
        if hard_cap is not None:
            max_items = min(max_items, int(hard_cap))
        max_items = max(0, max_items)

        if max_items == 0:
            return []

        # Build Prompt based on Regime
        prompt = self._build_prompt(regime, user_profile, model_topk, max_items)

        # Call API
        try:
            client = get_openai_client()
            resp = client.responses.create(model=self.model, input=prompt)
            text = (resp.output_text or "").strip()
        except Exception as e:
            print(f"[Generator] OpenAI API Error: {e}")
            return []

        # Parse
        data = _safe_json_loads(text)
        if not data or "items" not in data or not isinstance(data["items"], list):
            return []

        # Parse IDs + local guardrails
        seen = set()
        banned = set()
        for x in recent_items:
            try:
                banned.add(int(x))
            except Exception:
                pass
        for x in model_topk:
            try:
                banned.add(int(x))
            except Exception:
                pass

        out: List[int] = []
        for it in data["items"]:
            if len(out) >= max_items:
                break
            if not isinstance(it, dict) or "id" not in it:
                continue
            try:
                item_id = int(it["id"])
            except Exception:
                continue

            # optional range check
            if num_items is not None:
                if item_id < 1 or item_id > int(num_items):
                    continue

            # exclude repeats
            if item_id in banned or item_id in seen:
                continue

            seen.add(item_id)
            out.append(item_id)

        return out