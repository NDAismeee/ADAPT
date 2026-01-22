# trigger_agent.py
# =========================================================
# ADAPT Trigger Agent (server-side) - LOCAL LLM VERSION
#
# Paper alignment (Section 4.3):
#   - Input: behavioral signals s_{i,t} (Loss, Diversity, Alignment)
#   - Output: Generation Budget k_i^{t} (Integer >= 0)
#   - Implemented as a Local LLM-based policy (e.g., Qwen-0.5B)
#   - No access to raw data or item identities
#
# Changes:
#   - Returns INT (budget) instead of BOOL (trigger).
#   - Logic considers "Alignment" signal explicitly.
# =========================================================

from __future__ import annotations

import json
import torch
from typing import Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# -----------------------------
# Safe JSON parsing for BUDGET
# -----------------------------
def _safe_budget_parse(text: str) -> int:
    """
    Robust parse for LLM outputs:
      - Expected JSON: {"budget": 5}
      - Fallback: Look for integers in text.
      - Default: 0
    """
    if not text:
        return 0

    text = text.strip()
    
    # 1) Try JSON parse
    try:
        # Extract JSON block if wrapped
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            json_str = text[start : end + 1]
            data = json.loads(json_str)
            if isinstance(data, dict):
                # Check probable keys
                for k in ["budget", "k", "num_items", "count"]:
                    if k in data and isinstance(data[k], (int, float)):
                        return int(data[k])
    except Exception:
        pass

    # 2) Fallback: Regex or simple search for numbers
    # If the model just says "5", try to parse it.
    try:
        # Clean punctuation
        clean_text = ''.join(c if c.isdigit() else ' ' for c in text)
        numbers = [int(s) for s in clean_text.split() if s.isdigit()]
        if numbers:
            # Return the first number found (usually the budget)
            return numbers[0]
    except Exception:
        pass

    return 0


# =========================================================
# Trigger Agent (Local LLM)
# =========================================================
class TriggerAgent:
    """
    Local LLM-based controller for ADAPT.
    Allocates generation budget based on client signals.
    """

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.1,
        debug: bool = False,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.debug = debug
        self.pipe = None

        print(f"[TriggerAgent] Initializing local model: {model_name}...")
        
        try:
            device_map = "auto" if torch.cuda.is_available() else "cpu"
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=True
            )
            
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=20,
                do_sample=(temperature > 0),
                temperature=temperature,
            )
            print(f"[TriggerAgent] Model loaded successfully on {self.model.device}.")
            
        except Exception as e:
            print(f"[TriggerAgent] ❌ CRITICAL ERROR loading model: {e}")
            print("[TriggerAgent] Agent will default to BUDGET 0.")
            self.pipe = None

    def run(self, signals: Dict[str, Any]) -> int:
        """
        Decide generation budget k_i using local LLM.
        Returns: int (number of items to generate)
        """
        if self.pipe is None:
            return 0

        # Construct compact signal representation
        # Note: "align" will be computed in client/server and passed here
        compact_signals = {
            "loss": round(signals.get("performance", {}).get("train_loss", 0.0) or 0.0, 4),
            "short_hist": signals.get("performance", {}).get("is_short_history", False),
            "align": round(signals.get("update_quality", {}).get("alignment", 1.0), 4), # Default 1.0 (good)
            "diversity": signals.get("distribution_shift", {}).get("diversity_score", "medium") 
        }

        prompt = self._build_prompt(compact_signals)

        try:
            messages = [
                {"role": "system", "content": "You are a resource allocator for federated learning. Output strictly JSON."},
                {"role": "user", "content": prompt}
            ]
            
            prompt_formatted = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )

            outputs = self.pipe(prompt_formatted)
            generated_text = outputs[0]["generated_text"]
            
            response_text = generated_text[len(prompt_formatted):].strip()

            if self.debug:
                print(f"[TriggerAgent] Input: {compact_signals}")
                print(f"[TriggerAgent] Output: {repr(response_text)}")

            budget = _safe_budget_parse(response_text)
            
            # Sanity check: Cap budget to reasonable limits (e.g., max 10)
            budget = max(0, min(budget, 10))
            
            return budget

        except Exception as e:
            if self.debug:
                print(f"[TriggerAgent] Inference error: {e}")
            return 0

    def _build_prompt(self, signals: Dict[str, Any]) -> str:
        """
        Construct prompt for budget allocation.
        """
        return f"""
Analyze client signals:
{json.dumps(signals)}

Task: Allocate synthetic data budget (0 to 10 items).
Rules:
1. Low Alignment (< 0.5) OR High Loss (> 0.8) -> Need help -> Budget: 5-8
2. Short History (Cold Start) -> Need help -> Budget: 3-5
3. Good Alignment (> 0.8) AND Low Loss -> Stable -> Budget: 0
4. Otherwise -> Budget: 1-2

Output JSON only: {{"budget": <int>}}
""".strip()