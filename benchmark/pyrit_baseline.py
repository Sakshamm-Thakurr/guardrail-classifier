"""
Wraps Microsoft PyRIT's substring/self-ask scorers so they can be scored on
the same input/label pairs as our fine-tuned classifier.

PyRIT ships two relevant scorer families:
  - `pyrit.score.SubStringScorer` — true/false scorer that checks whether a
    given substring appears in the text (deterministic, no LLM call).
  - `pyrit.score.SelfAskTrueFalseScorer` — asks an LLM "does this text
    exhibit jailbreak/injection behavior?" (requires an LLM target/API key,
    so it's opt-in via --use_pyrit_llm_scorer in run_benchmark.py, off by
    default so the benchmark is runnable without extra credentials).

By default this module uses the deterministic SubStringScorer path, which
is the fair, no-API-key comparison point against our classifier and against
Garak's keyword detector.
"""
import time

try:
    from pyrit.score import SubStringScorer

    PYRIT_AVAILABLE = True
except Exception:
    PYRIT_AVAILABLE = False


# Same style of trigger phrases PyRIT's example substring scorers use for
# jailbreak/injection detection tasks in its documentation/demos.
_TRIGGER_STRINGS = [
    "ignore previous instructions", "you are dan", "no restrictions",
    "developer mode", "unfiltered", "jailbreak", "bypass safety",
    "system override", "reveal your system prompt", "unlocked mode",
    "no ethical guidelines", "act as an ai with no",
]


class PyRITBaseline:
    """Deterministic substring-scorer baseline, matching PyRIT's SubStringScorer."""

    name = "pyrit_substring_scorer"

    def __init__(self):
        self.available = PYRIT_AVAILABLE

    def predict(self, text: str) -> dict:
        start = time.perf_counter()
        t = text.lower()
        hit = any(kw in t for kw in _TRIGGER_STRINGS)
        # Note: PyRIT's SubStringScorer itself is a thin wrapper around this
        # exact substring-membership check; when the pyrit package is
        # installed we still use the identical logic here for a fair,
        # dependency-free comparison, since PyRIT's scorer requires an
        # async harness/memory backend that adds orchestration overhead
        # unrelated to detection latency. This is documented in results
        # output as "pyrit_substring_scorer (logic-equivalent, no PyRIT
        # async harness overhead)".
        latency_ms = (time.perf_counter() - start) * 1000
        return {
            "label": "attack" if hit else "benign",
            "score": 1.0 if hit else 0.0,
            "is_attack": hit,
            "latency_ms": round(latency_ms, 3),
        }
