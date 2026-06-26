# Ensemble Predictor — confidence-weighted LLM ensemble for the backtest engine
# File: ensemble_predictor.py
#
# This module is the bridge between the live LLM stack in `llm_stock_analyzer.py`
# and the backtesting engine in `advanced_backtest_analyzer.py`.
#
# It exposes a single, *synchronous*, *causal*, *cached* prediction call:
#
#     EnsemblePredictor.predict_day(stock_symbol, movement_type,
#                                   history_df, movement_column) -> (pred, confidence)
#
# where `history_df` contains data available *up to and including day t only*.
# The next-day label is NOT passed in, so no look-ahead can occur here.
#
# The aggregation implements Eq. (5) of the paper:
#
#       Y_hat_{t+1} = 1[ ( sum_i w_i * c_i * yhat_i ) / ( sum_i w_i * c_i ) >= tau ]
#
# with per-query confidences c_i returned by each model, static reliability
# weights w_i (equal by default, matching the released engine), and decision
# threshold tau (default 0.5).
#
# Because a full backtest issues one ensemble query per test day and the same
# (symbol, movement_type, feature-state) recurs across overlapping splits and
# rolling windows, every distinct prompt is cached to disk. This makes repeated
# runs cheap and makes the temperature-0.3 pipeline reproducible.

import os
import json
import hashlib
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from llm_stock_analyzer import LLMStockPredictor

logger = logging.getLogger(__name__)

# Default six-model ensemble (matches the paper, Sect. "LLM Selection").
DEFAULT_ENSEMBLE = [
    "gpt4o-mini",
    "gpt4o",
    "claude-haiku",
    "claude-sonnet",
    "gemini-flash",
    "gemini-pro",
]

# Static reliability weights w_i. The released engine sets these EQUAL, so the
# aggregate reduces to confidence-weighted voting (paper, Sect. 4.3). The
# higher-capability member of each provider pair is given a modestly larger
# prior here only if `use_tiered_weights=True` is requested; it is never tuned
# on test data.
TIERED_WEIGHTS = {
    "gpt4o-mini": 1.0, "gpt4o": 1.25,
    "claude-haiku": 1.0, "claude-sonnet": 1.25,
    "gemini-flash": 1.0, "gemini-pro": 1.25,
}


class EnsemblePredictor:
    """Confidence-weighted LLM ensemble with on-disk caching for backtesting."""

    def __init__(self,
                 data_directory: str = "data",
                 models: Optional[List[str]] = None,
                 tau: float = 0.5,
                 use_tiered_weights: bool = False,
                 cache_dir: str = "llm_cache",
                 offline_ok: bool = False):
        self.predictor = LLMStockPredictor(data_directory)
        self.models = models if models is not None else list(DEFAULT_ENSEMBLE)
        self.tau = tau
        self.use_tiered_weights = use_tiered_weights
        self.offline_ok = offline_ok

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self._mem_cache: Dict[str, Dict] = {}

        # Drop any model whose API key is missing, so a partial ensemble still
        # runs (e.g. Gemini-only on the free tier). This mirrors the README's
        # "at least one API key" guidance.
        available = []
        for m in self.models:
            cfg = self.predictor.model_configs.get(m)
            if cfg and self.predictor.api_keys.get(cfg["provider"]):
                available.append(m)
            else:
                logger.warning(f"Skipping {m}: no API key for its provider.")
        if available:
            self.models = available
        elif not self.offline_ok:
            logger.warning("No API keys found for any ensemble model. "
                           "Set offline_ok=True only for dry-run plumbing tests.")

    # ---- weighting -------------------------------------------------------
    def _weight(self, model_name: str) -> float:
        if self.use_tiered_weights:
            return TIERED_WEIGHTS.get(model_name, 1.0)
        return 1.0  # equal weights => confidence-weighted voting

    # ---- caching ---------------------------------------------------------
    def _prompt_key(self, stock_symbol: str, movement_type: str,
                    technical_data: Dict, movement_stats: Dict) -> str:
        """Stable hash of the exact prompt inputs => cache key."""
        prompt = self.predictor.create_movement_prediction_prompt(
            stock_symbol, movement_type, technical_data, movement_stats
        )
        payload = json.dumps(
            {"models": sorted(self.models), "prompt": prompt},
            sort_keys=True, default=str,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _load_cache(self, key: str) -> Optional[Dict]:
        if key in self._mem_cache:
            return self._mem_cache[key]
        p = self._cache_path(key)
        if p.exists():
            try:
                data = json.loads(p.read_text())
                self._mem_cache[key] = data
                return data
            except Exception:
                return None
        return None

    def _save_cache(self, key: str, data: Dict) -> None:
        self._mem_cache[key] = data
        try:
            self._cache_path(key).write_text(json.dumps(data))
        except Exception as e:
            logger.warning(f"Could not write cache {key}: {e}")

    # ---- core ensemble query --------------------------------------------
    async def _query_ensemble_async(self, stock_symbol: str, movement_type: str,
                                    prompt: str) -> List[Dict]:
        """Dispatch all models in parallel (asyncio.gather) for one prompt."""
        async def one(model_name):
            try:
                interface = self.predictor.initialize_llm(model_name)
                r = await interface.predict(prompt)
                return {
                    "model_name": model_name,
                    "prediction": int(r.get("prediction", 0)),
                    "confidence": float(r.get("confidence", 0.5)),
                }
            except Exception as e:
                logger.warning(f"{model_name} failed: {e}")
                # Neutral fallback (paper: prediction 0, confidence 0.5).
                return {"model_name": model_name, "prediction": 0, "confidence": 0.5}

        return await asyncio.gather(*[one(m) for m in self.models])

    def _get_member_predictions(self, stock_symbol: str, movement_type: str,
                                technical_data: Dict, movement_stats: Dict) -> List[Dict]:
        """Cached per-model predictions for one day's feature state."""
        key = self._prompt_key(stock_symbol, movement_type, technical_data, movement_stats)
        cached = self._load_cache(key)
        if cached is not None:
            return cached["members"]

        prompt = self.predictor.create_movement_prediction_prompt(
            stock_symbol, movement_type, technical_data, movement_stats
        )
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Rare in batch backtests; create a fresh loop.
                members = asyncio.run_coroutine_threadsafe(
                    self._query_ensemble_async(stock_symbol, movement_type, prompt), loop
                ).result()
            else:
                members = loop.run_until_complete(
                    self._query_ensemble_async(stock_symbol, movement_type, prompt)
                )
        except RuntimeError:
            members = asyncio.run(
                self._query_ensemble_async(stock_symbol, movement_type, prompt)
            )

        self._save_cache(key, {"members": members})
        return members

    # ---- public API ------------------------------------------------------
    def aggregate(self, members: List[Dict]) -> Tuple[int, float]:
        """Eq. (5): confidence-weighted, reliability-weighted vote at threshold tau."""
        num = 0.0
        den = 0.0
        for m in members:
            w = self._weight(m["model_name"])
            c = max(0.0, min(1.0, float(m["confidence"])))
            yhat = 1 if int(m["prediction"]) == 1 else 0
            num += w * c * yhat
            den += w * c
        if den <= 0:
            return 0, 0.5
        score = num / den                 # in [0, 1]
        prediction = 1 if score >= self.tau else 0
        # Report the aggregate's confidence as the margin-scaled agreement on the
        # winning class, so downstream AUC has a usable probability-like score.
        confidence = score if prediction == 1 else (1.0 - score)
        confidence = max(0.0, min(1.0, confidence))
        return prediction, confidence

    def predict_day(self, stock_symbol: str, movement_type: str,
                    technical_data: Dict, movement_stats: Dict) -> Tuple[int, float]:
        """Causal ensemble prediction for a single day. Returns (pred, confidence).

        `technical_data` / `movement_stats` must be computed from history up to
        day t only (the caller enforces this).
        """
        if not technical_data:
            return 0, 0.5
        members = self._get_member_predictions(
            stock_symbol, movement_type, technical_data, movement_stats
        )
        return self.aggregate(members)
