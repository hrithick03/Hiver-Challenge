"""
Robust LLM client for Google Gemini API with local caching, retries, and fallback support.
"""

import os
import sys
import json
import time
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import requests
from dotenv import load_dotenv

# Ensure root directory is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import (
    CACHE_DIR,
    DEFAULT_GEMINI_MODEL,
    FALLBACK_GEMINI_MODEL,
    MAX_OUTPUT_TOKENS,
)

load_dotenv()
logger = logging.getLogger(__name__)

CACHE_FILE = CACHE_DIR / "llm_cache.json"


class GeminiClient:
    _quota_exhausted: bool = False

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        use_cache: bool = True,
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "").strip()
        self.model_name = model_name or os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        self.use_cache = use_cache
        self.cache: Dict[str, str] = self._load_cache()

    def _load_cache(self) -> Dict[str, str]:
        if self.use_cache and CACHE_FILE.exists():
            try:
                with open(CACHE_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return {}

    def _save_cache(self):
        if not self.use_cache:
            return
        try:
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def _cache_key(self, prompt: str, system_prompt: Optional[str], model: str) -> str:
        combined = f"{model}::{system_prompt or ''}::{prompt}"
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        response_json_schema: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = None,
        max_retries: int = 2,
    ) -> str:
        """
        Generate completion using Gemini API with disk-backed cache.
        """
        target_model = model_name or self.model_name
        cache_key = self._cache_key(prompt, system_prompt, target_model)

        # Return cached response if available
        if self.use_cache and cache_key in self.cache:
            return self.cache[cache_key]

        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Please set GEMINI_API_KEY in your .env file or environment, "
                "or run in pre-cached benchmark mode."
            )

        if GeminiClient._quota_exhausted:
            raise RuntimeError("Daily Gemini API quota exhausted. Using instant local fallback.")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:generateContent?key={self.api_key}"

        body: Dict[str, Any] = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": MAX_OUTPUT_TOKENS,
            },
        }

        if system_prompt:
            body["systemInstruction"] = {"parts": [{"text": system_prompt}]}

        if response_json_schema:
            body["generationConfig"]["responseMimeType"] = "application/json"

        headers = {"Content-Type": "application/json"}

        last_error = None
        for attempt in range(max_retries):
            try:
                response = requests.post(url, headers=headers, json=body, timeout=12)
                if response.status_code == 200:
                    data = response.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts:
                            text = parts[0].get("text", "").strip()
                            if self.use_cache:
                                self.cache[cache_key] = text
                                self._save_cache()
                            return text
                    raise ValueError(f"Empty candidate response from Gemini: {data}")
                elif response.status_code == 429:
                    # Check if daily quota exhausted
                    if "RESOURCE_EXHAUSTED" in response.text or "QuotaFailure" in response.text:
                        GeminiClient._quota_exhausted = True
                        raise RuntimeError("Gemini Free Tier daily quota limit reached (20 requests/day). Switching immediately to fast local pipeline.")
                    time.sleep(1.0)
                    last_error = f"HTTP 429: {response.text}"
                else:
                    raise RuntimeError(f"Gemini API Error {response.status_code}: {response.text}")
            except requests.RequestException as e:
                last_error = str(e)
                time.sleep(1.0)

        raise RuntimeError(f"Failed to generate after {max_retries} attempts: {last_error}")
