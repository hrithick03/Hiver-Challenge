"""
Historical Resolution Retriever: Retrieves historically proven customer-support
interaction pairs using TF-IDF / BM25 semantic keyword matching.
"""

import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# Ensure root directory is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import KB_PATH

logger = logging.getLogger(__name__)


class HistoricalRetriever:
    def __init__(self, kb_path: Optional[Path] = None):
        self.kb_path = kb_path or KB_PATH
        self.entries: List[Dict[str, Any]] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self._load_and_index()

    def _load_and_index(self):
        if not self.kb_path.exists():
            logger.warning(f"Knowledge base file {self.kb_path} not found. Operating with empty index.")
            return

        with open(self.kb_path, "r", encoding="utf-8") as f:
            self.entries = json.load(f)

        if not self.entries:
            logger.warning("Knowledge base entries list is empty.")
            return

        # Prepare corpus from customer query + historical resolution
        corpus = [
            f"{e.get('customer_query', '')} {e.get('historical_resolution', '')}"
            for e in self.entries
        ]

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        logger.info(f"Indexed {len(self.entries)} historical resolutions in retriever.")

    def retrieve(
        self, query: str, top_k: int = 3, intent_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top_k historically relevant resolutions for an incoming customer query.
        Optionally filters by intent.
        """
        if not self.entries or self.vectorizer is None or self.tfidf_matrix is None:
            return []

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Apply intent filter if requested
        if intent_filter:
            filtered_indices = [
                i for i, e in enumerate(self.entries) if e.get("intent") == intent_filter
            ]
            if filtered_indices:
                filtered_sims = [(i, similarities[i]) for i in filtered_indices]
                filtered_sims.sort(key=lambda x: x[1], reverse=True)
                top_indices = [idx for idx, _ in filtered_sims[:top_k]]
            else:
                top_indices = np.argsort(similarities)[::-1][:top_k].tolist()
        else:
            top_indices = np.argsort(similarities)[::-1][:top_k].tolist()

        results = []
        for idx in top_indices:
            entry = self.entries[idx].copy()
            entry["score"] = float(similarities[idx])
            results.append(entry)

        return results


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    retriever = HistoricalRetriever()
    test_query = "My battery is dying so fast since updating to iOS 11"
    hits = retriever.retrieve(test_query, top_k=2)
    print(f"Retrieved {len(hits)} hits for '{test_query}':")
    for i, h in enumerate(hits, 1):
        print(f"\nHit {i} (Score: {h['score']:.3f}):")
        print("Customer:", h["customer_query"])
        print("Historical Resolution:", h["historical_resolution"])
