"""HybridMemoryAgent — Personal AI Assistant Memory System.

Combines Vector Store (episodic memory via Qdrant) and Feature Store
(stable user profile + recent query velocity via Feast).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, Filter, FieldCondition, MatchValue, PointStruct, VectorParams
from rank_bm25 import BM25Okapi

from feast import FeatureStore

ROOT_DIR = Path(__file__).resolve().parent.parent
FEAST_DIR = ROOT_DIR / "app" / "feast_repo"
COLLECTION_NAME = "bonus_episodic_memory"


class HybridMemoryAgent:
    """Combines Qdrant vector episodic memory and Feast online feature store."""

    def __init__(self, user_id: str = "u_001") -> None:
        self.default_user_id = user_id
        self.embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        
        # Initialize in-memory Qdrant client for episodic memory
        self.client = QdrantClient(":memory:")
        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

        # Initialize Feast FeatureStore if registry exists, else fallback to mock
        self.fs: FeatureStore | None = None
        if (FEAST_DIR / "registry.db").exists():
            try:
                self.fs = FeatureStore(repo_path=str(FEAST_DIR))
            except Exception:
                self.fs = None

        # Memory storage for BM25 keyword matching fallback
        self.memories: list[dict[str, Any]] = []
        self._memory_id_counter = 0

    def remember(self, text: str, user_id: str | None = None, topic: str = "general") -> None:
        """Add a new piece of episodic memory for this user."""
        uid = user_id or self.default_user_id
        self._memory_id_counter += 1
        mem_id = self._memory_id_counter

        # 1. Embed memory text
        vec = next(self.embedder.embed([text])).tolist()

        # 2. Store in Qdrant with payload filters
        payload = {
            "memory_id": mem_id,
            "user_id": uid,
            "topic": topic,
            "text": text,
        }
        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=[PointStruct(id=mem_id, vector=vec, payload=payload)],
        )
        self.memories.append(payload)

    def _get_user_features(self, uid: str) -> dict[str, Any]:
        """Fetch stable profile + recent activity from Feast online store."""
        if self.fs is not None:
            try:
                features = self.fs.get_online_features(
                    features=[
                        "user_profile_features:reading_speed_wpm",
                        "user_profile_features:preferred_language",
                        "user_profile_features:topic_affinity",
                        "query_velocity_features:queries_last_hour",
                        "query_velocity_features:distinct_topics_24h",
                    ],
                    entity_rows=[{"user_id": uid}],
                ).to_dict()
                return {k: v[0] for k, v in features.items()}
            except Exception:
                pass

        # Fallback profile if Feast store is uninitialized
        return {
            "reading_speed_wpm": 220,
            "preferred_language": "vi",
            "topic_affinity": "cloud",
            "queries_last_hour": 14,
            "distinct_topics_24h": 4,
        }

    def recall(self, query: str, user_id: str | None = None, top_k: int = 3) -> str:
        """Retrieve top-K episodic memories + user profile features -> assembled context string."""
        uid = user_id or self.default_user_id
        
        # Step 1: Retrieve user profile from Feast
        profile = self._get_user_features(uid)

        # Step 2: Vector search Qdrant filtered by user_id
        q_vec = next(self.embedder.embed([query])).tolist()
        user_filter = Filter(
            must=[FieldCondition(key="user_id", match=MatchValue(value=uid))]
        )
        
        search_res = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=q_vec,
            query_filter=user_filter,
            limit=top_k * 2,
        ).points

        # Step 3: BM25 keyword rerank / fusion on user's memories
        user_mems = [m for m in self.memories if m["user_id"] == uid]
        top_memories = []
        if user_mems:
            tokenized = [m["text"].lower().split() for m in user_mems]
            bm25 = BM25Okapi(tokenized)
            bm25_scores = bm25.get_scores(query.lower().split())
            
            # Simple RRF combining Qdrant dense rank and BM25 sparse rank
            rrf_scores: dict[int, float] = {}
            mem_by_id = {m["memory_id"]: m for m in user_mems}
            
            # Vector ranks
            for r, point in enumerate(search_res, 1):
                m_id = point.payload["memory_id"]
                rrf_scores[m_id] = rrf_scores.get(m_id, 0.0) + 1.0 / (60 + r)
            
            # BM25 ranks
            ranked_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: -bm25_scores[i])
            for r, idx in enumerate(ranked_bm25_indices[:top_k * 2], 1):
                m_id = user_mems[idx]["memory_id"]
                rrf_scores[m_id] = rrf_scores.get(m_id, 0.0) + 1.0 / (60 + r)
            
            sorted_mids = sorted(rrf_scores.keys(), key=lambda mid: -rrf_scores[mid])[:top_k]
            top_memories = [mem_by_id[mid]["text"] for mid in sorted_mids]
        else:
            top_memories = [p.payload["text"] for p in search_res[:top_k]]

        # Step 4: Assemble structured prompt context
        speed = profile.get("reading_speed_wpm", "N/A")
        lang = profile.get("preferred_language", "vi")
        affinity = profile.get("topic_affinity", "AI/Cloud")
        q_hour = profile.get("queries_last_hour", 0)

        context_blocks = [
            "=== USER PROFILE & STABLE CONTEXT ===",
            f"• Preferred Language: {lang}",
            f"• Topic Affinity: {affinity}",
            f"• Reading Speed: {speed} wpm",
            f"• Recent Activity (1h): {q_hour} queries",
            "",
            "=== RETRIEVED EPISODIC MEMORIES ===",
        ]

        if top_memories:
            for i, mem in enumerate(top_memories, 1):
                context_blocks.append(f"  {i}. {mem}")
        else:
            context_blocks.append("  (No relevant episodic memory found)")

        return "\n".join(context_blocks)
