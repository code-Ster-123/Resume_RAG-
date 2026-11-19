import os
import re
import json
import time
import asyncio
import numpy as np
from typing import Optional, List, Dict, Any
from llama_index.core.embeddings import BaseEmbedding
from app.core.interfaces import CacheMatch

def normalize_query(query: str) -> str:
    """
    Normalizes a user query to extract canonical intent and remove syntactic framing.
    Used strictly for embedding comparison in the semantic cache.
    """
    text = query.lower().strip()
    
    # 1. Strip punctuation
    text = re.sub(r'[?!.,;:\'\"`~()\[\]{}]+', ' ', text)
    
    # 2. Strip question framing and polite introductory phrasing
    question_frames = (
        r'\b(does he have|has he done|has he published|is there any|what are|what were|'
        r'what is|tell me about|can you tell me|can you list|list|give me|describe|'
        r'do you know if|do you know|who is|please share)\b'
    )
    text = re.sub(question_frames, ' ', text)
    
    # 3. Strip candidate references / pronouns
    entity_refs = r'\b(his|the candidate\'?s?|utkarsh\'?s?|he|him|candidate|person)\b'
    text = re.sub(entity_refs, ' ', text)
    
    clean_core = re.sub(r'\s+', ' ', text).strip()
    
    # 4. Resume Facet Anchoring (ordered intentionally: specific facets before broad work experience)
    # Research & Publications
    if any(k in clean_core for k in ['research', 'paper', 'publication', 'conference', 'workshop', 'cikm', 'ijcai', 'patent', 'thesis']):
        return f"research experience publications papers and contributions {clean_core}"
        
    # Educational Background
    if any(k in clean_core for k in ['education', 'degree', 'college', 'university', 'academic', 'study', 'studied', 'qualification', 'gpa', 'liverpool', 'meri']):
        return f"educational background degrees academic qualifications {clean_core}"
        
    # Technical Skills & Tools
    if any(k in clean_core for k in ['skill', 'stack', 'tool', 'language', 'python', 'technolog', 'framework']):
        return f"technical skills programming languages and tools {clean_core}"
        
    # Work Experience & Employment History
    if any(k in clean_core for k in ['work', 'job', 'experience', 'company', 'companies', 'employ', 'career', 'role', 'techdome']):
        return f"work experience employment history past jobs {clean_core}"
        
    # Projects & Portfolio
    if any(k in clean_core for k in ['project', 'portfolio', 'github', 'built', 'application']):
        return f"projects portfolio applications built {clean_core}"
        
    return clean_core or query.strip()


class InMemorySemanticCache:
    """
    In-Memory Semantic Cache using vectorized cosine similarity on canonicalized query vectors.
    Persists cached queries, normalized embeddings, and responses to JSON on disk.
    """
    def __init__(
        self,
        embed_model: BaseEmbedding,
        threshold: float = 0.85,
        persist_path: Optional[str] = "data/semantic_cache.json",
        max_entries: int = 1000,
    ):
        self.embed_model = embed_model
        self.threshold = threshold
        self.persist_path = persist_path
        self.max_entries = max_entries
        
        self.entries: List[Dict[str, Any]] = []
        self.matrix: Optional[np.ndarray] = None  # Shape: (N, D), normalized rows
        self._lock = asyncio.Lock()
        
        self._load_from_disk()

    def _load_from_disk(self) -> None:
        if not self.persist_path or not os.path.exists(self.persist_path):
            return

        try:
            with open(self.persist_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list):
                return

            loaded_entries = []
            vectors = []
            for item in data:
                raw_query = item.get("query", "")
                if not raw_query:
                    continue

                # Re-compute embedding using canonicalized query to ensure consistency
                norm_q = normalize_query(raw_query)
                vec = self.embed_model.get_text_embedding(norm_q)
                
                v_arr = np.array(vec, dtype=np.float32)
                norm = np.linalg.norm(v_arr)
                if norm > 0:
                    v_norm = v_arr / norm
                    vectors.append(v_norm)
                    loaded_entries.append({
                        "query": raw_query,
                        "normalized_query": norm_q,
                        "response": item.get("response", ""),
                        "hit_count": item.get("hit_count", 0),
                        "timestamp": item.get("timestamp", time.time())
                    })

            if vectors:
                self.matrix = np.vstack(vectors)
                self.entries = loaded_entries
                print(f"[SemanticCache] Successfully loaded and canonicalized {len(self.entries)} entries from {self.persist_path}")
        except Exception as e:
            print(f"[SemanticCache] Warning: Failed to load cache from {self.persist_path}: {e}")

    def persist(self) -> None:
        if not self.persist_path or not self.entries:
            return

        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.persist_path)), exist_ok=True)
            export_data = []
            for idx, entry in enumerate(self.entries):
                vec = self.matrix[idx].tolist() if self.matrix is not None else []
                export_data.append({
                    "query": entry["query"],
                    "normalized_query": entry.get("normalized_query", normalize_query(entry["query"])),
                    "embedding": vec,
                    "response": entry["response"],
                    "hit_count": entry.get("hit_count", 0),
                    "timestamp": entry.get("timestamp", time.time()),
                })

            with open(self.persist_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            print(f"[SemanticCache] Persisted {len(export_data)} entries to {self.persist_path}")
        except Exception as e:
            print(f"[SemanticCache] Warning: Failed to persist cache: {e}")

    async def get(self, query: str) -> Optional[CacheMatch]:
        clean_query = query.strip()
        if not clean_query or len(self.entries) == 0 or self.matrix is None:
            return None

        # Canonicalize the incoming query for embedding
        norm_q = normalize_query(clean_query)
        print(f"[SemanticCache] Query: '{clean_query}' -> Canonical: '{norm_q}'")

        try:
            raw_vec = await asyncio.to_thread(self.embed_model.get_text_embedding, norm_q)
            q_arr = np.array(raw_vec, dtype=np.float32)
            norm = np.linalg.norm(q_arr)
            if norm == 0:
                return None
            q_norm = q_arr / norm
        except Exception as e:
            print(f"[SemanticCache] Embedding computation error during get: {e}")
            return None

        # Compute cosine similarities in parallel against all cached embeddings
        sims = np.dot(self.matrix, q_norm)
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        print(f"[SemanticCache] Best match: '{self.entries[best_idx]['query']}' (Sim: {best_sim:.4f}, Threshold: {self.threshold})")

        if best_sim >= self.threshold:
            self.entries[best_idx]["hit_count"] = self.entries[best_idx].get("hit_count", 0) + 1
            return CacheMatch(
                query=self.entries[best_idx]["query"],
                response=self.entries[best_idx]["response"],
                similarity=best_sim
            )

        return None

    async def set(self, query: str, response: str) -> None:
        clean_query = query.strip()
        if not clean_query or not response:
            return

        norm_q = normalize_query(clean_query)

        async with self._lock:
            # Re-check exact or existing query
            for idx, entry in enumerate(self.entries):
                if entry["query"].lower() == clean_query.lower() or entry.get("normalized_query") == norm_q:
                    entry["response"] = response
                    entry["timestamp"] = time.time()
                    self.persist()
                    return

            try:
                raw_vec = await asyncio.to_thread(self.embed_model.get_text_embedding, norm_q)
                q_arr = np.array(raw_vec, dtype=np.float32)
                norm = np.linalg.norm(q_arr)
                if norm == 0:
                    return
                q_norm = q_arr / norm
            except Exception as e:
                print(f"[SemanticCache] Embedding computation error during set: {e}")
                return

            new_entry = {
                "query": clean_query,
                "normalized_query": norm_q,
                "response": response,
                "hit_count": 0,
                "timestamp": time.time()
            }

            # Manage eviction if cache capacity exceeded
            if len(self.entries) >= self.max_entries:
                oldest_idx = int(np.argmin([e.get("timestamp", 0) for e in self.entries]))
                self.entries.pop(oldest_idx)
                if self.matrix is not None:
                    self.matrix = np.delete(self.matrix, oldest_idx, axis=0)

            # Append entry and update matrix
            self.entries.append(new_entry)
            q_norm_2d = q_norm.reshape(1, -1)
            if self.matrix is None or len(self.matrix) == 0:
                self.matrix = q_norm_2d
            else:
                self.matrix = np.vstack([self.matrix, q_norm_2d])

            print(f"[SemanticCache] Cached new response for '{clean_query}'. Total entries: {len(self.entries)}")
            self.persist()
