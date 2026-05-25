"""
News Article Scoring - Three Methods
  Method A: Keyword Baseline   - simple word matching
  Method B: Hybrid System      - NER/Regex + semantic embeddings + time decay
  Method C: Random Projection LSH - approximate nearest-neighbor search

Importable module; no FastAPI dependency.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ─── Default keyword lists ────────────────────────────────────────────────────

CLEAN_ENERGY_KEYWORDS: List[str] = [
    "renewable energy", "clean energy", "solar", "wind power", "hydroelectric",
    "green energy", "carbon neutral", "sustainability", "emission reduction",
    "renewable", "photovoltaic", "wind farm", "solar panel", "green innovation",
    "climate", "decarbonisation", "ESG",
]

INVESTMENT_THEMES: Dict[str, List[str]] = {
    "Clean Energy":    ["renewable energy", "solar", "wind power", "clean energy",
                        "carbon neutral", "green innovation", "sustainability", "climate"],
    "Digital Banking": ["digital banking", "fintech", "mobile payment", "blockchain", "cryptocurrency"],
    "Logistics":       ["logistics", "supply chain", "warehouse", "shipping", "delivery", "transport"],
    "Manufacturing":   ["manufacturing", "production", "factory", "industrial", "automation"],
    "Real Estate":     ["real estate", "property", "construction", "housing", "development"],
}


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_news_data(path: str = "news_data.json") -> pd.DataFrame:
    """
    Load articles from a news_data.json file.
    Returns a DataFrame with columns: news_id, publish_date, text, title, ...
    """
    with open(path, "r") as f:
        news_json = json.load(f)

    articles = news_json.get("articles", [])
    df = pd.DataFrame(articles)

    if df.empty:
        return df

    df["news_id"] = range(len(df))
    df["publish_date"] = pd.to_datetime(df["published_date"])
    df["text"] = df["title"] + " " + df["description"].fillna("")
    if "body" in df.columns:
        df["full_text"] = df["body"]
    df["text_length"] = df["text"].str.len()
    return df


def load_news_from_supabase(
    supabase_url: str,
    supabase_key: str,
    schema: str = "market_data",
    limit: int = 0,
    asset_id: Optional[str] = None,
    page_size: int = 500,
) -> pd.DataFrame:
    """
    Load news articles from Supabase `news` table with automatic pagination.
    Returns DataFrame with columns: news_id, publish_date, title, text, source, source_url, asset_id, sentiment_score, text_length

    Args:
        limit: Maximum total rows to fetch (0 = fetch all).
        page_size: Rows per paginated request (max allowed by server).
    """
    from supabase import create_client

    sb = create_client(supabase_url, supabase_key)
    sbs = sb.schema(schema)

    all_rows: list = []
    offset = 0

    while True:
        batch_size = page_size if (limit == 0) else min(page_size, limit - offset)
        if batch_size <= 0:
            break

        query = sbs.table("news").select(
            "id,news_id,publish_date,title,source,news_content,sentiment_score,source_url,asset_id"
        )
        if asset_id:
            query = query.eq("asset_id", asset_id)
        query = query.range(offset, offset + batch_size - 1)

        r = query.execute()
        rows = r.data
        all_rows.extend(rows)

        if len(rows) < batch_size:
            break  # last page
        offset += batch_size
        if limit and offset >= limit:
            break

    df = pd.DataFrame(all_rows)

    if df.empty:
        return df

    df["publish_date"] = pd.to_datetime(df["publish_date"], format="ISO8601", utc=True)
    df["text"] = df["title"].fillna("") + " " + df["news_content"].fillna("")
    df["text_length"] = df["text"].str.len()
    # Expose UUID primary key as the join key for embeddings
    if "id" in df.columns:
        df["news_uuid"] = df["id"].astype(str)
    return df


def load_embeddings_from_supabase(
    supabase_url: str,
    supabase_key: str,
    schema: str = "market_data",
    news_ids: Optional[List[int]] = None,
    page_size: int = 500,
) -> Dict[int, np.ndarray]:
    """
    Load pre-computed embeddings from Supabase `news_embeddings` table with pagination.
    All chunks per article are mean-pooled into a single 768-d vector so the
    full title+content is represented, not only the first chunk.
    Returns dict mapping news_id -> numpy array.
    """
    from supabase import create_client
    from collections import defaultdict

    sb = create_client(supabase_url, supabase_key)
    sbs = sb.schema(schema)

    all_rows: list = []
    offset = 0
    while True:
        query = sbs.table("news_embeddings").select("news_row_id,embedding,chunk_index")
        if news_ids:
            query = query.in_("news_row_id", news_ids)
        query = query.range(offset, offset + page_size - 1)
        r = query.execute()
        all_rows.extend(r.data)
        if len(r.data) < page_size:
            break
        offset += page_size

    chunks: Dict[str, List[np.ndarray]] = defaultdict(list)
    for row in all_rows:
        emb = row["embedding"]
        if isinstance(emb, str):
            emb = json.loads(emb)
        chunks[str(row["news_row_id"])].append(np.array(emb, dtype=np.float32))

    result: Dict[str, np.ndarray] = {}
    for nid, vecs in chunks.items():
        stacked = np.stack(vecs)
        mean_vec = stacked.mean(axis=0)
        norm = np.linalg.norm(mean_vec)
        result[nid] = mean_vec / norm if norm > 0 else mean_vec

    return result


# ─── Ground truth helpers ─────────────────────────────────────────────────────

def load_ground_truth(path: str = "ground_truth_labels.json") -> Dict[str, bool]:
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def save_ground_truth(labels: Dict[str, bool], path: str = "ground_truth_labels.json") -> None:
    with open(path, "w") as f:
        json.dump(labels, f, indent=2)


def create_labeling_sample(df: pd.DataFrame, n: int = 50, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    return df.sample(min(n, len(df)))


# ═══════════════════════════════════════════════════════════════════════════════
# CHUNK PIPELINE  (split → retrieve → rerank → classify → topic-score → causal)
# ═══════════════════════════════════════════════════════════════════════════════

def split_article(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
) -> List[str]:
    """
    Split article text into overlapping chunks, preferring sentence boundaries.

    Args:
        text:       Raw article text (title + content concatenated).
        chunk_size: Target chunk length in characters.
        overlap:    Character overlap between consecutive chunks.

    Returns:
        List of non-empty chunk strings.
    """
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            for sep in (". ", "? ", "! ", "\n"):
                idx = text.rfind(sep, start + chunk_size // 2, end)
                if idx != -1:
                    end = idx + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
    return chunks


class ChunkRetriever:
    """
    Step 2 of the chunk pipeline: embed chunks and retrieve the top-k most
    similar to a query embedding via cosine similarity.

    When no embed_fn is provided the retriever falls back to keyword-density
    ranking so the pipeline still works without a live embedding model.
    """

    def __init__(
        self,
        embed_fn=None,          # callable(List[str]) -> np.ndarray (n, dim)
        top_k: int = 5,
        keywords: Optional[List[str]] = None,
    ):
        self.embed_fn = embed_fn
        self.top_k    = top_k
        self._kw_pat  = (
            re.compile("|".join(map(re.escape, keywords)), re.IGNORECASE)
            if keywords else None
        )

    def _kw_density(self, text: str) -> float:
        if not self._kw_pat or not text:
            return 0.0
        return len(self._kw_pat.findall(text)) / max(len(text), 1)

    def retrieve(
        self,
        chunks: List[str],
        query_embedding: Optional[np.ndarray] = None,
    ) -> List[Tuple[str, float, int]]:
        """
        Returns list of (chunk_text, score, original_index) sorted descending.
        Score is cosine similarity when embeddings are available, else keyword density.
        """
        if not chunks:
            return []

        if query_embedding is not None and self.embed_fn is not None:
            chunk_embs = np.array(self.embed_fn(chunks), dtype=np.float32)
            q = query_embedding.astype(np.float32)
            sims = np.dot(chunk_embs, q) / (
                np.linalg.norm(chunk_embs, axis=1) * np.linalg.norm(q) + 1e-9
            )
            order = np.argsort(sims)[::-1][: self.top_k]
            return [(chunks[i], float(sims[i]), int(i)) for i in order]

        # Keyword-density fallback
        scored = [(c, self._kw_density(c), i) for i, c in enumerate(chunks)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[: self.top_k]


class ChunkReranker:
    """
    Step 3: re-score retrieved chunks by boosting those that contain domain
    keywords, improving precision before topic classification.

    strategy='cosine'        – keep the retrieval order unchanged.
    strategy='keyword_boost' – add a keyword-density bonus (capped at 0.3).
    """

    def __init__(
        self,
        keywords: Optional[List[str]] = None,
        strategy: str = "keyword_boost",
    ):
        self.strategy = strategy
        self._kw_pat  = (
            re.compile("|".join(map(re.escape, keywords)), re.IGNORECASE)
            if keywords else None
        )

    def rerank(
        self, chunks_with_scores: List[Tuple[str, float, int]]
    ) -> List[Tuple[str, float, int]]:
        """Input/output: list of (text, score, original_index)."""
        if self.strategy == "cosine" or self._kw_pat is None:
            return chunks_with_scores

        reranked = []
        for text, score, idx in chunks_with_scores:
            hits  = len(self._kw_pat.findall(text)) if text else 0
            boost = min(hits / max(len(text), 1) * 1000, 0.3)
            reranked.append((text, score + boost, idx))
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked


class TopicClassifier:
    """
    Step 4: keyword-based topic classifier — no ML model required.

    classify_chunk(text) → (topic_name, confidence)
    classify_article(chunks) → {topic: aggregate_score}
    """

    def __init__(self, themes: Optional[Dict[str, List[str]]] = None):
        self.themes   = themes or INVESTMENT_THEMES
        self._patterns = {
            name: re.compile("|".join(map(re.escape, kws)), re.IGNORECASE)
            for name, kws in self.themes.items()
        }

    def classify_chunk(self, text: str) -> Tuple[str, float]:
        if not text:
            return ("Unknown", 0.0)
        best_topic, best_score = "Unknown", 0.0
        n = max(len(text), 1)
        for topic, pat in self._patterns.items():
            score = len(pat.findall(text)) / n * 1000
            if score > best_score:
                best_topic, best_score = topic, score
        return (best_topic, round(best_score, 6))

    def classify_chunks(self, chunks: List[str]) -> List[Tuple[str, float]]:
        return [self.classify_chunk(c) for c in chunks]

    def classify_article(self, chunks: List[str]) -> Dict[str, float]:
        """Aggregate per-chunk hit densities into per-topic scores."""
        totals: Dict[str, float] = {t: 0.0 for t in self.themes}
        for text in chunks:
            n = max(len(text), 1)
            for topic, pat in self._patterns.items():
                totals[topic] += len(pat.findall(text)) / n * 1000
        return totals


class CausalImpactEstimator:
    """
    Step 6 (optional): lightweight causal-language detector.

    **Naming note:** Despite its name, this is a *heuristic* language detector,
    not a causal inference model (no do-calculus, no counterfactuals). It scans
    for causal-language phrases ("led to", "driven by", "as a result of") near
    topic keywords and returns a small score boost. A more accurate name would
    be `CausalPhraseBooster` or `CausalLanguageHeuristic`.

    Scans for causal phrases near topic keywords and returns a score boost
    in [0, max_boost] that can be added to the base hybrid score.
    """

    _CAUSAL_PHRASES: List[str] = [
        "because", "therefore", "as a result", "leads to", "caused by",
        "due to", "resulting in", "impacted by", "driven by", "following",
        "triggered", "boosted by", "weighed on", "lifted by", "attributed to",
        "on the back of", "amid concerns", "amid expectations",
    ]

    def __init__(self, max_boost: float = 0.15):
        self.max_boost = max_boost
        self._pattern  = re.compile(
            "|".join(map(re.escape, self._CAUSAL_PHRASES)), re.IGNORECASE
        )

    def estimate(self, text: str, topic_scores: Dict[str, float]) -> float:
        """
        Returns a causal boost proportional to causal-phrase density,
        scaled by whether the dominant topic already has a strong signal.
        """
        if not text:
            return 0.0
        hits        = len(self._pattern.findall(text))
        density     = hits / max(len(text), 1) * 5000
        topic_signal = max(topic_scores.values()) if topic_scores else 0.0
        boost        = min(density * (1 + topic_signal), self.max_boost)
        return round(float(boost), 6)


class TopicAwareScorer:
    """
    Full chunk-level scoring pipeline:

        segments = split_article(text)
             ↓
        Embedding → ChunkRetriever (top-k chunks)
             ↓
        ChunkReranker (keyword-boost rerank)
             ↓
        TopicClassifier (per-chunk topic + confidence)
             ↓
        Topic-wise scoring  (aggregation='max' | 'weighted')
             ↓
        (Optional) CausalImpactEstimator → score boost

    Usage:
        scorer = TopicAwareScorer(CLEAN_ENERGY_KEYWORDS, query_embedding=q_emb)
        result = scorer.score_article(text, publish_date, article_embedding=emb)
        df_out = scorer.score_batch(df, article_embeddings=embeddings_map)
    """

    def __init__(
        self,
        keywords: List[str],
        query_embedding: Optional[np.ndarray] = None,
        themes: Optional[Dict[str, List[str]]] = None,
        embed_fn=None,
        chunk_size: int = 512,
        overlap: int = 64,
        top_k_chunks: int = 5,
        aggregation: str = "weighted",        # "max" | "weighted"
        keyword_weight: float = 0.3,
        semantic_weight: float = 0.5,
        time_weight: float = 0.2,
        causal_impact: bool = False,
        causal_max_boost: float = 0.15,
    ):
        self.keywords         = keywords
        self.query_embedding  = (
            np.array(query_embedding, dtype=np.float32)
            if query_embedding is not None else None
        )
        self.themes           = themes or INVESTMENT_THEMES
        self.chunk_size       = chunk_size
        self.overlap          = overlap
        self.top_k_chunks     = top_k_chunks
        self.aggregation      = aggregation
        self.keyword_weight   = keyword_weight
        self.semantic_weight  = semantic_weight
        self.time_weight      = time_weight
        self.causal_impact    = causal_impact

        self._retriever  = ChunkRetriever(embed_fn=embed_fn, top_k=top_k_chunks, keywords=keywords)
        self._reranker   = ChunkReranker(keywords=keywords, strategy="keyword_boost")
        self._classifier = TopicClassifier(themes=self.themes)
        self._kw_scorer  = KeywordBaseline(keywords)
        self._causal     = CausalImpactEstimator(max_boost=causal_max_boost) if causal_impact else None

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _time_decay(self, publish_date: datetime) -> float:
        ref = datetime.now()
        if hasattr(publish_date, "tzinfo") and publish_date.tzinfo is not None:
            publish_date = publish_date.replace(tzinfo=None)
        days_old = (ref - publish_date).days
        return float(np.exp(-days_old / 365.0))

    def _aggregate_topic_scores(
        self, chunk_topic_scores: List[Tuple[str, float]]
    ) -> Dict[str, float]:
        """
        Collect per-chunk (topic, sim_score) and reduce to per-topic scalar.
        max      → single best chunk score per topic
        weighted → mean of top-3 chunk scores per topic
        """
        topic_buckets: Dict[str, List[float]] = defaultdict(list)
        for topic, score in chunk_topic_scores:
            topic_buckets[topic].append(score)

        result: Dict[str, float] = {}
        for topic in self.themes:
            scores = sorted(topic_buckets.get(topic, [0.0]), reverse=True)
            if self.aggregation == "max":
                result[topic] = scores[0]
            else:
                top3 = scores[:3]
                result[topic] = float(np.mean(top3)) if top3 else 0.0
        return result

    # ── Public API ────────────────────────────────────────────────────────────

    def score_article(
        self,
        text: str,
        publish_date: datetime,
        article_embedding: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Run the full chunk pipeline on a single article.

        When article_embedding is provided (mean-pooled from Supabase) it is used
        as the semantic similarity score; chunk retrieval falls back to keyword
        density because per-chunk embeddings are not stored separately.
        """
        # ── 1. Split ─────────────────────────────────────────────────────────
        chunks = split_article(text, chunk_size=self.chunk_size, overlap=self.overlap) or [text]

        # ── 2. Embed → retrieve top-k ─────────────────────────────────────
        if article_embedding is not None and self.query_embedding is not None:
            # Article-level embedding acts as semantic proxy; rank chunks by kw density
            sem_score = float(
                np.dot(self.query_embedding, article_embedding)
                / (np.linalg.norm(self.query_embedding) * np.linalg.norm(article_embedding) + 1e-9)
            )
            sem_score = max(0.0, sem_score)
            kw_retriever = ChunkRetriever(top_k=self.top_k_chunks, keywords=self.keywords)
            retrieved = kw_retriever.retrieve(chunks)
        else:
            retrieved = self._retriever.retrieve(chunks, self.query_embedding)
            chunk_sims = [s for _, s, _ in retrieved]
            sem_score  = float(max(chunk_sims)) if chunk_sims else 0.0

        # ── 3. Chunk-level rerank ─────────────────────────────────────────
        reranked    = self._reranker.rerank(retrieved)
        top_chunks  = [t for t, _, _ in reranked]
        chunk_sims  = [s for _, s, _ in reranked]

        # ── 4. Topic classification ───────────────────────────────────────
        chunk_topics = self._classifier.classify_chunks(top_chunks)   # [(topic, conf)]

        # ── 5. Topic-wise scoring ─────────────────────────────────────────
        chunk_topic_scores = [(t, s) for (t, _), s in zip(chunk_topics, chunk_sims)]
        topic_scores       = self._aggregate_topic_scores(chunk_topic_scores)
        dominant_topic     = max(topic_scores, key=topic_scores.__getitem__)

        # ── Base hybrid score ─────────────────────────────────────────────
        kw_score = min(self._kw_scorer.score_article(text) * 300, 1.0)
        td_score = self._time_decay(publish_date)

        if self.query_embedding is not None and sem_score > 0:
            base = (kw_score * self.keyword_weight
                    + sem_score * self.semantic_weight
                    + td_score * self.time_weight)
        else:
            base = kw_score * 0.7 + td_score * 0.3

        # ── 6. (Optional) causal impact ───────────────────────────────────
        causal_boost = self._causal.estimate(text, topic_scores) if self._causal else 0.0
        final_score  = min(base + causal_boost, 1.0)

        return {
            "hybrid_score":    round(final_score, 6),
            "keyword":         round(kw_score, 6),
            "semantic":        round(sem_score, 6),
            "time_decay":      round(td_score, 6),
            "causal_boost":    round(causal_boost, 6),
            "dominant_topic":  dominant_topic,
            "topic_scores":    {k: round(v, 6) for k, v in topic_scores.items()},
            "num_chunks":      len(chunks),
            "top_chunks_used": len(top_chunks),
        }

    def score_batch(
        self,
        df: pd.DataFrame,
        text_col: str = "text",
        date_col: str = "publish_date",
        uuid_col: str = "news_uuid",
        article_embeddings: Optional[Dict[str, np.ndarray]] = None,
    ) -> pd.DataFrame:
        article_embeddings = article_embeddings or {}
        results = []
        for _, row in df.iterrows():
            emb = article_embeddings.get(str(row.get(uuid_col, "")))
            results.append(self.score_article(row[text_col], row[date_col], article_embedding=emb))

        out = df.copy()
        out["hybrid_score"]   = [r["hybrid_score"]   for r in results]
        out["kw_score"]       = [r["keyword"]         for r in results]
        out["sem_score"]      = [r["semantic"]        for r in results]
        out["time_decay"]     = [r["time_decay"]      for r in results]
        out["causal_boost"]   = [r["causal_boost"]    for r in results]
        out["dominant_topic"] = [r["dominant_topic"]  for r in results]
        out["num_chunks"]     = [r["num_chunks"]      for r in results]
        for topic in self.themes:
            col = "topic_" + topic.lower().replace(" ", "_")
            out[col] = [r["topic_scores"].get(topic, 0.0) for r in results]
        return out.sort_values("hybrid_score", ascending=False)


# ═══════════════════════════════════════════════════════════════════════════════
# METHOD A - Keyword Baseline
# ═══════════════════════════════════════════════════════════════════════════════

class KeywordBaseline:
    """
    Method A: score = count(keyword matches in text) / len(text)

    Usage:
        scorer = KeywordBaseline(keywords)
        score  = scorer.score_article("solar panels are expanding ...")
        df_out = scorer.find_relevant(df, threshold=0.0)
    """

    def __init__(self, keywords: List[str]):
        self.keywords = [k.lower() for k in keywords]
        self.pattern = re.compile(
            "|".join(map(re.escape, self.keywords)), re.IGNORECASE
        )

    def score_article(self, text: str) -> float:
        if not text:
            return 0.0
        matches = self.pattern.findall(text.lower())
        return len(matches) / len(text)

    def score_batch(self, texts: List[str]) -> np.ndarray:
        return np.array([self.score_article(t) for t in texts])

    def find_relevant(
        self,
        df: pd.DataFrame,
        text_col: str = "text",
        threshold: float = 0.0,
    ) -> pd.DataFrame:
        scores = self.score_batch(df[text_col].tolist())
        result = df.copy()
        result["keyword_score"] = scores
        return result[result["keyword_score"] > threshold].sort_values(
            "keyword_score", ascending=False
        )


# ═══════════════════════════════════════════════════════════════════════════════
# METHOD B - Hybrid System
# ═══════════════════════════════════════════════════════════════════════════════

class HybridScorer:
    """
    Method B:
        score = keyword_match * 0.3 + semantic_similarity * 0.5 + time_decay * 0.2

    Degrades gracefully when sentence-transformers is not installed
    (keyword * 0.7 + time_decay * 0.3).

    Usage:
        scorer = HybridScorer(keywords, theme_query, model=model)
        result = scorer.score_article(text, publish_date)
        df_out = scorer.score_batch(df)
    """

    def __init__(
        self,
        keywords: List[str],
        theme_query: str,
        model=None,
        keyword_weight: float = 0.3,
        semantic_weight: float = 0.5,
        time_weight: float = 0.2,
        query_embedding: Optional[np.ndarray] = None,
        article_embeddings: Optional[Dict[int, np.ndarray]] = None,
    ):
        self.keywords = [k.lower() for k in keywords]
        self.theme_query = theme_query
        self.model = model
        self.keyword_weight = keyword_weight
        self.semantic_weight = semantic_weight
        self.time_weight = time_weight
        self.article_embeddings = article_embeddings or {}

        if query_embedding is not None:
            self.query_embedding = np.array(query_embedding, dtype=np.float32)
        elif model:
            self.query_embedding = model.encode([theme_query])[0]
        else:
            self.query_embedding = None

    # ── Component scores ──────────────────────────────────────────────────────

    def keyword_score(self, text: str) -> float:
        if not text:
            return 0.0
        text_lower = text.lower()
        matches = sum(1 for kw in self.keywords if kw in text_lower)
        return min(matches / 3.0, 1.0)

    def semantic_score(self, text: str, article_id: Optional[int] = None) -> float:
        if self.query_embedding is None or not text:
            return 0.0
        if article_id is not None and article_id in self.article_embeddings:
            text_emb = self.article_embeddings[article_id]
        elif self.model:
            text_emb = self.model.encode([text])[0]
        else:
            return 0.0
        q = self.query_embedding
        sim = np.dot(q, text_emb) / (np.linalg.norm(q) * np.linalg.norm(text_emb) + 1e-9)
        return float(max(0.0, sim))

    def time_decay_score(
        self,
        publish_date: datetime,
        reference_date: Optional[datetime] = None,
    ) -> float:
        if reference_date is None:
            reference_date = datetime.now()
        if publish_date.tzinfo is not None:
            publish_date = publish_date.replace(tzinfo=None)
        days_old = (reference_date - publish_date).days
        return float(np.exp(-days_old / 365.0))

    # ── Aggregate score ───────────────────────────────────────────────────────

    def score_article(
        self, text: str, publish_date: datetime, article_id: Optional[int] = None
    ) -> Dict[str, float]:
        kw = self.keyword_score(text)
        sem = self.semantic_score(text, article_id=article_id)
        td = self.time_decay_score(publish_date)

        has_semantic = self.query_embedding is not None and (
            self.model or (article_id is not None and article_id in self.article_embeddings)
        )
        if not has_semantic:
            total = kw * 0.7 + td * 0.3
        else:
            total = kw * self.keyword_weight + sem * self.semantic_weight + td * self.time_weight

        return {
            "keyword": kw,
            "semantic": sem,
            "time_decay": td,
            "hybrid_score": total,
        }

    def score_batch(
        self,
        df: pd.DataFrame,
        text_col: str = "text",
        date_col: str = "publish_date",
        id_col: str = "news_uuid",
    ) -> pd.DataFrame:
        results = [
            self.score_article(
                row[text_col],
                row[date_col],
                article_id=str(row[id_col]) if id_col in df.columns else None,
            )
            for _, row in df.iterrows()
        ]
        out = df.copy()
        for key in ("keyword", "semantic", "time_decay", "hybrid_score"):
            out[f"hybrid_{key}"] = [r[key] for r in results]
        return out.sort_values("hybrid_hybrid_score", ascending=False)


# ═══════════════════════════════════════════════════════════════════════════════
# METHOD C - Random Projection LSH
# ═══════════════════════════════════════════════════════════════════════════════

class RandomProjectionLSH:
    """
    Method C: Approximate nearest-neighbor search via random projection LSH.

    Usage:
        lsh = RandomProjectionLSH(num_hyperplanes=12, embedding_dim=384)
        lsh.add_batch(embeddings, metadata_list)
        results = lsh.query(query_vector, top_k=20)
        # results -> list of (metadata_dict, similarity_score)
    """

    def __init__(self, num_hyperplanes: int = 10, embedding_dim: int = 384):
        self.num_hyperplanes = num_hyperplanes
        self.embedding_dim = embedding_dim
        np.random.seed(42)
        self.hyperplanes = np.random.randn(num_hyperplanes, embedding_dim)
        self.hyperplanes /= np.linalg.norm(self.hyperplanes, axis=1, keepdims=True)
        self.hash_tables: Dict[str, List[int]] = defaultdict(list)
        self.embeddings: List[np.ndarray] = []
        self.metadata: List[dict] = []

    def _hash_vector(self, vector: np.ndarray) -> str:
        projections = np.dot(self.hyperplanes, vector)
        return "".join(map(str, (projections > 0).astype(int)))

    def add_item(self, vector: np.ndarray, metadata: dict) -> None:
        idx = len(self.embeddings)
        self.embeddings.append(vector)
        self.metadata.append(metadata)
        self.hash_tables[self._hash_vector(vector)].append(idx)

    def add_batch(
        self, vectors: np.ndarray, metadata_list: List[dict]
    ) -> None:
        for vec, meta in zip(vectors, metadata_list):
            self.add_item(vec, meta)

    def _get_candidates(self, query_vector: np.ndarray, top_k: int) -> List[int]:
        """Return de-duplicated candidate indices from LSH bucket lookup."""
        hash_key = self._hash_vector(query_vector)
        seen: set = set()
        candidates: List[int] = []
        for idx in self.hash_tables.get(hash_key, []):
            if idx not in seen:
                seen.add(idx)
                candidates.append(idx)
        if len(candidates) < top_k:
            for hk, idxs in self.hash_tables.items():
                if hk == hash_key:
                    continue
                for idx in idxs:
                    if idx not in seen:
                        seen.add(idx)
                        candidates.append(idx)
                if len(candidates) >= top_k * 5:
                    break
        if not candidates:
            candidates = list(range(len(self.embeddings)))
        return candidates

    def query(
        self, query_vector: np.ndarray, top_k: int = 20
    ) -> List[Tuple[dict, float]]:
        """
        LSH bucket lookup → exact cosine similarity on candidates.
        Returns list of (metadata_dict, cosine_similarity).
        """
        candidates = self._get_candidates(query_vector, top_k)
        sims = []
        for idx in candidates:
            vec = self.embeddings[idx]
            sim = float(
                np.dot(query_vector, vec)
                / (np.linalg.norm(query_vector) * np.linalg.norm(vec) + 1e-9)
            )
            sims.append((idx, sim))
        sims.sort(key=lambda x: x[1], reverse=True)
        return [(self.metadata[idx], sim) for idx, sim in sims[:top_k]]

    def query_hybrid(
        self,
        query_vector: np.ndarray,
        keywords: List[str],
        top_k: int = 20,
        keyword_weight: float = 0.3,
        semantic_weight: float = 0.5,
        time_weight: float = 0.2,
        reference_date: Optional[datetime] = None,
    ) -> List[Dict]:
        """
        Method C with Method-B-style scoring.

        Pipeline:
            Query → LSH candidates → exact cosine similarity
                  → keyword score (from metadata['text'])
                  → time-decay score (from metadata['publish_date'])
                  → hybrid_score = keyword*kw_w + semantic*sem_w + time_decay*td_w

        Each metadata dict **must** contain 'text' and 'publish_date' (ISO date str
        or datetime). Use add_batch with metadata including those fields.

        Returns list of dicts (sorted by hybrid_score desc), each with keys:
            hybrid_score, keyword, semantic, time_decay, + all metadata fields.
        """
        if reference_date is None:
            reference_date = datetime.now()

        kw_pat = re.compile("|".join(map(re.escape, [k.lower() for k in keywords])), re.IGNORECASE)

        def _kw_score(text: str) -> float:
            if not text:
                return 0.0
            matches = sum(1 for _ in kw_pat.finditer(text))
            return min(matches / 3.0, 1.0)

        def _time_decay(pub) -> float:
            if isinstance(pub, str):
                try:
                    pub = datetime.fromisoformat(pub)
                except ValueError:
                    return 1.0
            if hasattr(pub, "tzinfo") and pub.tzinfo is not None:
                pub = pub.replace(tzinfo=None)
            days_old = max((reference_date - pub).days, 0)
            return float(np.exp(-days_old / 365.0))

        candidates = self._get_candidates(query_vector, top_k)
        results = []
        for idx in candidates:
            vec  = self.embeddings[idx]
            meta = self.metadata[idx]
            sem  = float(
                np.dot(query_vector, vec)
                / (np.linalg.norm(query_vector) * np.linalg.norm(vec) + 1e-9)
            )
            sem  = max(0.0, sem)
            kw   = _kw_score(meta.get("text", ""))
            td   = _time_decay(meta.get("publish_date", reference_date))
            hyb  = kw * keyword_weight + sem * semantic_weight + td * time_weight
            results.append({
                "hybrid_score": round(hyb, 6),
                "keyword":      round(kw, 6),
                "semantic":     round(sem, 6),
                "time_decay":   round(td, 6),
                **meta,
            })

        results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return results[:top_k]

    def stats(self) -> Dict[str, float]:
        bucket_sizes = [len(v) for v in self.hash_tables.values()]
        return {
            "num_vectors": len(self.embeddings),
            "num_buckets": len(self.hash_tables),
            "avg_bucket_size": float(np.mean(bucket_sizes)) if bucket_sizes else 0.0,
            "max_bucket_size": int(max(bucket_sizes)) if bucket_sizes else 0,
            "num_hyperplanes": self.num_hyperplanes,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_metrics(
    predictions: List[bool], ground_truth_labels: List[bool]
) -> Dict[str, float]:
    """Compute precision, recall, F1, accuracy and confusion matrix counts."""
    p = np.array(predictions, dtype=bool)
    g = np.array(ground_truth_labels, dtype=bool)

    tp = int(np.sum(p & g))
    fp = int(np.sum(p & ~g))
    fn = int(np.sum(~p & g))
    tn = int(np.sum(~p & ~g))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy  = (tp + tn) / len(p) if len(p) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1_score":  round(f1, 4),
        "accuracy":  round(accuracy, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def evaluate_method(
    df_labeled: pd.DataFrame,
    score_col: str,
    threshold: float,
    label_col: str = "is_clean_energy",
) -> Dict[str, float]:
    """Threshold a score column and compare against ground-truth labels."""
    predictions = (df_labeled[score_col] > threshold).tolist()
    ground_truth_labels = df_labeled[label_col].astype(bool).tolist()
    return calculate_metrics(predictions, ground_truth_labels)


# ═══════════════════════════════════════════════════════════════════════════════
# LATENCY BENCHMARKS
# ═══════════════════════════════════════════════════════════════════════════════

def benchmark_exact_search(
    embeddings: np.ndarray,
    query_embedding: np.ndarray,
    top_k: int = 20,
    num_runs: int = 100,
) -> Dict[str, float]:
    """Benchmark brute-force cosine similarity search."""
    times = []
    for _ in range(num_runs):
        t0 = time.time()
        sims = np.dot(embeddings, query_embedding) / (
            np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        np.argsort(sims)[-top_k:]
        times.append(time.time() - t0)

    return {
        "median_ms": round(float(np.median(times)) * 1000, 4),
        "mean_ms":   round(float(np.mean(times))   * 1000, 4),
        "std_ms":    round(float(np.std(times))     * 1000, 4),
        "min_ms":    round(float(np.min(times))     * 1000, 4),
        "max_ms":    round(float(np.max(times))     * 1000, 4),
    }


def benchmark_lsh_search(
    lsh_index: RandomProjectionLSH,
    query_embedding: np.ndarray,
    top_k: int = 20,
    num_runs: int = 100,
) -> Dict[str, float]:
    """Benchmark LSH approximate search."""
    times = []
    for _ in range(num_runs):
        t0 = time.time()
        lsh_index.query(query_embedding, top_k=top_k)
        times.append(time.time() - t0)

    return {
        "median_ms": round(float(np.median(times)) * 1000, 4),
        "mean_ms":   round(float(np.mean(times))   * 1000, 4),
        "std_ms":    round(float(np.std(times))     * 1000, 4),
        "min_ms":    round(float(np.min(times))     * 1000, 4),
        "max_ms":    round(float(np.max(times))     * 1000, 4),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# THEMATIC COVERAGE
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_theme_score(
    df: pd.DataFrame, keywords: List[str], text_col: str = "text"
) -> float:
    """
    Keyword density score for a theme across all articles in *df*.
    Returns avg(keyword_matches / text_length * 1000).
    """
    if df.empty:
        return 0.0
    pattern = re.compile("|".join(map(re.escape, keywords)), re.IGNORECASE)
    scores = []
    for text in df[text_col]:
        if pd.notna(text):
            n = len(pattern.findall(str(text).lower()))
            scores.append(n / max(len(str(text)), 1) * 1000)
    return float(np.mean(scores)) if scores else 0.0


def thematic_coverage(
    df: pd.DataFrame,
    themes: Dict[str, List[str]] = INVESTMENT_THEMES,
    text_col: str = "text",
) -> List[Dict]:
    """
    Return a list of theme coverage dicts sorted by score descending.
    Each dict: {"theme", "coverage_score", "num_keywords"}
    """
    results = []
    for theme_name, keywords in themes.items():
        score = calculate_theme_score(df, keywords, text_col=text_col)
        results.append({
            "theme":          theme_name,
            "coverage_score": round(score, 6),
            "num_keywords":   len(keywords),
        })
    return sorted(results, key=lambda x: x["coverage_score"], reverse=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ASSET LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_assets_from_supabase(
    supabase_url: str,
    supabase_key: str,
    schema: str = "market_data",
    asset_class: Optional[str] = "STOCK",
    page_size: int = 500,
) -> pd.DataFrame:
    """
    Load assets from Supabase ``assets`` table with automatic pagination.

    Args:
        asset_class: Filter by asset class (``"STOCK"``, ``None`` = all classes).

    Returns:
        DataFrame with columns: id, symbol, name_en, name_local, sector,
        industry, asset_class, market, status.
    """
    from supabase import create_client

    sb  = create_client(supabase_url, supabase_key)
    sbs = sb.schema(schema)

    all_rows: list = []
    offset = 0
    while True:
        query = sbs.table("assets").select(
            "id,symbol,name_en,name_local,sector,industry,asset_class,market,status"
        )
        if asset_class:
            query = query.eq("asset_class", asset_class)
        query = query.eq("status", "active")
        query = query.range(offset, offset + page_size - 1)
        r = query.execute()
        all_rows.extend(r.data)
        if len(r.data) < page_size:
            break
        offset += page_size

    df = pd.DataFrame(all_rows)
    if df.empty:
        return df

    df["sector"]     = df["sector"].fillna("Unknown")
    df["industry"]   = df["industry"].fillna("")
    df["name_en"]    = df["name_en"].fillna("")
    df["name_local"] = df["name_local"].fillna("")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# TICKER IMPACT RECOMMENDER
# ═══════════════════════════════════════════════════════════════════════════════

class TickerImpactRecommender:
    """
    Given a news article, recommend which stock tickers are likely impacted.

    Three signals contribute to the impact score:

    1. **Direct mention** (weight 0.45) — symbol token or company name found
       verbatim in the article text.
    2. **Sector match** (weight 0.35) — the article's dominant investment topic
       (from :class:`TopicClassifier`) aligns with the asset's sector.
    3. **Hybrid relevance** (weight 0.20) — overall hybrid score from the
       :class:`TopicAwareScorer` pipeline.

    Usage::

        assets_df = load_assets_from_supabase(url, key)
        rec = TickerImpactRecommender(assets_df)

        impacts = rec.recommend(text="...", publish_date=datetime.now(), top_k=10)
        # [{"symbol": "VCB", "impact_score": 0.62, "direct_mention": True, ...}, ...]

        # Or score an entire DataFrame and get one row per (article, ticker) pair:
        long_df = rec.recommend_batch(news_df, top_k=5)

        # Or find the top news for a single ticker:
        news_for_ticker = rec.score_for_ticker(news_df, symbol="VCB", top_k=10)
    """

    # Maps INVESTMENT_THEMES keys → Supabase sector/industry substrings
    THEME_SECTOR_MAP: Dict[str, List[str]] = {
        "Clean Energy":    ["energy", "utilities", "electricity", "power", "oil", "gas"],
        "Digital Banking": ["banking", "insurance", "finance", "securities", "investment"],
        "Logistics":       ["transportation", "logistics", "aviation", "shipping", "freight"],
        "Manufacturing":   ["manufacturing", "industrial", "chemicals", "steel", "material",
                            "plastic", "rubber", "textile"],
        "Real Estate":     ["real estate", "construction", "property", "housing", "infrastructure"],
    }

    _DIRECT_WEIGHT: float = 0.45
    _SECTOR_WEIGHT: float = 0.35
    _HYBRID_WEIGHT: float = 0.20

    def __init__(self, assets_df: pd.DataFrame) -> None:
        self._assets = assets_df.copy().reset_index(drop=True)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _direct_mention_score(self, text: str) -> pd.Series:
        """Return float Series (1.0 or 0.0) for each asset — 1.0 if the asset's
        symbol or company name appears in the article text."""
        text_upper = text.upper()
        text_lower = text.lower()
        scores = []
        for _, row in self._assets.iterrows():
            sym = (row["symbol"] or "").upper()
            matched = bool(re.search(r"\b" + re.escape(sym) + r"\b", text_upper)) if sym else False
            if not matched and row["name_en"]:
                for word in row["name_en"].split():
                    if len(word) >= 4 and word.lower() in text_lower:
                        matched = True
                        break
            if not matched and row["name_local"]:
                if row["name_local"].lower() in text_lower:
                    matched = True
            scores.append(1.0 if matched else 0.0)
        return pd.Series(scores, index=self._assets.index)

    def _sector_match_score(
        self, topic_scores: Dict[str, float]
    ) -> pd.Series:
        """Return float Series in [0, 1] for each asset based on sector-theme
        alignment, weighted by normalised topic scores."""
        total = sum(topic_scores.values()) or 1.0
        norm  = {t: s / total for t, s in topic_scores.items()}

        scores = []
        for _, row in self._assets.iterrows():
            combined = (row["sector"] + " " + row["industry"]).lower()
            asset_score = 0.0
            for theme, weight in norm.items():
                if weight < 0.02:
                    continue
                if any(kw in combined for kw in self.THEME_SECTOR_MAP.get(theme, [])):
                    asset_score += weight
            scores.append(min(asset_score, 1.0))
        return pd.Series(scores, index=self._assets.index)

    # ── Public API ────────────────────────────────────────────────────────────

    def recommend(
        self,
        text: str,
        publish_date: datetime,
        article_embedding: Optional[np.ndarray] = None,
        query_embedding: Optional[np.ndarray] = None,
        top_k: int = 10,
        min_score: float = 0.05,
        keywords: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Score a single article and return the top-k impacted tickers.

        Returns list of dicts (sorted by ``impact_score`` desc), each with:
        ``symbol``, ``name_en``, ``sector``, ``industry``, ``impact_score``,
        ``direct_mention``, ``sector_match``, ``hybrid_score``,
        ``dominant_topic``, ``topic_scores``.
        """
        if self._assets.empty:
            return []

        all_kw = keywords or [kw for kws in INVESTMENT_THEMES.values() for kw in kws]
        scorer = TopicAwareScorer(
            keywords=all_kw,
            query_embedding=query_embedding,
            themes=INVESTMENT_THEMES,
        )
        result        = scorer.score_article(text, publish_date, article_embedding=article_embedding)
        hybrid_score  = result["hybrid_score"]
        topic_scores  = result["topic_scores"]
        dominant_topic = result["dominant_topic"]

        direct_scores = self._direct_mention_score(text)
        sector_scores = self._sector_match_score(topic_scores)

        impact = (
            direct_scores * self._DIRECT_WEIGHT
            + sector_scores * self._SECTOR_WEIGHT
            + hybrid_score  * self._HYBRID_WEIGHT
        )

        output = []
        for i in impact[impact >= min_score].sort_values(ascending=False).head(top_k).index:
            row = self._assets.loc[i]
            output.append({
                "symbol":         row["symbol"],
                "name_en":        row["name_en"],
                "sector":         row["sector"],
                "industry":       row["industry"],
                "impact_score":   round(float(impact[i]), 4),
                "direct_mention": bool(direct_scores[i] > 0),
                "sector_match":   round(float(sector_scores[i]), 4),
                "hybrid_score":   round(float(hybrid_score), 4),
                "dominant_topic": dominant_topic,
                "topic_scores":   topic_scores,
            })
        return output

    def recommend_batch(
        self,
        df: pd.DataFrame,
        text_col: str = "text",
        date_col: str = "publish_date",
        uuid_col: str = "news_uuid",
        article_embeddings: Optional[Dict[str, np.ndarray]] = None,
        top_k: int = 5,
        min_score: float = 0.05,
    ) -> pd.DataFrame:
        """
        Score all articles in *df* and return a long-form DataFrame with one
        row per (article, ticker) pair, sorted by ``impact_score`` descending.
        """
        article_embeddings = article_embeddings or {}
        rows_out: list = []
        for _, row in df.iterrows():
            emb     = article_embeddings.get(str(row.get(uuid_col, "")))
            impacts = self.recommend(
                text=row[text_col],
                publish_date=row[date_col],
                article_embedding=emb,
                top_k=top_k,
                min_score=min_score,
            )
            for imp in impacts:
                rows_out.append({
                    "news_uuid":    row.get(uuid_col, ""),
                    "title":        row.get("title", ""),
                    "publish_date": row.get(date_col, ""),
                    "source":       row.get("source", ""),
                    **imp,
                })
        return pd.DataFrame(rows_out)

    def score_for_ticker(
        self,
        df: pd.DataFrame,
        symbol: str,
        text_col: str = "text",
        date_col: str = "publish_date",
        uuid_col: str = "news_uuid",
        article_embeddings: Optional[Dict[str, np.ndarray]] = None,
        top_k: int = 10,
        min_score: float = 0.05,
    ) -> List[Dict]:
        """
        Efficient inverse lookup: given a ticker symbol, find the top news
        articles that most impact it.  Avoids the full O(articles × assets)
        cost of calling :meth:`recommend` per article.

        Returns list of article dicts sorted by ``impact_score`` descending.
        Each dict has: ``news_id``, ``title``, ``publish_date``, ``source``,
        ``source_url``, ``impact_score``, ``direct_mention``, ``sector_match``,
        ``hybrid_score``, ``dominant_topic``.
        """
        symbol_upper = symbol.upper()
        asset_rows   = self._assets[self._assets["symbol"] == symbol_upper]
        if asset_rows.empty:
            return []

        asset        = asset_rows.iloc[0]
        combined_sec = (asset["sector"] + " " + asset["industry"]).lower()
        name_words   = [w for w in (asset["name_en"] or "").split() if len(w) >= 4]

        all_kw  = [kw for kws in INVESTMENT_THEMES.values() for kw in kws]
        scorer  = TopicAwareScorer(keywords=all_kw, themes=INVESTMENT_THEMES)
        article_embeddings = article_embeddings or {}

        results: list = []
        for _, row in df.iterrows():
            text     = row[text_col]
            pub_date = row[date_col]
            emb      = article_embeddings.get(str(row.get(uuid_col, "")))

            # 1. Direct mention
            direct = 1.0 if (
                re.search(r"\b" + re.escape(symbol_upper) + r"\b", text.upper())
                or any(w.lower() in text.lower() for w in name_words)
                or (asset["name_local"] and asset["name_local"].lower() in text.lower())
            ) else 0.0

            # 2. Article scoring
            scored         = scorer.score_article(text, pub_date, article_embedding=emb)
            hybrid         = scored["hybrid_score"]
            topic_scores   = scored["topic_scores"]
            dominant_topic = scored["dominant_topic"]

            # 3. Sector match for this specific asset
            total    = sum(topic_scores.values()) or 1.0
            sec_score = 0.0
            for theme, ts in topic_scores.items():
                if ts / total < 0.02:
                    continue
                if any(kw in combined_sec for kw in self.THEME_SECTOR_MAP.get(theme, [])):
                    sec_score += ts / total
            sec_score = min(sec_score, 1.0)

            impact = (
                direct    * self._DIRECT_WEIGHT
                + sec_score * self._SECTOR_WEIGHT
                + hybrid    * self._HYBRID_WEIGHT
            )

            if impact >= min_score:
                results.append({
                    "news_id":       row.get("news_id", ""),
                    "title":         row.get("title", ""),
                    "publish_date":  str(pub_date),
                    "source":        row.get("source", ""),
                    "source_url":    row.get("source_url", ""),
                    "impact_score":  round(float(impact), 4),
                    "direct_mention": bool(direct > 0),
                    "sector_match":  round(float(sec_score), 4),
                    "hybrid_score":  round(float(hybrid), 4),
                    "dominant_topic": dominant_topic,
                })

        results.sort(key=lambda x: x["impact_score"], reverse=True)
        return results[:top_k]
