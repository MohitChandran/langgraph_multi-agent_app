"""Embedding model and reranker, loaded ONCE and shared (plan 02, section 5.7).

Embedding = turning a text into a list of 384 numbers (a vector) such that texts with similar
MEANING get vectors that point in similar directions. "service credit for AWS downtime" lands
near the SLA table even though they share few words. The vector search in R6 compares these.

Model (decision D3): ibm-granite/granite-embedding-small-english-r2 via FastEmbed:
  - FastEmbed runs ONNX models on the CPU, without PyTorch: small and fast to load.
  - 384 dimensions (matches the vector(384) column), 8,192-token input.
  - No "query:" / "passage:" prefixes needed (some models, e.g. E5/BGE, require them).
  - Its vectors are NOT normalized, so we normalize them (length 1) ourselves.

Reranker (decision D8, used in R6): a cross-encoder reads the question and a chunk TOGETHER and
scores how well the chunk answers it. Much more precise than comparing two separate vectors,
but too slow to run on every chunk, so it only re-orders the top ~20 candidates.

Both are CPU-bound: async callers must wrap them in asyncio.to_thread (plan gotcha 7).
The first use downloads the model files (~0.2 GB) into DATA_DIR/models, so it's done at startup.
"""

from pathlib import Path

import numpy as np
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

from app.agents.doc.store import EMBEDDING_DIM

EMBED_BATCH_SIZE = 32  # chunks are short; small batches keep memory low next to Docling


def normalize(vector: np.ndarray) -> np.ndarray:
    """Scale a vector to length 1 (float32), so its direction is all that matters.

    Cosine distance ignores length anyway, but normalized vectors make scores comparable
    across chunks and stay correct under any distance function.
    """
    vector = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


class Embedder:
    """Turns texts into normalized 384-d vectors. The SAME model embeds chunks and questions (gotcha 4)."""

    def __init__(self, model_name: str, cache_dir: Path) -> None:
        """Load the model (downloading it on first use) and check its dimension matches the database column.

        cache_dir: FastEmbed's default is the system temp folder, which is wiped on reboot;
        a folder under DATA_DIR keeps the download.
        """
        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name, cache_dir=str(cache_dir))
        dimension = len(self.embed_query("dimension check"))
        if dimension != EMBEDDING_DIM:
            raise RuntimeError(
                f"{model_name} produces {dimension}-d vectors but document_chunks.embedding is vector({EMBEDDING_DIM})."
            )

    def embed_documents(self, texts: list[str]) -> list[np.ndarray]:
        """Embed many texts (chunk embed_texts) in batches; returns one normalized vector per text, in order."""
        return [normalize(vector) for vector in self._model.embed(texts, batch_size=EMBED_BATCH_SIZE)]

    def embed_query(self, text: str) -> np.ndarray:
        """Embed one question. No prefix: this model doesn't use one (don't copy BGE's instruction, gotcha 4)."""
        return normalize(next(iter(self._model.embed([text]))))


class Reranker:
    """Cross-encoder that scores (question, chunk) pairs; higher = more relevant. Used in R6."""

    def __init__(self, model_name: str, cache_dir: Path) -> None:
        """Load the cross-encoder (downloading it on first use)."""
        self.model_name = model_name
        self._model = TextCrossEncoder(model_name=model_name, cache_dir=str(cache_dir))

    def score(self, query: str, texts: list[str]) -> list[float]:
        """One relevance score per text, in the same order as `texts`."""
        return [float(score) for score in self._model.rerank(query, texts)]
