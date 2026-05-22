"""CLAP zero-shot vocal/instrumental fallback (NOT WIRED IN).

Empirical calibration on 30+ tracks with cached CLAP embeddings showed the
HTSAT-tiny model with the current prompt set is heavily biased toward the
"instrumental" prompts — even tracks with featured singers (Liset Alea,
Malou) produce negative margins. Until prompts and/or model are recalibrated
this signal is less reliable than the spectral heuristic and should not be
the primary signal.

The module is left in place so a future iteration can replace prompts, swap
to a larger CLAP checkpoint, or use it as a tie-breaker alongside Demucs.
Demucs htdemucs (analyzers/vocal_stem.py) is the primary detector.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

_CLAP_MODEL = None
_PROMPT_EMBEDDINGS: np.ndarray | None = None

_VOCAL_PROMPTS: tuple[str, ...] = (
    "a song with a singer singing lyrics",
    "a track with prominent female vocals",
    "a track with prominent male vocals",
)
_INSTRUMENTAL_PROMPTS: tuple[str, ...] = (
    "an instrumental track with no vocals",
    "electronic music with no singing",
    "instrumental melodic techno without vocals",
)


@dataclass
class VocalClapResult:
    vocal_clap_score: float        # cosine similarity to vocal prompts (0-1)
    instrumental_clap_score: float  # cosine similarity to instrumental prompts (0-1)
    vocal_clap_margin: float        # vocal_score - instrumental_score


def is_clap_available() -> bool:
    try:
        import laion_clap  # noqa: F401
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _get_model():
    global _CLAP_MODEL
    if _CLAP_MODEL is None:
        import laion_clap
        _CLAP_MODEL = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-tiny")
        _CLAP_MODEL.load_ckpt()
        logger.info("CLAP model loaded for vocal scoring")
    return _CLAP_MODEL


def _get_prompt_embeddings() -> tuple[np.ndarray, np.ndarray]:
    """Return (vocal_mean, instrumental_mean) prompt embeddings, cached after first call."""
    global _PROMPT_EMBEDDINGS
    if _PROMPT_EMBEDDINGS is not None:
        return _PROMPT_EMBEDDINGS  # type: ignore[return-value]
    model = _get_model()
    all_prompts = list(_VOCAL_PROMPTS) + list(_INSTRUMENTAL_PROMPTS)
    text_emb = np.asarray(model.get_text_embedding(all_prompts, use_tensor=False))
    text_emb = text_emb / (np.linalg.norm(text_emb, axis=1, keepdims=True) + 1e-9)
    n_voc = len(_VOCAL_PROMPTS)
    vocal_mean = text_emb[:n_voc].mean(axis=0)
    inst_mean = text_emb[n_voc:].mean(axis=0)
    _PROMPT_EMBEDDINGS = (vocal_mean, inst_mean)  # type: ignore[assignment]
    return vocal_mean, inst_mean


def analyze_vocal_clap(clap_embedding: np.ndarray | None) -> VocalClapResult | None:
    """Score vocal vs instrumental via CLAP cosine similarity.

    ``clap_embedding`` is a 512-dim audio embedding (already cached by the
    grouper pipeline). Returns None if unavailable.
    """
    if clap_embedding is None or not isinstance(clap_embedding, np.ndarray):
        return None
    if clap_embedding.size == 0:
        return None
    if not is_clap_available():
        return None
    try:
        vocal_mean, inst_mean = _get_prompt_embeddings()
    except Exception:
        logger.warning("Failed to compute CLAP prompt embeddings", exc_info=True)
        return None

    audio = clap_embedding.astype(np.float32)
    audio = audio / (np.linalg.norm(audio) + 1e-9)
    voc_sim = float(np.dot(audio, vocal_mean))
    inst_sim = float(np.dot(audio, inst_mean))
    return VocalClapResult(
        vocal_clap_score=voc_sim,
        instrumental_clap_score=inst_sim,
        vocal_clap_margin=voc_sim - inst_sim,
    )


def result_to_dict(r: VocalClapResult | None) -> dict[str, float]:
    if r is None:
        return {}
    return {
        "vocal_clap_score": round(r.vocal_clap_score, 4),
        "vocal_clap_inst_score": round(r.instrumental_clap_score, 4),
        "vocal_clap_margin": round(r.vocal_clap_margin, 4),
    }
