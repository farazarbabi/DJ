"""Configuration: all weights, thresholds, and paths in one place."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GrouperConfig:
    # --- Layer weights for blended distance ---
    w_tags: float = 0.25
    w_dsp: float = 0.30
    w_embed: float = 0.45

    # When CLAP is not available, DSP drives similarity — tags constrain
    w_tags_no_embed: float = 0.30
    w_dsp_no_embed: float = 0.70

    # --- Key weighting by vibe ---
    key_weight_by_vibe: dict[str, float] = field(default_factory=lambda: {
        "MEL": 0.8, "ACID": 0.6,
        "DEEP": 0.4, "ATM": 0.3,
        "HYPN": 0.2, "TRIB": 0.2,
        "DRK": 0.15, "RAW": 0.1,
    })
    key_weight_vocal_boost: float = 0.2

    # --- BPM ---
    bpm_hard_cutoff_pct: float = 0.08  # 8% absolute hard cutoff for recommendations
    bpm_soft_penalty_pct: float = 0.04  # penalty starts at 4%
    bpm_penalty_weight: float = 0.15    # max penalty contribution
    bpm_norm_range: tuple[float, float] = (100.0, 40.0)  # (min, range) -> 100-140 BPM
    bpm_group_max_spread_pct: float = 0.06  # max BPM spread within a group (6%)
    energy_group_max_spread: int = 2        # max energy level spread within a group

    # --- Clustering ---
    linkage: str = "average"
    min_group_size: int = 1      # allow singletons
    max_group_size: int = 20
    target_group_size: tuple[int, int] = (1, 6)
    vocal_confidence_threshold: float = 0.5

    # --- CLAP ---
    clap_pca_dims: int = 64

    # --- Stable mode ---
    new_group_distance_threshold: float = 0.8

    # --- Recommendations ---
    n_recommendations: int = 10

    # --- Feedback ---
    good_pair_factor: float = 0.3
    bad_pair_factor: float = 0.5

    # --- Default paths ---
    input_dir: str = "./files"
    output_dir: str = "./outputs"
    cache_file: str = "./outputs/features_cache.pkl"
    feedback_file: str = "./outputs/feedback.csv"
    groups_file: str = "./outputs/groups.csv"
    recommendations_file: str = "./outputs/recommendations.csv"
    grouped_dir: str = "./outputs/Grouped"
    playlists_dir: str = "./outputs/playlists"
