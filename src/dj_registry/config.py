"""Registry configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RegistryConfig:
    """Central configuration for the track registry."""

    # Library
    library_roots: list[str] = field(default_factory=lambda: ["./files"])
    supported_extensions: list[str] = field(
        default_factory=lambda: [".mp3", ".aiff", ".aif"]
    )

    # Songstats
    songstats_api_key: str = ""
    songstats_base_url: str = "https://api.songstats.com/enterprise/v1"
    songstats_rate_limit: float = 1.0  # requests per second
    songstats_batch_limit: int = 100

    # Rekordbox
    rekordbox_xml_path: str = ""
    path_prefix_map: dict[str, str] = field(default_factory=dict)

    # Analysis
    run_essentia: bool = True
    # <= 0 means auto-detect (see local_analysis._default_analysis_workers).
    # Kept at 1 (serial) as the programmatic default so library/test callers are
    # deterministic; user-facing CLIs default to 0 (auto).
    analysis_workers: int = 1

    # Resolver weights
    source_weights: dict[str, float] = field(default_factory=lambda: {
        "manual": 1.00,
        "analysis_essentia": 0.85,
        "analysis_librosa": 0.80,
        "rekordbox": 0.75,
        "tag": 0.55,
        "songstats": 0.40,
    })
    confidence_threshold: float = 0.55  # librosa at 0.75 conf → score 0.60; needs headroom
    margin_threshold: float = 0.05     # analysis vs tag margin is typically 0.05 at this score range
    agreement_boost: float = 0.15
    cross_type_boost: float = 0.10

    # Identity
    duration_tolerance_strong: float = 2.0
    duration_tolerance_weak: float = 5.0

    # Paths
    output_dir: str = "./outputs/registry"
    cache_dir: str = ""

    @property
    def cache_root(self) -> str:
        """Directory for shared raw/derived caches.

        User-facing runs operate on a library root, so their cache should live
        beside that library rather than depend on the process working directory.
        The historical default ``./files`` keeps using ``./cache`` for tests and
        direct module callers that do not configure a real library root.
        """
        if self.cache_dir:
            return self.cache_dir
        root = self.library_roots[0] if self.library_roots else ""
        if root and os.path.normpath(root) not in {os.path.normpath("./files"), "files"}:
            return os.path.join(root, "cache")
        return os.path.join("cache")

    @property
    def raw_cache_path(self) -> str:
        return os.path.join(self.cache_root, "raw_cache.pkl")

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.output_dir, "raw")

    @property
    def reports_dir(self) -> str:
        return os.path.join(self.output_dir, "reports")

    @property
    def snapshots_dir(self) -> str:
        return os.path.join(self.output_dir, "snapshots")

    @property
    def tracks_master_path(self) -> str:
        return os.path.join(self.output_dir, "tracks_master.csv")

    @property
    def files_master_path(self) -> str:
        return os.path.join(self.output_dir, "files_master.csv")

    @property
    def observations_path(self) -> str:
        return os.path.join(self.output_dir, "source_observations.csv")

    @property
    def review_queue_path(self) -> str:
        return os.path.join(self.output_dir, "review_queue.csv")

    @property
    def payload_index_path(self) -> str:
        return os.path.join(self.output_dir, "source_payload_index.csv")

    def load_env(self) -> None:
        """Load environment variables from .env file."""
        try:
            from dotenv import load_dotenv
            candidates = [
                Path.cwd() / ".env",
                Path(__file__).resolve().parents[2] / ".env",
            ]
            loaded = False
            for candidate in candidates:
                if candidate.exists():
                    load_dotenv(candidate, override=False)
                    loaded = True
            if not loaded:
                load_dotenv()
        except Exception:
            pass
        if not self.songstats_api_key:
            self.songstats_api_key = os.environ.get("SONGSTATS_API_KEY", "")
