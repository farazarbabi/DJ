"""Registry configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


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
    confidence_threshold: float = 0.70
    margin_threshold: float = 0.20
    agreement_boost: float = 0.15
    cross_type_boost: float = 0.10

    # Identity
    duration_tolerance_strong: float = 2.0
    duration_tolerance_weak: float = 5.0

    # Paths
    output_dir: str = "./outputs/registry"

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
            load_dotenv()
        except ImportError:
            pass
        if not self.songstats_api_key:
            self.songstats_api_key = os.environ.get("SONGSTATS_API_KEY", "")
