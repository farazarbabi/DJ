"""CSV-based registry store with atomic writes and snapshots."""

from __future__ import annotations

import csv
import logging
import os
import shutil
from dataclasses import asdict, fields
from datetime import datetime
from typing import TypeVar, Type

from ..models import (
    FileRecord,
    LogicalTrack,
    PayloadIndexEntry,
    ReviewItem,
    SourceObservation,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", LogicalTrack, FileRecord, SourceObservation, ReviewItem, PayloadIndexEntry)


def _field_names(cls: Type[T]) -> list[str]:
    """Get ordered field names for a dataclass."""
    return [f.name for f in fields(cls)]


def _row_to_dataclass(cls: Type[T], row: dict[str, str]) -> T:
    """Convert a CSV row dict to a dataclass instance, tolerating missing/extra columns."""
    kwargs: dict = {}
    for f in fields(cls):
        raw = row.get(f.name, "")
        if f.type in ("float", float) or (hasattr(f.type, "__args__") and float in getattr(f.type, "__args__", ())):
            try:
                kwargs[f.name] = float(raw) if raw else 0.0
            except ValueError:
                kwargs[f.name] = 0.0
        elif f.type in ("int", int):
            try:
                kwargs[f.name] = int(raw) if raw else 0
            except ValueError:
                kwargs[f.name] = 0
        elif f.type in ("bool", bool):
            kwargs[f.name] = raw.lower() in ("true", "1", "yes") if raw else False
        else:
            kwargs[f.name] = raw
    return cls(**kwargs)


def _dataclass_to_row(obj: T) -> dict[str, str]:
    """Convert a dataclass instance to a CSV row dict."""
    row: dict[str, str] = {}
    for key, val in asdict(obj).items():
        if isinstance(val, bool):
            row[key] = str(val).lower()
        elif val is None:
            row[key] = ""
        else:
            row[key] = str(val)
    return row


def _atomic_write_csv(path: str, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    """Write CSV atomically: write to temp file, then rename."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _read_csv(path: str) -> list[dict[str, str]]:
    """Read CSV file, return list of row dicts. Returns [] if file missing."""
    if not os.path.exists(path):
        return []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


class CsvStore:
    """Registry CSV store with atomic writes."""

    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _path(self, name: str) -> str:
        return os.path.join(self.output_dir, name)

    # -- Snapshots -----------------------------------------------------------

    def snapshot(self, run_id: str) -> str:
        """Copy current CSVs to a timestamped snapshot directory. Returns path."""
        snap_dir = os.path.join(self.output_dir, "snapshots", run_id)
        os.makedirs(snap_dir, exist_ok=True)
        for name in ("tracks_master.csv", "files_master.csv",
                      "source_observations.csv", "review_queue.csv",
                      "source_payload_index.csv"):
            src = self._path(name)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(snap_dir, name))
        logger.debug("Snapshot: %s", snap_dir)
        return snap_dir

    # -- Generic load/save ---------------------------------------------------

    def _load(self, filename: str, cls: Type[T]) -> list[T]:
        rows = _read_csv(self._path(filename))
        return [_row_to_dataclass(cls, r) for r in rows]

    def _save(self, filename: str, items: list[T], cls: Type[T]) -> None:
        fnames = _field_names(cls)
        rows = [_dataclass_to_row(item) for item in items]
        _atomic_write_csv(self._path(filename), rows, fnames)

    # -- Tracks --------------------------------------------------------------

    def load_tracks(self) -> list[LogicalTrack]:
        return self._load("tracks_master.csv", LogicalTrack)

    def save_tracks(self, tracks: list[LogicalTrack]) -> None:
        self._save("tracks_master.csv", tracks, LogicalTrack)

    def upsert_track(self, track: LogicalTrack) -> None:
        """Insert or update a track by track_id."""
        tracks = self.load_tracks()
        for i, t in enumerate(tracks):
            if t.track_id == track.track_id:
                tracks[i] = track
                self.save_tracks(tracks)
                return
        tracks.append(track)
        self.save_tracks(tracks)

    def get_track(self, track_id: str) -> LogicalTrack | None:
        for t in self.load_tracks():
            if t.track_id == track_id:
                return t
        return None

    # -- Files ---------------------------------------------------------------

    def load_files(self) -> list[FileRecord]:
        return self._load("files_master.csv", FileRecord)

    def save_files(self, files: list[FileRecord]) -> None:
        self._save("files_master.csv", files, FileRecord)

    def upsert_file(self, file_rec: FileRecord) -> None:
        """Insert or update a file record by file_id."""
        files = self.load_files()
        for i, f in enumerate(files):
            if f.file_id == file_rec.file_id:
                files[i] = file_rec
                self.save_files(files)
                return
        files.append(file_rec)
        self.save_files(files)

    def get_file(self, file_id: str) -> FileRecord | None:
        for f in self.load_files():
            if f.file_id == file_id:
                return f
        return None

    def get_file_by_path(self, path_abs: str) -> FileRecord | None:
        for f in self.load_files():
            if f.path_abs == path_abs:
                return f
        return None

    # -- Observations --------------------------------------------------------

    def load_observations(self) -> list[SourceObservation]:
        return self._load("source_observations.csv", SourceObservation)

    def save_observations(self, obs: list[SourceObservation]) -> None:
        self._save("source_observations.csv", obs, SourceObservation)

    def add_observation(self, obs: SourceObservation) -> None:
        all_obs = self.load_observations()
        all_obs.append(obs)
        self.save_observations(all_obs)

    def add_observations(self, new_obs: list[SourceObservation]) -> None:
        all_obs = self.load_observations()
        all_obs.extend(new_obs)
        self.save_observations(all_obs)

    def get_observations_for_track(
        self, track_id: str, source_system: str | None = None
    ) -> list[SourceObservation]:
        result = []
        for o in self.load_observations():
            if o.track_id == track_id:
                if source_system is None or o.source_system == source_system:
                    result.append(o)
        return result

    def delete_observations(
        self, track_id: str, source_system: str | None = None
    ) -> int:
        """Delete observations for a track (optionally filtered by source). Returns count deleted."""
        all_obs = self.load_observations()
        keep = []
        deleted = 0
        for o in all_obs:
            if o.track_id == track_id and (source_system is None or o.source_system == source_system):
                deleted += 1
            else:
                keep.append(o)
        if deleted:
            self.save_observations(keep)
        return deleted

    # -- Review Queue --------------------------------------------------------

    def load_review_queue(self) -> list[ReviewItem]:
        return self._load("review_queue.csv", ReviewItem)

    def save_review_queue(self, items: list[ReviewItem]) -> None:
        self._save("review_queue.csv", items, ReviewItem)

    # -- Payload Index -------------------------------------------------------

    def load_payload_index(self) -> list[PayloadIndexEntry]:
        return self._load("source_payload_index.csv", PayloadIndexEntry)

    def save_payload_index(self, entries: list[PayloadIndexEntry]) -> None:
        self._save("source_payload_index.csv", entries, PayloadIndexEntry)

    def add_payload_entry(self, entry: PayloadIndexEntry) -> None:
        entries = self.load_payload_index()
        entries.append(entry)
        self.save_payload_index(entries)
