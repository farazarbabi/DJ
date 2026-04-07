"""Tests for CSV export."""

import csv
from pathlib import Path

import numpy as np

from dj_grouper.features.builder import TrackFeatures
from dj_grouper.scanner import TrackInfo
from dj_grouper.grouping.assignment import GroupInfo, GroupAssignment
from dj_grouper.output.csv_export import export_groups_csv, export_recommendations_csv
from dj_grouper.recommend.neighbors import Recommendation


def test_groups_csv_export(tmp_path):
    info = TrackInfo(path="/music/a.aiff", energy=3, key="9A", bpm=126,
                     structure="64H", vibe="HYPN", vocal="NV")
    tracks = [TrackFeatures("/music/a.aiff", info, np.zeros(19), np.zeros(21))]
    group = GroupInfo(
        group_id="G001", member_indices=[0], medoid_index=0,
        key="9A", energy=3, vibe="HYPN", structure="64H",
        vocal="NV", bpm=126, folder_name="9A_E3_HYPN_64H_NV_126",
    )
    assignment = GroupAssignment(groups=[group], track_to_group={"/music/a.aiff": "G001"})

    csv_path = str(tmp_path / "outputs" / "groups.csv")
    export_groups_csv(tracks, assignment, csv_path)

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 1
    assert rows[0]["group_id"] == "G001"
    assert rows[0]["vibe"] == "HYPN"


def test_recommendations_csv_export(tmp_path):
    recs = [
        Recommendation("a.aiff", "b.aiff", 1, 0.87, 0, 2, True, True),
        Recommendation("a.aiff", "c.aiff", 2, 0.79, 1, 1, True, False),
    ]
    csv_path = str(tmp_path / "outputs" / "recs.csv")
    export_recommendations_csv(recs, csv_path)

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 2
    assert "mode" not in rows[0]  # no mode column
    assert rows[0]["rank"] == "1"
    assert rows[1]["energy_delta"] == "1"
