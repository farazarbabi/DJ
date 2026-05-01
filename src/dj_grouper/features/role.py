"""Track role inference: TOOL, DRIVER, PEAK, RESET, BREAKDOWN, BRIDGE."""

from __future__ import annotations

from dj_tagger.moods import normalize_mood_code

from ..scanner import TrackInfo

ROLES = ["TOOL", "DRIVER", "PEAK", "RESET", "BREAKDOWN", "BRIDGE"]


def infer_role(info: TrackInfo, dsp: dict[str, float]) -> str:
    """Infer a practical DJ role from tags and DSP features.

    Roles:
        TOOL      - E1-E2, sparse, layer/loop tool
        DRIVER    - E3, steady groove, main set workhorse
        PEAK      - E4-E5, high density, peak-time track
        RESET     - E1-E2, atmospheric/deep vibe, used to bring energy down
        BREAKDOWN - breakdown-heavy structure
        BRIDGE    - moderate energy, versatile vibe, connects different moods
    """
    energy = info.energy or 3
    flow = info.flow_type or "H"
    vibe = normalize_mood_code(info.vibe) or "HYPN"
    dsp.get("onset_density", 0.0)
    dsp.get("rms_mean", 0.0)

    # Breakdown-heavy structure
    if flow == "B":
        return "BREAKDOWN"

    # Layer tool: sparse, low energy, loop-based
    if energy <= 2 and flow == "L":
        return "TOOL"

    # Reset: low energy + atmospheric/deep vibe
    if energy <= 2 and vibe in ("ATM", "CIN", "DEEP", "SUB", "SUN"):
        return "RESET"

    # Tool: low energy + sparse onset
    if energy <= 2:
        return "TOOL"

    # Peak: high energy
    if energy >= 4:
        return "PEAK"

    # Bridge: E3 with melodic or atmospheric qualities, or mixed vibe signals
    if energy == 3 and vibe in ("MEL", "EMO", "EUP", "SOUL", "WARM", "ATM", "CIN", "DEEP", "SUN"):
        return "BRIDGE"

    # Driver: E3, steady groove (default for E3)
    return "DRIVER"
