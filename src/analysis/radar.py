"""Radar module for RedditRadar.

Tracks cluster patterns over time:
- New: clusters in current run but NOT in previous run
- Continuing: clusters in current run AND previous run
- Repeating: clusters appearing in >=2 of last 3 runs
- Rising: clusters with increasing frequency or engagement vs previous run
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from src.analysis.clustering import ClusterResult

logger = logging.getLogger(__name__)


@dataclass
class RisingDelta:
    """Delta information for a rising cluster."""
    freq_delta: int  # +/- posts
    engagement_delta: int  # +/- engagement score
    freq_prev: int
    engagement_prev: int


@dataclass
class RepeatingInfo:
    """Info about a repeating cluster."""
    appearances: int  # Number of runs it appeared in
    total_runs: int  # Total runs in window (e.g., 3)


@dataclass
class RadarSection:
    """Radar analysis results."""
    # Cluster categories
    new_clusters: list[str]  # Cluster names new today (vs previous run)
    continuing_clusters: list[str]  # Clusters in current AND previous run
    repeating_clusters: list[str]  # Clusters appearing in >=2 of last N runs
    rising_clusters: list[str]  # Clusters with rising engagement/freq vs previous run

    # Detailed info for display
    rising_info: dict[str, RisingDelta] = field(default_factory=dict)
    repeating_info: dict[str, RepeatingInfo] = field(default_factory=dict)

    # Historical data used
    days_analyzed: int = 7
    runs_analyzed: int = 0

    # Insufficient history flags
    insufficient_history: bool = False
    insufficient_for_rising: bool = False


def compute_radar(
    current_clusters: list[ClusterResult],
    historical_clusters: list[dict],  # From DB: [{run_id, cluster_name, frequency_posts, engagement_score_sum, ...}]
    days: int = 7,
    repeating_window: int = 3,  # Look at last N runs for repeating
    min_repeats: int = 2,  # Min appearances to be "repeating"
    min_runs_for_radar: int = 2,  # Min runs needed for New/Continuing
    min_runs_for_rising: int = 2,  # Min runs needed for Rising/Repeating
) -> RadarSection:
    """Compute radar section by comparing current run to history.

    Definitions:
    - New: clusters in current_run but NOT in prev_run
    - Continuing: clusters in current_run AND prev_run
    - Repeating: clusters in >= min_repeats of last repeating_window runs
    - Rising: clusters where (freq_delta >= 2) OR (engagement_delta >= 500) OR (engagement_ratio >= 1.5)

    Args:
        current_clusters: Cluster results from current run
        historical_clusters: Historical cluster data from database
        days: Number of days to look back
        repeating_window: Number of recent runs to check for repeating
        min_repeats: Minimum appearances to be considered "repeating"
        min_runs_for_radar: Minimum runs for New/Continuing
        min_runs_for_rising: Minimum runs for Rising/Repeating calculations

    Returns:
        RadarSection with categorized clusters
    """
    # Get unique run IDs from history, sorted by time (assuming run_id is sortable)
    run_ids = sorted(set(hc.get("run_id", "") for hc in historical_clusters if hc.get("run_id")))
    runs_analyzed = len(run_ids)

    logger.info(f"[Radar] Analyzing {runs_analyzed} historical runs")

    # Check if we have minimum history
    if runs_analyzed < min_runs_for_radar:
        logger.info(f"[Radar] Insufficient history ({runs_analyzed} < {min_runs_for_radar} runs)")
        return RadarSection(
            new_clusters=[],
            continuing_clusters=[],
            repeating_clusters=[],
            rising_clusters=[],
            days_analyzed=days,
            runs_analyzed=runs_analyzed,
            insufficient_history=True,
            insufficient_for_rising=True,
        )

    # Get current cluster IDs with posts (use cluster_id for stable comparison)
    current_ids = {c.cluster_id for c in current_clusters if c.frequency_posts > 0}
    current_by_id = {c.cluster_id: c for c in current_clusters}

    # Group historical data by run_id, keyed by cluster_id (stable) or cluster_name (fallback)
    clusters_by_run: dict[str, dict[str, dict]] = {}
    for hc in historical_clusters:
        run_id = hc.get("run_id", "")
        # Prefer cluster_id, fallback to cluster_name for old data
        cid = hc.get("cluster_id") or hc.get("cluster_name", "")
        if not run_id or not cid:
            continue

        if run_id not in clusters_by_run:
            clusters_by_run[run_id] = {}
        clusters_by_run[run_id][cid] = hc

    # Get previous run (most recent in history)
    previous_run_id = run_ids[-1] if run_ids else None
    previous_clusters = clusters_by_run.get(previous_run_id, {}) if previous_run_id else {}
    previous_ids = set(previous_clusters.keys())

    logger.info(f"[Radar] Previous run: {previous_run_id}, clusters: {len(previous_ids)}")
    logger.info(f"[Radar] Current IDs: {current_ids}")
    logger.info(f"[Radar] Previous IDs: {previous_ids}")

    # =========================================================================
    # NEW: in current but NOT in previous run
    # =========================================================================
    new_cluster_ids = [cid for cid in current_ids if cid not in previous_ids]
    # Map IDs to display names for output
    new_clusters = [current_by_id[cid].cluster_name for cid in new_cluster_ids if cid in current_by_id]
    logger.info(f"[Radar] New clusters: {new_clusters}")

    # =========================================================================
    # CONTINUING: in current AND previous run
    # =========================================================================
    continuing_ids = [cid for cid in current_ids if cid in previous_ids]
    continuing_clusters = [current_by_id[cid].cluster_name for cid in continuing_ids if cid in current_by_id]
    logger.info(f"[Radar] Continuing clusters: {continuing_clusters}")

    # =========================================================================
    # REPEATING: appeared in >= min_repeats of last repeating_window runs
    # =========================================================================
    repeating_clusters = []
    repeating_info = {}

    # Get the last N run IDs for the window
    recent_run_ids = run_ids[-repeating_window:] if len(run_ids) >= repeating_window else run_ids
    total_runs_in_window = len(recent_run_ids)

    if total_runs_in_window >= min_runs_for_rising:
        # Count appearances in window by cluster_id
        cluster_appearances_in_window: dict[str, int] = {}
        for rid in recent_run_ids:
            run_clusters = clusters_by_run.get(rid, {})
            for cid in run_clusters.keys():
                cluster_appearances_in_window[cid] = cluster_appearances_in_window.get(cid, 0) + 1

        logger.info(f"[Radar] Cluster appearances in window: {cluster_appearances_in_window}")

        # Clusters that are repeating AND present now (by cluster_id)
        for cid, count in cluster_appearances_in_window.items():
            if count >= min_repeats and cid in current_ids:
                # Get display name
                display_name = current_by_id[cid].cluster_name if cid in current_by_id else cid
                repeating_clusters.append(display_name)
                repeating_info[display_name] = RepeatingInfo(
                    appearances=count,
                    total_runs=total_runs_in_window,
                )

        logger.info(f"[Radar] Repeating clusters (>={min_repeats} in last {total_runs_in_window}): {repeating_clusters}")
    else:
        logger.info(f"[Radar] Insufficient runs for repeating calculation ({total_runs_in_window} < {min_runs_for_rising})")

    # =========================================================================
    # RISING: clusters with significant delta vs previous run
    # Rising if: (freq_delta >= 2) OR (engagement_delta >= 500) OR (engagement_ratio >= 1.5)
    # =========================================================================
    rising_clusters = []
    rising_info = {}
    insufficient_for_rising = runs_analyzed < min_runs_for_rising

    if not insufficient_for_rising:
        for c in current_clusters:
            # Compare by cluster_id
            if c.cluster_id not in previous_clusters:
                continue  # Can only compare if in previous run

            prev = previous_clusters[c.cluster_id]

            current_freq = c.frequency_posts
            prev_freq = prev.get("frequency_posts", 0)
            freq_delta = current_freq - prev_freq

            current_engagement = c.engagement_score_sum + c.engagement_comments_sum
            prev_engagement = prev.get("engagement_score_sum", 0) + prev.get("engagement_comments_sum", 0)
            engagement_delta = current_engagement - prev_engagement

            # Check rising criteria
            is_rising = False
            if freq_delta >= 2:
                is_rising = True
                logger.info(f"[Radar] Rising (freq): {c.cluster_name} +{freq_delta} posts")
            elif engagement_delta >= 500:
                is_rising = True
                logger.info(f"[Radar] Rising (engagement): {c.cluster_name} +{engagement_delta} score")
            elif prev_engagement > 0 and current_engagement / prev_engagement >= 1.5:
                is_rising = True
                logger.info(f"[Radar] Rising (ratio): {c.cluster_name} {current_engagement/prev_engagement:.1f}x")

            if is_rising:
                rising_clusters.append(c.cluster_name)
                rising_info[c.cluster_name] = RisingDelta(
                    freq_delta=freq_delta,
                    engagement_delta=engagement_delta,
                    freq_prev=prev_freq,
                    engagement_prev=prev_engagement,
                )

        # Sort rising by composite delta score
        def rising_score(name: str) -> float:
            info = rising_info.get(name)
            if not info:
                return 0
            # Normalize and combine freq + engagement deltas
            freq_score = min(info.freq_delta / 5, 1.0) if info.freq_delta > 0 else 0
            eng_score = min(info.engagement_delta / 2000, 1.0) if info.engagement_delta > 0 else 0
            return 0.6 * freq_score + 0.4 * eng_score

        rising_clusters.sort(key=rising_score, reverse=True)
        logger.info(f"[Radar] Rising clusters: {rising_clusters[:5]}")

    return RadarSection(
        new_clusters=new_clusters[:5],
        continuing_clusters=continuing_clusters[:5],
        repeating_clusters=repeating_clusters[:5],
        rising_clusters=rising_clusters[:5],
        rising_info=rising_info,
        repeating_info=repeating_info,
        days_analyzed=days,
        runs_analyzed=runs_analyzed,
        insufficient_history=False,
        insufficient_for_rising=insufficient_for_rising,
    )


def _format_engagement(value: int) -> str:
    """Format engagement number for display (e.g., 1.2k, 15k)."""
    if value >= 1000:
        return f"{value/1000:.1f}k"
    return str(value)


def format_radar_section(radar: RadarSection) -> str:
    """Format radar section as markdown."""
    lines = ["## Radar\n"]

    # Handle insufficient history
    if radar.insufficient_history:
        runs_needed = 2 - radar.runs_analyzed
        lines.append("*Insufficient history for radar analysis.*")
        lines.append(f"*Currently have {radar.runs_analyzed} run(s). Need at least 2 runs to detect patterns.*")
        if runs_needed > 0:
            lines.append(f"*Run {runs_needed} more analysis(es) to enable radar tracking.*\n")
        return "\n".join(lines)

    lines.append(f"*Based on {radar.runs_analyzed} runs over last {radar.days_analyzed} days*\n")

    # New (vs previous run)
    if radar.new_clusters:
        lines.append("**New (vs previous run):**")
        for name in radar.new_clusters:
            lines.append(f"- {name}")
        lines.append("")
    else:
        lines.append("**New:** None\n")

    # Continuing (still active)
    if radar.continuing_clusters:
        lines.append("**Continuing (still active):**")
        for name in radar.continuing_clusters:
            lines.append(f"- {name}")
        lines.append("")
    else:
        lines.append("**Continuing:** None\n")

    # Repeating (persistent)
    if radar.repeating_clusters:
        lines.append("**Repeating (persistent):**")
        for name in radar.repeating_clusters:
            info = radar.repeating_info.get(name)
            if info:
                lines.append(f"- {name} (seen in {info.appearances}/{info.total_runs} runs)")
            else:
                lines.append(f"- {name}")
        lines.append("")
    elif radar.insufficient_for_rising:
        lines.append("**Repeating:** *Insufficient history*\n")
    else:
        lines.append("**Repeating:** None\n")

    # Rising (accelerating)
    if radar.rising_clusters:
        lines.append("**Rising (accelerating):**")
        for name in radar.rising_clusters:
            info = radar.rising_info.get(name)
            if info:
                parts = []
                if info.freq_delta != 0:
                    sign = "+" if info.freq_delta > 0 else ""
                    parts.append(f"{sign}{info.freq_delta} posts")
                if info.engagement_delta != 0:
                    sign = "+" if info.engagement_delta > 0 else ""
                    parts.append(f"{sign}{_format_engagement(info.engagement_delta)} score")
                delta_str = ", ".join(parts) if parts else "rising"
                lines.append(f"- {name} ({delta_str})")
            else:
                lines.append(f"- {name}")
        lines.append("")
    elif radar.insufficient_for_rising:
        lines.append("**Rising:** *Insufficient history*\n")
    else:
        lines.append("**Rising:** None\n")

    return "\n".join(lines)
