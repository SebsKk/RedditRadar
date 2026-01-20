"""Report generator module for RedditRadar.

Generates structured markdown reports with:
1. Radar section (New/Repeating/Rising)
2. Cluster Summary with metrics
3. Top 3 Items with scorecards
4. Evidence & Links
5. Next Actions
"""

import json
import logging
from dataclasses import dataclass
from typing import Optional

from src.analysis.clustering import (
    ClusterResult,
    PostFeatures,
    compute_post_features,
    aggregate_cluster_metrics,
    format_cluster_summary,
    intensity_label,
)
from src.analysis.scoring import (
    TemplateWeights,
    get_template_weights,
    compute_base_score,
    rank_by_cluster,
    generate_scorecards,
    format_scorecards,
    Scorecard,
    score_clusters_with_llm,
)
from src.analysis.radar import (
    RadarSection,
    compute_radar,
    format_radar_section,
)

logger = logging.getLogger(__name__)


@dataclass
class StructuredReport:
    """Complete structured report data."""
    template_id: str
    template_name: str

    # Radar
    radar: RadarSection

    # Clusters
    clusters: list[ClusterResult]

    # Top items with scorecards
    scorecards: list[Scorecard]

    # Post features (for reference)
    post_features: list[PostFeatures]

    # Full markdown
    markdown: str

    # So what summary (1-liner actionable insight)
    so_what: str = ""


def truncate_text(text: str, max_chars: int = 500) -> str:
    """Truncate text to max chars, preserving word boundaries."""
    if not text or len(text) <= max_chars:
        return text or ""
    # Find last space before max_chars
    truncated = text[:max_chars]
    last_space = truncated.rfind(' ')
    if last_space > max_chars * 0.8:
        truncated = truncated[:last_space]
    return truncated + "..."


def prepare_posts_for_analysis(
    posts: list[dict],
    comments_by_post: dict[str, list[dict]],
    max_body_chars: int = 500,
    max_comment_chars: int = 250,
    max_comments: int = 5,
) -> list[dict]:
    """Prepare posts for analysis with truncation for token control.

    Args:
        posts: List of post dicts with keys: post_id, title, body, score, num_comments, created_utc
        comments_by_post: Dict mapping post_id to list of comment dicts
        max_body_chars: Max chars for post body
        max_comment_chars: Max chars per comment
        max_comments: Max comments to include per post

    Returns:
        List of prepared post dicts
    """
    prepared = []
    for post in posts:
        # Truncate body
        body = truncate_text(post.get("body", ""), max_body_chars)

        # Get and truncate comments
        post_comments = comments_by_post.get(post["post_id"], [])[:max_comments]
        comments_text = " ".join(
            truncate_text(c.get("body", ""), max_comment_chars)
            for c in post_comments
        )

        prepared.append({
            **post,
            "body_truncated": body,
            "comments_text": comments_text,
        })

    return prepared


def analyze_posts(
    posts: list[dict],
    goal_prompt: str,
    template_id: str,
) -> tuple[list[PostFeatures], list[ClusterResult]]:
    """Analyze posts and compute features and clusters.

    Args:
        posts: Prepared posts with body_truncated and comments_text
        goal_prompt: The goal prompt for keyword matching
        template_id: Template ID for weight selection

    Returns:
        Tuple of (post_features, cluster_results)
    """
    logger.info(f"[Analysis] Computing features for {len(posts)} posts...")

    # Compute features for each post
    post_features = []
    for post in posts:
        features = compute_post_features(
            post_id=post["post_id"],
            title=post.get("title", ""),
            body=post.get("body_truncated", ""),
            comments_text=post.get("comments_text", ""),
            goal_prompt=goal_prompt,
            template_id=template_id,
            permalink=post.get("url", ""),  # Pass Reddit permalink
        )
        post_features.append(features)

    logger.info(f"[Analysis] Computed features for {len(post_features)} posts")

    # Prepare posts_data for aggregation (includes url, subreddit, title, body for key_signals)
    posts_data = [
        {
            "post_id": p["post_id"],
            "score": p.get("score", 0),
            "num_comments": p.get("num_comments", 0),
            "created_utc": p.get("created_utc", 0),
            "url": p.get("url", ""),  # Reddit permalink
            "subreddit": p.get("subreddit", ""),
            "title": p.get("title", ""),
            "body_truncated": p.get("body_truncated", ""),
        }
        for p in posts
    ]

    # Aggregate into clusters
    clusters = aggregate_cluster_metrics(post_features, posts_data)
    logger.info(f"[Analysis] Aggregated into {len(clusters)} clusters")

    # Log top clusters
    for c in clusters[:5]:
        logger.info(f"[Analysis]   {c.cluster_name}: {c.frequency_posts} posts, {c.engagement_score_sum} score")

    return post_features, clusters


def apply_llm_value_scores(
    clusters: list[ClusterResult],
    goal_prompt: str,
    template_id: str,
) -> list[ClusterResult]:
    """Apply LLM-based value scores to clusters for heatmap visualization.

    Args:
        clusters: List of cluster results
        goal_prompt: The user's goal/use case
        template_id: Template ID for context

    Returns:
        Updated clusters with value_score set
    """
    logger.info("[Analysis] Scoring clusters with LLM for heatmap...")

    scores = score_clusters_with_llm(clusters, goal_prompt, template_id)

    # Update clusters with scores (create new objects since dataclass is immutable)
    from dataclasses import replace

    updated = []
    for c in clusters:
        score = scores.get(c.cluster_id, 0.0)
        # Use dataclass replace to update value_score
        updated.append(ClusterResult(
            cluster_id=c.cluster_id,
            cluster_name=c.cluster_name,
            frequency_posts=c.frequency_posts,
            frequency_comments=c.frequency_comments,
            engagement_score_sum=c.engagement_score_sum,
            engagement_comments_sum=c.engagement_comments_sum,
            avg_age_hours=c.avg_age_hours,
            intensity_score=c.intensity_score,
            proxy_score=c.proxy_score,
            key_signals=c.key_signals,
            top_urls=c.top_urls,
            value_score=score,
        ))
        logger.debug(f"[Analysis]   {c.cluster_name}: value_score={score}")

    logger.info(f"[Analysis] Applied value scores to {len(updated)} clusters")
    return updated


def generate_structured_report(
    posts: list[dict],
    goal_prompt: str,
    template_id: str,
    historical_clusters: list[dict] = None,
    llm_content: str = "",
) -> StructuredReport:
    """Generate a complete structured report.

    Args:
        posts: Prepared posts with body_truncated and comments_text
        goal_prompt: The goal prompt
        template_id: Template ID
        historical_clusters: Historical cluster data for Radar
        llm_content: Optional LLM-generated content to include

    Returns:
        StructuredReport with all sections
    """
    # Get template weights
    weights = get_template_weights(template_id)
    logger.info(f"[Report] Generating report for template: {weights.template_name}")

    # Analyze posts
    post_features, clusters = analyze_posts(posts, goal_prompt, template_id)

    # Compute Radar
    radar = compute_radar(
        current_clusters=clusters,
        historical_clusters=historical_clusters or [],
        days=7,
        min_repeats=2,
    )

    # Rank clusters
    top_clusters = rank_by_cluster(clusters, weights, top_n=5)

    # Generate scorecards
    scorecards = generate_scorecards(top_clusters, post_features, weights, top_n=3)

    # Generate "So what?" actionable summary
    so_what = generate_so_what_summary(
        scorecards=scorecards,
        clusters=clusters,
        goal_prompt=goal_prompt,
        template_id=template_id,
    )

    # Build markdown
    markdown = build_markdown_report(
        radar=radar,
        clusters=clusters,
        scorecards=scorecards,
        weights=weights,
        llm_content=llm_content,
        so_what=so_what,
    )

    return StructuredReport(
        template_id=template_id,
        template_name=weights.template_name,
        radar=radar,
        clusters=clusters,
        scorecards=scorecards,
        post_features=post_features,
        markdown=markdown,
        so_what=so_what,
    )


def build_markdown_report(
    radar: RadarSection,
    clusters: list[ClusterResult],
    scorecards: list[Scorecard],
    weights: TemplateWeights,
    llm_content: str = "",
    so_what: str = "",
) -> str:
    """Build the final markdown report.

    Sections:
    0. So What? (actionable 1-liner)
    1. Radar (New/Repeating/Rising)
    2. Cluster Summary
    3. Top 3 Items with scorecards
    4. LLM Analysis (if provided)
    5. Evidence & Links
    6. Next Actions
    """
    sections = []

    # 0. So What? (actionable 1-liner at the very top)
    if so_what:
        sections.append("## So What?\n")
        sections.append(f"*{so_what}*\n")

    # 1. Radar
    sections.append(format_radar_section(radar))

    # 2. Cluster Summary
    sections.append(format_cluster_summary(clusters))

    # 3. Top Items with Scorecards
    sections.append(format_scorecards(scorecards, weights.output_name))

    # 4. LLM Analysis (if provided)
    if llm_content:
        sections.append("## Detailed Analysis\n")
        sections.append(llm_content)
        sections.append("")

    # 5. Evidence & Links
    sections.append(format_evidence_section(clusters))

    # 6. Next Actions
    sections.append(format_next_actions(scorecards, weights))

    return "\n".join(sections)


def format_evidence_section(clusters: list[ClusterResult]) -> str:
    """Format evidence and links section."""
    lines = ["## Evidence & Links\n"]

    # Collect all URLs by cluster
    url_by_cluster = {}
    for c in clusters:
        if c.top_urls:
            url_by_cluster[c.cluster_name] = c.top_urls

    if not url_by_cluster:
        lines.append("*No external links found in analyzed posts.*\n")
        return "\n".join(lines)

    for cluster_name, urls in url_by_cluster.items():
        lines.append(f"**{cluster_name}:**")
        for url in urls[:3]:
            # Truncate URL for display
            display_url = url[:60] + "..." if len(url) > 60 else url
            lines.append(f"- [{display_url}]({url})")
        lines.append("")

    return "\n".join(lines)


def _get_action_insight(sc: Scorecard, weights: TemplateWeights) -> str:
    """Generate a specific, contextual action insight based on scorecard metrics.

    Uses the combination of signal_strength, impact_wtp, complexity, reachability,
    and moat to provide actionable recommendations specific to each item.
    """
    # Determine the opportunity profile
    is_quick_win = sc.signal_strength >= 4 and sc.complexity <= 2
    is_high_value = sc.impact_wtp >= 4 and sc.moat >= 3
    is_accessible = sc.reachability >= 4 and sc.complexity <= 3
    is_complex_but_worthy = sc.signal_strength >= 4 and sc.complexity >= 4
    is_emerging = sc.signal_strength <= 3 and sc.impact_wtp >= 3

    # Item type specific prefixes
    type_actions = {
        "software": "build a simple MVP",
        "info-product": "create a guide or course outline",
        "community": "engage with the community or create content",
        "workflow": "document the workflow and create templates",
    }

    action_verb = type_actions.get(sc.item_type, "explore this opportunity")

    # Generate contextual recommendation based on profile
    if is_quick_win:
        return f"**Quick win** - High demand ({sc.signal_strength}★) with low complexity. {action_verb.capitalize()} within 2-4 weeks to validate."

    if is_high_value and is_accessible:
        return f"**Strategic priority** - Strong monetization potential (WTP: {sc.impact_wtp}★) and good defensibility. Invest time to {action_verb}."

    if is_accessible and not is_complex_but_worthy:
        return f"**Low-hanging fruit** - Easy to reach audience (reach: {sc.reachability}★). Test with a minimal version first."

    if is_complex_but_worthy:
        return f"**Worth the investment** - Strong signal ({sc.signal_strength}★) but requires planning (complexity: {sc.complexity}★). Break into phases."

    if is_emerging:
        return f"**Emerging opportunity** - Signal building (WTP: {sc.impact_wtp}★). Monitor for 2-4 weeks before committing."

    if sc.moat >= 4:
        return f"**Defensible niche** - High moat ({sc.moat}★) means less competition. Worth deeper research."

    # Fallback based on strongest dimension
    if sc.signal_strength >= 4:
        return f"Strong demand signal ({sc.signal_strength}★). Validate with quick experiments."
    if sc.impact_wtp >= 4:
        return f"Good monetization potential (WTP: {sc.impact_wtp}★). Research pricing models."
    if sc.reachability >= 4:
        return f"Accessible market (reach: {sc.reachability}★). Start with content or outreach."

    return f"Monitor this topic. Current signal: {sc.signal_strength}★, complexity: {sc.complexity}★."


def format_next_actions(scorecards: list[Scorecard], weights: TemplateWeights) -> str:
    """Format next actions based on scorecard metrics.

    Generates contextual, specific recommendations based on the actual
    scorecard dimensions (signal_strength, impact_wtp, complexity, etc.)
    rather than generic template-based advice.
    """
    lines = ["## Next Actions\n"]

    if not scorecards:
        lines.append("*Run more analyses to generate actionable insights.*\n")
        return "\n".join(lines)

    # Group by action priority
    quick_wins = [sc for sc in scorecards if sc.signal_strength >= 4 and sc.complexity <= 2]
    strategic = [sc for sc in scorecards if sc not in quick_wins and sc.impact_wtp >= 4]
    monitor = [sc for sc in scorecards if sc not in quick_wins and sc not in strategic]

    # Header based on template context
    context_headers = {
        "content_ideas": "Content creation priorities",
        "trend_radar": "Trend response actions",
        "career_intel": "Skill development priorities",
        "industry_intel": "Strategic responses",
        "deep_research": "Research follow-ups",
    }
    header = context_headers.get(weights.template_id, "Recommended actions")
    lines.append(f"**{header}:**\n")

    # Generate specific actions for top 3
    for i, sc in enumerate(scorecards[:3], 1):
        insight = _get_action_insight(sc, weights)
        lines.append(f"{i}. **{sc.title}** ({sc.cluster})")
        lines.append(f"   {insight}")

        # Add key insight if available
        if sc.key_insight:
            lines.append(f"   *Insight: {sc.key_insight[:150]}{'...' if len(sc.key_insight) > 150 else ''}*")
        lines.append("")

    # Summary by priority if we have enough items
    if len(scorecards) > 3:
        lines.append("**Priority summary:**")
        if quick_wins:
            lines.append(f"- Quick wins ({len(quick_wins)}): {', '.join(sc.title for sc in quick_wins[:3])}")
        if strategic:
            lines.append(f"- Strategic bets ({len(strategic)}): {', '.join(sc.title for sc in strategic[:3])}")
        if monitor:
            lines.append(f"- Monitor ({len(monitor)}): {', '.join(sc.title for sc in monitor[:3])}")
        lines.append("")

    return "\n".join(lines)


def generate_so_what_summary(
    scorecards: list[Scorecard],
    clusters: list[ClusterResult],
    goal_prompt: str,
    template_id: str,
) -> str:
    """Generate a 1-liner 'So what?' actionable summary using LLM.

    This summary distills the entire analysis into a single actionable insight
    that tells the user what to do with the findings.

    Args:
        scorecards: Top scorecards from analysis
        clusters: Cluster results
        goal_prompt: User's goal/use case
        template_id: Template ID for context

    Returns:
        1-2 sentence actionable summary, or empty string on failure
    """
    try:
        from src.llm_client import get_llm_client, LLMClientError
    except ImportError:
        logger.warning("[SoWhat] LLM client not available")
        return ""

    if not scorecards:
        return ""

    # Build context for LLM
    top_items = []
    for sc in scorecards[:3]:
        top_items.append(
            f"- {sc.title} ({sc.cluster}): signal={sc.signal_strength}★, "
            f"WTP={sc.impact_wtp}★, complexity={sc.complexity}★"
        )

    # Get rising/new clusters for context
    cluster_context = []
    for c in clusters[:5]:
        signals = ", ".join(c.key_signals[:3]) if c.key_signals else "mixed"
        cluster_context.append(f"- {c.cluster_name}: {c.frequency_posts} posts, signals: {signals}")

    prompt = f"""Based on this Reddit analysis, write ONE actionable 1-2 sentence "So what?" summary.

USER'S GOAL: {goal_prompt}

TOP OPPORTUNITIES:
{chr(10).join(top_items)}

TOPIC CLUSTERS:
{chr(10).join(cluster_context)}

The summary should:
1. Identify the STRONGEST actionable theme/opportunity
2. Give ONE specific validation step or quick win
3. Be concrete, not generic - use actual topics from the analysis

Example format:
"This week's strongest theme: [theme]. Quick validation: [specific action]."

Write ONLY the 1-2 sentence summary, nothing else:"""

    try:
        client = get_llm_client()
        response = client.complete(
            prompt=prompt,
            max_tokens=150,
            temperature=0.3,  # Lower temperature for concise output
        )

        # Clean up response
        summary = response.strip()
        # Remove quotes if LLM wrapped it
        if summary.startswith('"') and summary.endswith('"'):
            summary = summary[1:-1]
        # Ensure it's not too long
        if len(summary) > 300:
            summary = summary[:297] + "..."

        logger.info(f"[SoWhat] Generated summary: {summary[:100]}...")
        return summary

    except Exception as e:
        logger.warning(f"[SoWhat] Failed to generate summary: {e}")
        return ""


def clusters_to_db_format(clusters: list[ClusterResult], run_id: str) -> list[dict]:
    """Convert ClusterResult to database format.

    Returns list of dicts ready for RunCluster dataclass.
    """
    from src.database import RunCluster

    return [
        RunCluster(
            run_id=run_id,
            cluster_id=c.cluster_id,  # Stable ID for radar comparison
            cluster_name=c.cluster_name,
            frequency_posts=c.frequency_posts,
            engagement_score_sum=c.engagement_score_sum,
            engagement_comments_sum=c.engagement_comments_sum,
            avg_age_hours=c.avg_age_hours,
            intensity_score=c.intensity_score,
            proxy_score=c.proxy_score,
            top_urls_json=json.dumps(c.top_urls),
            value_score=c.value_score,  # LLM-assigned value for heatmap
        )
        for c in clusters
    ]


def format_posts_for_llm_structured(
    posts: list[dict],
    post_features: list[PostFeatures],
    clusters: list[ClusterResult],
    max_posts: int = 15,
) -> str:
    """Format posts for LLM with cluster context for better analysis.

    This provides structured context to the LLM about clusters and top posts.
    """
    lines = []

    # Cluster summary for context
    lines.append("## Cluster Context\n")
    for c in clusters[:5]:
        lines.append(f"- **{c.cluster_name}**: {c.frequency_posts} posts, intensity={intensity_label(c.intensity_score)}")
    lines.append("")

    # Build feature lookup
    features_by_id = {pf.post_id: pf for pf in post_features}

    # Group posts by cluster
    posts_by_cluster = {}
    for post in posts:
        pf = features_by_id.get(post["post_id"])
        if pf:
            cluster = pf.cluster_name
            if cluster not in posts_by_cluster:
                posts_by_cluster[cluster] = []
            posts_by_cluster[cluster].append((post, pf))

    # Format top posts per cluster
    lines.append("## Top Posts by Cluster\n")
    post_count = 0
    for cluster_name, cluster_posts in posts_by_cluster.items():
        if post_count >= max_posts:
            break

        # Sort by base score
        cluster_posts.sort(key=lambda x: x[1].base_score, reverse=True)

        lines.append(f"### {cluster_name}")
        for post, pf in cluster_posts[:3]:
            if post_count >= max_posts:
                break

            lines.append(f"\n**{post.get('title', 'Untitled')}** (r/{post.get('subreddit', 'unknown')})")
            lines.append(f"Score: {post.get('score', 0)} | Comments: {post.get('num_comments', 0)}")
            lines.append(f"Keywords: {', '.join(pf.keywords[:5])}")

            body = post.get("body_truncated", "")
            if body:
                lines.append(f"\n{body}\n")

            comments = post.get("comments_text", "")
            if comments:
                lines.append(f"*Top comments:* {comments[:300]}...\n")

            post_count += 1

        lines.append("")

    return "\n".join(lines)
