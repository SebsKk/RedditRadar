"""Scoring module for RedditRadar.

Provides template-agnostic scoring with configurable weights
and scorecard generation.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from src.analysis.clustering import PostFeatures, ClusterResult

logger = logging.getLogger(__name__)


# =============================================================================
# TEMPLATE WEIGHTS CONFIGURATION
# =============================================================================

@dataclass
class TemplateWeights:
    """Weights for computing base score, varies by template."""
    template_id: str
    template_name: str
    output_name: str  # "Ideas", "Trends", "Shifts", "Skills", "Claims"

    # Score weights (should sum to ~1.0)
    w_engagement: float = 0.25
    w_recency: float = 0.15
    w_keyword_overlap: float = 0.20
    w_intensity: float = 0.15
    w_proxy_signal: float = 0.25

    # Recency decay: score = base * exp(-age_hours / decay_hours)
    recency_decay_hours: float = 48.0

    # Scorecard weights for final ranking
    signal_weight: float = 0.25
    impact_weight: float = 0.25
    reachability_weight: float = 0.20
    complexity_weight: float = 0.15  # Lower is better
    moat_weight: float = 0.15


# Template-specific configurations
TEMPLATE_WEIGHTS = {
    "content_ideas": TemplateWeights(
        template_id="content_ideas",
        template_name="Content Ideas",
        output_name="Content Ideas",
        w_engagement=0.30,  # Engagement signals content resonance
        w_recency=0.15,
        w_keyword_overlap=0.20,
        w_intensity=0.10,
        w_proxy_signal=0.25,
    ),
    "trend_radar": TemplateWeights(
        template_id="trend_radar",
        template_name="Trend Radar",
        output_name="Trends",
        w_engagement=0.25,
        w_recency=0.25,  # Recency more important for trends
        w_keyword_overlap=0.15,
        w_intensity=0.10,
        w_proxy_signal=0.25,
        recency_decay_hours=24.0,  # Faster decay for trends
    ),
    "industry_intel": TemplateWeights(
        template_id="industry_intel",
        template_name="Industry Intel",
        output_name="Shifts",
        w_engagement=0.30,
        w_recency=0.15,
        w_keyword_overlap=0.20,
        w_intensity=0.10,
        w_proxy_signal=0.25,
    ),
    "career_intel": TemplateWeights(
        template_id="career_intel",
        template_name="Career Intel",
        output_name="Skills/Plays",
        w_engagement=0.25,
        w_recency=0.15,
        w_keyword_overlap=0.20,
        w_intensity=0.15,
        w_proxy_signal=0.25,
    ),
    "deep_research": TemplateWeights(
        template_id="deep_research",
        template_name="Deep Research",
        output_name="Claims",
        w_engagement=0.15,  # Less important for research
        w_recency=0.10,
        w_keyword_overlap=0.25,
        w_intensity=0.15,
        w_proxy_signal=0.35,  # Evidence signals important
        recency_decay_hours=168.0,  # Research can be older
    ),
    "custom": TemplateWeights(
        template_id="custom",
        template_name="Custom",
        output_name="Insights",
        w_engagement=0.25,
        w_recency=0.15,
        w_keyword_overlap=0.20,
        w_intensity=0.15,
        w_proxy_signal=0.25,
    ),
}


def get_template_weights(template_id: str) -> TemplateWeights:
    """Get weights for a template, with fallback to custom."""
    return TEMPLATE_WEIGHTS.get(template_id, TEMPLATE_WEIGHTS["custom"])


# =============================================================================
# SCORE COMPUTATION
# =============================================================================

def normalize_engagement(score: int, num_comments: int) -> float:
    """Normalize engagement to 0-1 scale using log scaling."""
    import math
    # Combined engagement
    engagement = score + (num_comments * 2)  # Comments weighted more
    if engagement <= 0:
        return 0.0
    # Log scale: 10 = 0.3, 100 = 0.6, 1000 = 0.9
    return min(1.0, math.log10(engagement + 1) / 3.5)


def recency_decay(age_hours: float, decay_hours: float) -> float:
    """Compute recency decay factor (0-1)."""
    import math
    if age_hours < 0:
        age_hours = 0
    return math.exp(-age_hours / decay_hours)


def compute_base_score(
    post_features: PostFeatures,
    engagement_score: int,
    num_comments: int,
    age_hours: float,
    weights: TemplateWeights,
) -> float:
    """Compute weighted base score for a post.

    Returns:
        Score from 0 to 1
    """
    # Normalize inputs
    engagement_norm = normalize_engagement(engagement_score, num_comments)
    recency_norm = recency_decay(age_hours, weights.recency_decay_hours)

    # Compute weighted score
    score = (
        weights.w_engagement * engagement_norm +
        weights.w_recency * recency_norm +
        weights.w_keyword_overlap * post_features.keyword_overlap +
        weights.w_intensity * post_features.intensity_score +
        weights.w_proxy_signal * post_features.proxy_score
    )

    return min(1.0, score)


# =============================================================================
# CLUSTER RANKING
# =============================================================================

def rank_by_cluster(
    clusters: list[ClusterResult],
    weights: TemplateWeights,
    top_n: int = 5,
) -> list[ClusterResult]:
    """Rank clusters by combined score.

    Applies penalty to "general" cluster to avoid it dominating insights.
    General can still win if it has >2x engagement of second-best cluster.

    Returns:
        Top N clusters sorted by score
    """
    def cluster_score(c: ClusterResult) -> float:
        # Engagement component (log normalized)
        import math
        engagement = c.engagement_score_sum + (c.engagement_comments_sum * 2)
        eng_norm = math.log10(engagement + 1) / 4.0 if engagement > 0 else 0

        # Age component (inverse - older = lower score)
        rec_norm = recency_decay(c.avg_age_hours, weights.recency_decay_hours)

        # Combine with weights
        base_score = (
            weights.w_engagement * eng_norm +
            weights.w_recency * rec_norm +
            weights.w_intensity * c.intensity_score +
            weights.w_proxy_signal * c.proxy_score +
            0.1 * (c.frequency_posts / 10)  # Small boost for frequency
        )

        # Penalty for "general" cluster - it's a catch-all and less actionable
        if c.cluster_id == "general":
            base_score *= 0.70  # 30% penalty

        return base_score

    # Score and sort
    scored = [(c, cluster_score(c)) for c in clusters]
    scored.sort(key=lambda x: x[1], reverse=True)

    return [c for c, _ in scored[:top_n]]


# =============================================================================
# SCORECARDS
# =============================================================================

@dataclass
class Scorecard:
    """Mini scorecard for a top item."""
    title: str
    cluster: str
    item_type: str  # software, community, workflow, info-product

    signal_strength: int  # 1-5
    impact_wtp: int  # 1-5
    reachability: int  # 1-5
    complexity: int  # 1-5
    moat: int  # 1-5

    total_score: float
    evidence_urls: list[str]
    key_insight: str


def estimate_item_type(cluster_id: str, keywords: list[str]) -> str:
    """Estimate item type based on cluster and keywords."""
    keywords_lower = [k.lower() for k in keywords]

    # Check for software indicators
    software_words = ["app", "tool", "software", "saas", "api", "platform", "plugin"]
    if any(w in keywords_lower for w in software_words):
        return "software"

    # Check for community indicators
    community_words = ["community", "forum", "discord", "slack", "group", "network"]
    if any(w in keywords_lower for w in community_words):
        return "community"

    # Check for info-product indicators
    info_words = ["course", "book", "guide", "newsletter", "content", "template"]
    if any(w in keywords_lower for w in info_words):
        return "info-product"

    # Default based on cluster
    if cluster_id in ["ops_systems", "tools_tech"]:
        return "software"
    elif cluster_id in ["info_management", "research_uncertainty"]:
        return "info-product"
    else:
        return "workflow"


def estimate_complexity(cluster_id: str, keywords: list[str]) -> int:
    """Estimate complexity score 1-5 based on cluster and keywords."""
    keywords_lower = [k.lower() for k in keywords]

    # High complexity indicators
    complex_words = ["integration", "api", "scraping", "auth", "oauth", "database",
                     "ml", "ai", "infrastructure", "enterprise", "compliance"]
    complex_count = sum(1 for w in complex_words if w in keywords_lower)

    if complex_count >= 3:
        return 5
    elif complex_count >= 2:
        return 4
    elif complex_count >= 1:
        return 3

    # Lower complexity for certain clusters
    if cluster_id in ["info_management", "career_skills"]:
        return 2
    return 3  # Default medium


def estimate_moat(cluster_id: str, keywords: list[str], engagement: int) -> int:
    """Estimate moat score 1-5."""
    keywords_lower = [k.lower() for k in keywords]

    # Strong moat indicators
    moat_words = ["data", "network", "community", "workflow", "integration",
                  "lock-in", "switching", "habit", "daily"]
    moat_count = sum(1 for w in moat_words if w in keywords_lower)

    base_moat = min(4, moat_count + 1)

    # Boost for high engagement (validates demand)
    if engagement > 1000:
        base_moat = min(5, base_moat + 1)

    return max(1, base_moat)


def generate_scorecards(
    clusters: list[ClusterResult],
    post_features: list[PostFeatures],
    weights: TemplateWeights,
    top_n: int = 3,
) -> list[Scorecard]:
    """Generate scorecards for top items.

    Args:
        clusters: Ranked cluster results
        post_features: All computed post features
        weights: Template weights
        top_n: Number of scorecards to generate

    Returns:
        List of Scorecard objects
    """
    # Build lookup for post features by cluster
    features_by_cluster: dict[str, list[PostFeatures]] = {}
    for pf in post_features:
        if pf.cluster not in features_by_cluster:
            features_by_cluster[pf.cluster] = []
        features_by_cluster[pf.cluster].append(pf)

    scorecards = []

    for cluster in clusters[:top_n]:
        cluster_features = features_by_cluster.get(cluster.cluster_id, [])
        if not cluster_features:
            continue

        # Get best post in cluster
        best_post = max(cluster_features, key=lambda x: x.base_score)

        # Compute scorecard metrics
        item_type = estimate_item_type(cluster.cluster_id, best_post.keywords)

        # Signal strength: based on frequency and cross-post support
        signal = min(5, 1 + cluster.frequency_posts)

        # Impact/WTP: based on proxy score
        impact = min(5, int(cluster.proxy_score * 5) + 1)

        # Reachability: based on engagement (high engagement = reachable audience)
        reach = min(5, int(normalize_engagement(
            cluster.engagement_score_sum,
            cluster.engagement_comments_sum
        ) * 5) + 1)

        # Complexity
        complexity = estimate_complexity(cluster.cluster_id, best_post.keywords)

        # Moat
        moat = estimate_moat(
            cluster.cluster_id,
            best_post.keywords,
            cluster.engagement_score_sum
        )

        # Total score (complexity inverted - lower is better)
        total = (
            weights.signal_weight * signal +
            weights.impact_weight * impact +
            weights.reachability_weight * reach +
            weights.complexity_weight * (6 - complexity) +  # Invert
            weights.moat_weight * moat
        )

        # Key insight from cluster-level signals (not single post keywords)
        # Use cluster.key_signals if available, fallback to post keywords
        if hasattr(cluster, 'key_signals') and cluster.key_signals:
            key_insight = ", ".join(cluster.key_signals[:5])
        else:
            key_insight = ", ".join(best_post.keywords[:5])

        scorecards.append(Scorecard(
            title=cluster.cluster_name,
            cluster=cluster.cluster_id,
            item_type=item_type,
            signal_strength=signal,
            impact_wtp=impact,
            reachability=reach,
            complexity=complexity,
            moat=moat,
            total_score=total,
            evidence_urls=cluster.top_urls[:3],
            key_insight=key_insight,
        ))

    # Sort by total score
    scorecards.sort(key=lambda x: x.total_score, reverse=True)

    return scorecards


def format_scorecards(
    scorecards: list[Scorecard],
    output_name: str = "Items",
) -> str:
    """Format scorecards as markdown."""
    lines = [f"## Top 3 {output_name}\n"]

    for i, sc in enumerate(scorecards[:3], 1):
        lines.append(f"### {i}. {sc.title}")
        lines.append(f"**Type:** {sc.item_type}")
        lines.append(f"**Key signals:** {sc.key_insight}")
        lines.append("")

        # Scorecard table
        lines.append("| Metric | Score |")
        lines.append("|--------|-------|")
        lines.append(f"| Signal Strength | {'★' * sc.signal_strength}{'☆' * (5 - sc.signal_strength)} |")
        lines.append(f"| Impact/WTP | {'★' * sc.impact_wtp}{'☆' * (5 - sc.impact_wtp)} |")
        lines.append(f"| Reachability | {'★' * sc.reachability}{'☆' * (5 - sc.reachability)} |")
        lines.append(f"| Complexity | {'★' * sc.complexity}{'☆' * (5 - sc.complexity)} |")
        lines.append(f"| Moat Potential | {'★' * sc.moat}{'☆' * (5 - sc.moat)} |")
        lines.append("")

        if sc.evidence_urls:
            lines.append("**Evidence:**")
            for url in sc.evidence_urls:
                lines.append(f"- [{url[:50]}...]({url})")
            lines.append("")

        lines.append("---\n")

    return "\n".join(lines)


# =============================================================================
# LLM VALUE SCORING (for heatmap)
# =============================================================================

def score_clusters_with_llm(
    clusters: list[ClusterResult],
    goal_prompt: str,
    template_id: str = "content_ideas",
) -> dict[str, float]:
    """Score clusters using LLM based on value to user's goal.

    Args:
        clusters: List of cluster results to score
        goal_prompt: The user's goal/use case
        template_id: Template ID for context

    Returns:
        Dict mapping cluster_id to value score (0-10)
    """
    import json
    import re

    try:
        from src.llm_client import get_llm_client, LLMClientError
    except ImportError:
        logger.warning("LLM client not available, using fallback scoring")
        return _fallback_cluster_scores(clusters)

    if not clusters:
        return {}

    # Build cluster info for prompt
    cluster_info = []
    for c in clusters:
        signals = ", ".join(c.key_signals[:5]) if c.key_signals else "general discussion"
        cluster_info.append(
            f"- {c.cluster_id}: \"{c.cluster_name}\" - {c.frequency_posts} posts, "
            f"signals: {signals}"
        )

    cluster_list = "\n".join(cluster_info)

    prompt = f"""You are evaluating topic clusters from Reddit discussions for their value to a specific use case.

USER'S GOAL: {goal_prompt}

CLUSTERS FOUND:
{cluster_list}

Score each cluster from 0-10 based on how valuable it is for the user's goal:
- 0-2: Low value, not relevant to the goal
- 3-4: Some relevance but limited actionable value
- 5-6: Moderately valuable, useful insights
- 7-8: High value, directly actionable for the goal
- 9-10: Extremely valuable, critical insights for the goal

Return ONLY a JSON object mapping cluster_id to score. Example:
{{"learning_skills": 8.5, "tools_tech": 7.0, "general": 3.0}}

JSON response:"""

    try:
        client = get_llm_client()
        response = client.generate(
            prompt=prompt,
            system_prompt="You are a helpful analyst. Return only valid JSON.",
            temperature=0.3,
            max_tokens=500,
        )

        # Parse JSON response
        content = response.content.strip()
        # Handle markdown code blocks
        if "```" in content:
            match = re.search(r'```(?:json)?\s*(.*?)\s*```', content, re.DOTALL)
            if match:
                content = match.group(1)

        scores = json.loads(content)

        # Validate and normalize scores
        result = {}
        for cluster_id, score in scores.items():
            try:
                score_val = float(score)
                result[cluster_id] = max(0, min(10, score_val))
            except (ValueError, TypeError):
                continue

        logger.info(f"[LLM Scoring] Scored {len(result)} clusters")
        return result

    except (LLMClientError, json.JSONDecodeError) as e:
        logger.warning(f"[LLM Scoring] Failed: {e}, using fallback")
        return _fallback_cluster_scores(clusters)
    except Exception as e:
        logger.error(f"[LLM Scoring] Unexpected error: {e}")
        return _fallback_cluster_scores(clusters)


def _fallback_cluster_scores(clusters: list[ClusterResult]) -> dict[str, float]:
    """Fallback scoring when LLM is not available.

    Uses engagement and proxy_score to estimate value.
    """
    import math

    scores = {}
    for c in clusters:
        # Combine frequency, engagement, and proxy_score
        eng_score = math.log10(c.engagement_score_sum + 1) / 4.0  # 0-1 scale
        freq_score = min(c.frequency_posts / 10, 1.0)
        proxy = c.proxy_score

        # Weighted combination, scaled to 0-10
        raw_score = (eng_score * 0.3 + freq_score * 0.3 + proxy * 0.4) * 10

        # Penalty for "general" cluster
        if c.cluster_id == "general":
            raw_score *= 0.6

        scores[c.cluster_id] = round(min(10, max(0, raw_score)), 2)

    return scores
