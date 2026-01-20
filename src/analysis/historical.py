"""Historical analysis module for RedditRadar.

Provides LLM-powered analysis of trends across multiple runs:
- Retrieves past runs with digests, posts, and clusters
- Identifies recurring ideas and patterns
- Generates insights about what themes persist over time
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from src.database import Database, Run, Digest, Post, get_database

logger = logging.getLogger(__name__)


@dataclass
class HistoricalRunData:
    """Data from a single historical run."""
    run_id: str
    run_date: str
    template: str
    created_at: float

    # Digest content
    digest_markdown: str

    # Top posts from this run
    top_posts: list[dict]  # [{title, body, subreddit, score, num_comments}]

    # Cluster data
    clusters: list[dict]  # [{cluster_name, frequency, engagement, intensity, proxy}]

    # Extracted ideas (parsed from digest)
    extracted_ideas: list[str]


@dataclass
class HistoricalAnalysisResult:
    """Result of historical analysis."""
    days_analyzed: int
    runs_analyzed: int

    # Recurring patterns
    recurring_ideas: list[dict]  # [{idea, frequency, first_seen, last_seen}]
    recurring_clusters: list[dict]  # [{cluster, appearances, avg_engagement}]

    # Trends
    rising_themes: list[str]
    declining_themes: list[str]

    # LLM-generated insights
    llm_analysis: str

    # Raw data for reference
    runs_data: list[HistoricalRunData]


def get_historical_runs(
    db: Database,
    days: int = 7,
    template_filter: Optional[str] = None,
) -> list[HistoricalRunData]:
    """Retrieve historical runs with full context.

    Args:
        db: Database instance
        days: Number of days to look back
        template_filter: Optional template ID to filter by

    Returns:
        List of HistoricalRunData with digests, posts, and clusters
    """
    from datetime import datetime, timedelta

    cutoff = datetime.now().timestamp() - (days * 24 * 60 * 60)

    # Get recent runs
    runs = db.get_recent_runs(limit=50)
    runs = [r for r in runs if r.created_at >= cutoff]

    if template_filter:
        runs = [r for r in runs if r.goal_template == template_filter]

    results = []

    for run in runs:
        # Get digest
        digest = db.get_digest(run.run_id)
        if not digest:
            continue

        # Get posts for this run
        posts = db.get_posts_for_run(run.run_id)
        top_posts = [
            {
                "title": p.title,
                "body": (p.body or "")[:300],
                "subreddit": p.subreddit,
                "score": p.score,
                "num_comments": p.num_comments,
            }
            for p in posts[:10]  # Top 10 posts
        ]

        # Get clusters for this run
        clusters_raw = db.get_run_clusters(run.run_id)
        clusters = [
            {
                "cluster_name": c.cluster_name,
                "frequency": c.frequency_posts,
                "engagement": c.engagement_score_sum + c.engagement_comments_sum,
                "intensity": c.intensity_score,
                "proxy": c.proxy_score,
            }
            for c in clusters_raw
        ]

        # Extract ideas from digest markdown
        extracted_ideas = extract_ideas_from_digest(digest.markdown)

        results.append(HistoricalRunData(
            run_id=run.run_id,
            run_date=run.run_date,
            template=run.goal_template,
            created_at=run.created_at,
            digest_markdown=digest.markdown,
            top_posts=top_posts,
            clusters=clusters,
            extracted_ideas=extracted_ideas,
        ))

    # Sort by date descending
    results.sort(key=lambda x: x.created_at, reverse=True)

    return results


def extract_ideas_from_digest(markdown: str) -> list[str]:
    """Extract idea titles/summaries from digest markdown.

    Looks for patterns like:
    - ### 1. Idea Title
    - **Idea:** description
    - Numbered lists with bold text
    """
    import re

    ideas = []

    # Pattern 1: ### N. Title
    pattern1 = re.findall(r'###\s*\d+\.\s*(.+)', markdown)
    ideas.extend(pattern1)

    # Pattern 2: **Title** at start of line (often in scorecards)
    pattern2 = re.findall(r'^\*\*([^*]+)\*\*', markdown, re.MULTILINE)
    for p in pattern2:
        if len(p) > 10 and len(p) < 100 and not p.startswith(('Type:', 'Key', 'Evidence', 'Frequency', 'Engagement')):
            ideas.append(p)

    # Pattern 3: Numbered items with bold
    pattern3 = re.findall(r'^\d+\.\s*\*\*([^*]+)\*\*', markdown, re.MULTILINE)
    ideas.extend(pattern3)

    # Dedupe while preserving order
    seen = set()
    unique_ideas = []
    for idea in ideas:
        idea_clean = idea.strip()
        if idea_clean and idea_clean.lower() not in seen:
            seen.add(idea_clean.lower())
            unique_ideas.append(idea_clean)

    return unique_ideas[:20]  # Max 20 ideas per run


def aggregate_recurring_patterns(runs: list[HistoricalRunData]) -> dict:
    """Aggregate patterns across runs to find recurring themes.

    Returns dict with:
    - idea_frequency: {idea: count}
    - cluster_frequency: {cluster: {count, total_engagement}}
    - idea_timeline: {idea: [dates]}
    """
    from collections import defaultdict

    idea_frequency = defaultdict(int)
    idea_timeline = defaultdict(list)
    cluster_stats = defaultdict(lambda: {"count": 0, "total_engagement": 0})

    for run in runs:
        run_date = run.run_date

        # Count ideas
        for idea in run.extracted_ideas:
            # Normalize idea for matching
            idea_norm = idea.lower().strip()
            idea_frequency[idea] += 1
            idea_timeline[idea].append(run_date)

        # Count clusters
        for cluster in run.clusters:
            name = cluster["cluster_name"]
            cluster_stats[name]["count"] += 1
            cluster_stats[name]["total_engagement"] += cluster["engagement"]

    return {
        "idea_frequency": dict(idea_frequency),
        "idea_timeline": dict(idea_timeline),
        "cluster_stats": dict(cluster_stats),
    }


def format_historical_context_for_llm(
    runs: list[HistoricalRunData],
    patterns: dict,
    max_tokens: int = 4000,
) -> str:
    """Format historical data for LLM analysis.

    Creates a structured summary that fits within token limits.
    """
    lines = []

    # Header
    lines.append("# Historical Analysis Context")
    lines.append(f"\nAnalyzing {len(runs)} runs from the past {len(set(r.run_date for r in runs))} days.\n")

    # Section 1: Recurring Ideas
    lines.append("## Recurring Ideas Across Runs\n")

    idea_freq = patterns["idea_frequency"]
    recurring = [(idea, count) for idea, count in idea_freq.items() if count >= 2]
    recurring.sort(key=lambda x: x[1], reverse=True)

    if recurring:
        for idea, count in recurring[:15]:
            dates = patterns["idea_timeline"].get(idea, [])
            lines.append(f"- **{idea}** (appeared {count}x: {', '.join(dates[:3])})")
    else:
        lines.append("*No ideas appeared multiple times yet.*")
    lines.append("")

    # Section 2: Cluster Trends
    lines.append("## Cluster Trends\n")

    cluster_stats = patterns["cluster_stats"]
    cluster_list = [
        (name, stats["count"], stats["total_engagement"])
        for name, stats in cluster_stats.items()
    ]
    cluster_list.sort(key=lambda x: x[1], reverse=True)

    for name, count, engagement in cluster_list[:10]:
        avg_eng = engagement // count if count > 0 else 0
        lines.append(f"- **{name}**: {count} appearances, avg engagement {avg_eng}")
    lines.append("")

    # Section 3: Run Summaries (most recent first)
    lines.append("## Recent Run Summaries\n")

    for run in runs[:5]:  # Last 5 runs
        lines.append(f"### {run.run_date} ({run.template})")

        # Top ideas from this run
        if run.extracted_ideas:
            lines.append("**Top ideas:**")
            for idea in run.extracted_ideas[:5]:
                lines.append(f"  - {idea}")

        # Top posts
        if run.top_posts:
            lines.append("**Top posts:**")
            for post in run.top_posts[:3]:
                lines.append(f"  - [{post['subreddit']}] {post['title'][:60]}... (score: {post['score']})")

        lines.append("")

    # Section 4: All unique ideas for pattern matching
    lines.append("## All Ideas Discovered\n")

    all_ideas = set()
    for run in runs:
        all_ideas.update(run.extracted_ideas)

    for idea in list(all_ideas)[:30]:
        freq = idea_freq.get(idea, 1)
        marker = "🔄" if freq >= 2 else ""
        lines.append(f"- {idea} {marker}")

    result = "\n".join(lines)

    # Truncate if too long (rough estimate: 4 chars per token)
    max_chars = max_tokens * 4
    if len(result) > max_chars:
        result = result[:max_chars] + "\n\n[Truncated for length]"

    return result


def build_historical_analysis_prompt(context: str, template_id: str) -> tuple[str, str]:
    """Build system and user prompts for historical analysis.

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    system_prompt = """You are an expert analyst specializing in identifying patterns and trends across multiple research sessions.

Your task is to analyze historical data from Reddit analysis runs and identify:
1. **Recurring Themes**: Ideas or problems that appear repeatedly across different runs
2. **Persistent Opportunities**: Topics with consistent engagement that suggest real demand
3. **Emerging Trends**: New themes gaining momentum
4. **Pattern Insights**: What the repetition tells us about the market/community

Be specific and cite evidence from the data. Focus on actionable insights."""

    if template_id == "content_ideas":
        focus = "content opportunities, audience interests, and engagement patterns"
    elif template_id == "career_intel":
        focus = "career trends, skill demands, and job market patterns"
    elif template_id == "trend_radar":
        focus = "emerging trends, technology shifts, and market movements"
    elif template_id == "industry_intel":
        focus = "industry dynamics, competitive movements, and market shifts"
    else:
        focus = "patterns, trends, and recurring themes"

    user_prompt = f"""Analyze the following historical data from multiple Reddit analysis runs.

Focus on: {focus}

{context}

---

Please provide:

## 1. Recurring Themes (Ideas that keep appearing)
Identify ideas/topics that appear multiple times. Why might they persist?

## 2. Strongest Signals (High-confidence opportunities)
Which themes have the strongest evidence based on frequency + engagement?

## 3. Emerging Patterns (New but growing)
Any newer themes that seem to be gaining traction?

## 4. Key Insights
What does the pattern of recurring ideas tell us? Any surprising findings?

## 5. Recommended Actions
Based on historical patterns, what should be prioritized for further investigation?
"""

    return system_prompt, user_prompt


async def run_historical_analysis(
    db: Database,
    llm_client,
    days: int = 7,
    template_filter: Optional[str] = None,
) -> HistoricalAnalysisResult:
    """Run full historical analysis with LLM.

    Args:
        db: Database instance
        llm_client: LLM client for generating insights
        days: Days to analyze
        template_filter: Optional template to filter by

    Returns:
        HistoricalAnalysisResult with patterns and LLM insights
    """
    logger.info(f"[Historical] Starting analysis for past {days} days...")

    # Get historical runs
    runs = get_historical_runs(db, days=days, template_filter=template_filter)
    logger.info(f"[Historical] Found {len(runs)} runs to analyze")

    if not runs:
        return HistoricalAnalysisResult(
            days_analyzed=days,
            runs_analyzed=0,
            recurring_ideas=[],
            recurring_clusters=[],
            rising_themes=[],
            declining_themes=[],
            llm_analysis="No historical data found. Run more analyses to build history.",
            runs_data=[],
        )

    # Aggregate patterns
    patterns = aggregate_recurring_patterns(runs)
    logger.info(f"[Historical] Found {len(patterns['idea_frequency'])} unique ideas")

    # Format for LLM
    context = format_historical_context_for_llm(runs, patterns)

    # Get template for prompt customization
    template_id = template_filter or (runs[0].template if runs else "custom")

    # Build prompts
    system_prompt, user_prompt = build_historical_analysis_prompt(context, template_id)

    # Generate LLM analysis
    logger.info("[Historical] Generating LLM analysis...")
    response = llm_client.generate(user_prompt, system_prompt)
    llm_analysis = response.content

    # Build recurring ideas list
    recurring_ideas = []
    for idea, count in patterns["idea_frequency"].items():
        if count >= 2:
            dates = patterns["idea_timeline"].get(idea, [])
            recurring_ideas.append({
                "idea": idea,
                "frequency": count,
                "first_seen": min(dates) if dates else None,
                "last_seen": max(dates) if dates else None,
            })
    recurring_ideas.sort(key=lambda x: x["frequency"], reverse=True)

    # Build recurring clusters list
    recurring_clusters = []
    for name, stats in patterns["cluster_stats"].items():
        if stats["count"] >= 2:
            recurring_clusters.append({
                "cluster": name,
                "appearances": stats["count"],
                "avg_engagement": stats["total_engagement"] // stats["count"],
            })
    recurring_clusters.sort(key=lambda x: x["appearances"], reverse=True)

    return HistoricalAnalysisResult(
        days_analyzed=days,
        runs_analyzed=len(runs),
        recurring_ideas=recurring_ideas,
        recurring_clusters=recurring_clusters,
        rising_themes=[],  # Could be computed from engagement trends
        declining_themes=[],
        llm_analysis=llm_analysis,
        runs_data=runs,
    )


def run_historical_analysis_sync(
    db: Database,
    llm_client,
    days: int = 7,
    template_filter: Optional[str] = None,
) -> HistoricalAnalysisResult:
    """Synchronous wrapper for historical analysis."""
    import asyncio

    # Check if we're in an async context
    try:
        loop = asyncio.get_running_loop()
        # We're in async context, need to handle differently
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(
                asyncio.run,
                run_historical_analysis(db, llm_client, days, template_filter)
            )
            return future.result()
    except RuntimeError:
        # No running loop, safe to use asyncio.run
        return asyncio.run(
            run_historical_analysis(db, llm_client, days, template_filter)
        )


def format_historical_report(result: HistoricalAnalysisResult) -> str:
    """Format historical analysis result as markdown."""
    lines = []

    lines.append("# Historical Trend Analysis")
    lines.append(f"\n*Analyzed {result.runs_analyzed} runs over the past {result.days_analyzed} days*\n")

    # Recurring Ideas
    lines.append("## Recurring Ideas\n")
    if result.recurring_ideas:
        lines.append("Ideas that appeared multiple times across runs:\n")
        for item in result.recurring_ideas[:10]:
            lines.append(f"- **{item['idea']}** ({item['frequency']}x, {item['first_seen']} → {item['last_seen']})")
    else:
        lines.append("*No recurring ideas found yet. Run more analyses.*")
    lines.append("")

    # Recurring Clusters
    lines.append("## Recurring Clusters\n")
    if result.recurring_clusters:
        for item in result.recurring_clusters[:10]:
            lines.append(f"- **{item['cluster']}**: {item['appearances']} appearances, avg engagement {item['avg_engagement']}")
    else:
        lines.append("*No recurring clusters found yet.*")
    lines.append("")

    # LLM Analysis
    lines.append("## AI Analysis\n")
    lines.append(result.llm_analysis)

    return "\n".join(lines)
