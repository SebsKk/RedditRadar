"""Goal templates for Reddit Radar.

This module contains predefined goal templates and the prompt builder
for generating customized analysis prompts.
"""

from dataclasses import dataclass
from typing import Literal

from src.config import PromptBuilderConfig


# Type aliases
OutputFormat = Literal["bullets", "brief", "structured"]
OutputLength = Literal["short", "medium", "long"]
Creativity = Literal["low", "medium", "high"]
Factuality = Literal["strict", "balanced", "exploratory"]
Tone = Literal["neutral", "direct", "playful"]


@dataclass
class GoalTemplate:
    """A goal template for analysis."""
    id: str
    name: str
    description: str
    system_prompt: str
    user_prompt_template: str
    suggested_subreddits: list[str]


# ============================================================================
# Goal Templates
# ============================================================================

CONTENT_IDEAS = GoalTemplate(
    id="content_ideas",
    name="Content Ideas",
    description="Discover content topics, angles, and formats that resonate with audiences",
    system_prompt="""You are a content strategist analyzing Reddit discussions
to identify content opportunities. Your job is to find topics people care about,
questions they're asking, and angles that would resonate.

Focus on:
- Questions people keep asking (content gaps)
- Topics generating high engagement
- Debates and controversies (opinion content)
- How-to requests and tutorials needed
- Myths and misconceptions to address
- Success stories people want to hear

Be specific about content formats and angles. Support ideas with evidence.

IMPORTANT: When citing evidence or sources, ONLY use the Reddit post URLs provided in the input data (URLs starting with https://reddit.com or https://www.reddit.com). Do NOT reference external websites like Wikipedia. Each post in the input includes a "URL:" field - use those URLs as your sources.""",
    user_prompt_template="""Analyze these Reddit posts and comments to identify content opportunities.

{content}

Provide your analysis with these sections:

## Hot Topics
List 5-10 topics generating the most discussion and engagement.

## Top 10 Content Ideas
For each idea:
- **Topic**: What to cover
- **Format**: Blog post / Video / Thread / Guide / Comparison / etc.
- **Angle**: Specific hook or perspective
- **Evidence**: Reddit post URLs from the input showing demand (use the URL field from each post)
- **Target Audience**: Who would consume this

## Questions to Answer
Common questions people are asking (FAQ content opportunities).

## Myths to Bust
Misconceptions or debates where you could provide clarity.

## Content Gaps
Topics people want covered but can't find good content about.

## Engagement Patterns
What types of content get the most engagement in this space?

## Action Plan
Top 5 content pieces to create first, prioritized by potential impact.
""",
    suggested_subreddits=[
        "content_marketing", "blogging", "socialmedia", "youtube",
        "copywriting", "marketing", "SEO"
    ],
)

TREND_RADAR = GoalTemplate(
    id="trend_radar",
    name="Trend Radar",
    description="Identify emerging trends and predict future directions",
    system_prompt="""You are a trend analyst specializing in identifying emerging patterns
from online discussions. Your job is to spot early signals of trends before they become mainstream.

Focus on:
- Topics gaining unusual traction
- Shifts in sentiment or opinion
- New tools, technologies, or methods being discussed
- Changes in behavior or preferences
- Emerging terminology or concepts

Distinguish between noise and genuine signals. Support findings with evidence.

IMPORTANT: When citing evidence or sources, ONLY use the Reddit post URLs provided in the input data (URLs starting with https://reddit.com or https://www.reddit.com). Do NOT reference external websites like Wikipedia. Each post in the input includes a "URL:" field - use those URLs as your sources.""",
    user_prompt_template="""Analyze these Reddit posts and comments to identify emerging trends.

{content}

Provide your analysis with these sections:

## Emerging Trends
List 5-10 trends you've identified, for each:
- **Trend**: Name/description
- **Signal Strength**: Strong / Moderate / Early
- **Evidence**: Reddit post URLs from the input (use the URL field from each post)
- **Why It Matters**: Potential impact

## Trend Analysis
For the top 3 trends:
- **Current State**: Where is this trend now?
- **Predicted Direction**: Where is it heading?
- **Timeline**: When might it become mainstream?
- **Who's Affected**: Industries, demographics impacted
- **Opportunities**: How to capitalize on this trend

## Weak Signals
Early-stage patterns that might become trends (high uncertainty).

## Counter-Trends
Any pushback or opposing movements worth noting.
""",
    suggested_subreddits=[
        "Futurology", "technology", "gadgets", "tech",
        "TrendingTechnology", "emerging_tech"
    ],
)

INDUSTRY_INTEL = GoalTemplate(
    id="industry_intel",
    name="Industry Intel",
    description="Gather competitive intelligence and industry insights",
    system_prompt="""You are a competitive intelligence analyst. Your job is to extract
actionable business intelligence from online discussions.

Focus on:
- Industry changes and disruptions
- Competitor mentions and sentiment
- Customer complaints about existing solutions
- Pricing and positioning discussions
- Hiring and talent trends
- Strategic moves and announcements

Be objective and cite sources for claims.

IMPORTANT: When citing evidence or sources, ONLY use the Reddit post URLs provided in the input data (URLs starting with https://reddit.com or https://www.reddit.com). Do NOT reference external websites like Wikipedia. Each post in the input includes a "URL:" field - use those URLs as your sources.""",
    user_prompt_template="""Analyze these Reddit posts and comments for industry intelligence.

{content}

Provide your analysis with these sections:

## Industry Overview
What's the current state of discussion in this space?

## Key Changes
What significant changes are happening?
- **What**: Description of change
- **Evidence**: Reddit post URLs from the input (use the URL field from each post)
- **Impact**: Who's affected and how

## Competitive Landscape
- Notable mentions of companies/products
- Sentiment toward competitors
- Feature comparisons being discussed

## Customer Voice
What are customers saying?
- Common complaints
- Feature requests
- Praise and positive mentions

## Strategic Opportunities
Based on this intelligence:
- Gaps in the market
- Positioning opportunities
- Threats to watch

## Key Debates
What controversial topics are being discussed?

## Actionable Takeaways
Top 5 actions based on this intelligence.
""",
    suggested_subreddits=[
        "business", "marketing", "sales", "consulting",
        "BusinessIntelligence", "strategy"
    ],
)

CAREER_INTEL = GoalTemplate(
    id="career_intel",
    name="Career Intel",
    description="Identify hot skills, career trends, and learning opportunities",
    system_prompt="""You are a career intelligence analyst. Your job is to identify
valuable skills, career trends, and learning opportunities from professional discussions.

Focus on:
- Skills in high demand
- Technologies gaining traction
- Career paths and transitions
- Salary and compensation trends
- Interview and hiring patterns
- Learning resources recommended

Be practical and actionable for career development.

IMPORTANT: When citing evidence or sources, ONLY use the Reddit post URLs provided in the input data (URLs starting with https://reddit.com or https://www.reddit.com). Do NOT reference external websites like Wikipedia. Each post in the input includes a "URL:" field - use those URLs as your sources.""",
    user_prompt_template="""Analyze these Reddit posts and comments for career intelligence.

{content}

Provide your analysis with these sections:

## Hot Skills
Top 10 skills being discussed as valuable:
- **Skill**: Name
- **Demand Level**: High / Growing / Stable
- **Context**: Why it's valuable
- **Evidence**: Reddit post URLs from the input (use the URL field from each post)

## Technology Trends
What tools/technologies are professionals talking about?
- Rising in popularity
- Declining or becoming obsolete
- Controversially discussed

## Career Paths
Common career transitions or paths discussed.

## Learning Recommendations
What are people recommending for skill development?
- Courses
- Books
- Projects
- Certifications

## Job Market Insights
- Hiring trends
- Company mentions (good/bad employers)
- Compensation discussions

## What to Build
Project ideas that would demonstrate valuable skills.

## Action Plan
Top 5 things to do based on this intelligence.
""",
    suggested_subreddits=[
        "cscareerquestions", "ExperiencedDevs", "dataengineering",
        "MachineLearning", "learnprogramming", "ITCareerQuestions"
    ],
)

DEEP_RESEARCH = GoalTemplate(
    id="deep_research",
    name="Deep Research",
    description="Comprehensive research with evidence and uncertainty analysis",
    system_prompt="""You are a research analyst conducting thorough investigation
of a topic. Your job is to synthesize information, evaluate evidence quality,
and clearly communicate uncertainty.

Focus on:
- Key claims and their evidence
- Consensus vs. disputed points
- Quality and reliability of sources
- Gaps in knowledge
- Multiple perspectives on issues

Be rigorous and transparent about limitations.

IMPORTANT: When citing evidence or sources, ONLY use the Reddit post URLs provided in the input data (URLs starting with https://reddit.com or https://www.reddit.com). Do NOT reference external websites like Wikipedia. Each post in the input includes a "URL:" field - use those URLs as your sources.""",
    user_prompt_template="""Conduct deep research analysis on these Reddit posts and comments.

{content}

Provide your analysis with these sections:

## Executive Summary
3-5 sentence overview of key findings.

## Key Claims & Evidence
For each major claim discussed:
- **Claim**: What's being said
- **Evidence Quality**: Strong / Moderate / Weak / Anecdotal
- **Sources**: Reddit post URLs from the input (use the URL field from each post)
- **Counter-Evidence**: Any contradicting information

## Areas of Consensus
What do most people agree on?

## Areas of Debate
What's contested or controversial?
- Different viewpoints
- Quality of arguments on each side

## Knowledge Gaps
What's unclear or needs more research?

## Source Analysis
- Types of contributors (experts, enthusiasts, etc.)
- Potential biases in the discussion
- Quality of information shared

## Uncertainty Notes
What should readers be cautious about?

## Further Research
Suggested follow-up questions or areas to investigate.

## Key Takeaways
Top 5 insights with confidence levels.
""",
    suggested_subreddits=[
        "askscience", "AskHistorians", "explainlikeimfive",
        "TrueReddit", "NeutralPolitics", "DepthHub"
    ],
)


# Template registry
TEMPLATES: dict[str, GoalTemplate] = {
    "content_ideas": CONTENT_IDEAS,
    "trend_radar": TREND_RADAR,
    "industry_intel": INDUSTRY_INTEL,
    "career_intel": CAREER_INTEL,
    "deep_research": DEEP_RESEARCH,
}


def get_template(template_id: str) -> GoalTemplate | None:
    """Get a template by ID.

    Args:
        template_id: Template identifier.

    Returns:
        GoalTemplate if found, None otherwise.
    """
    return TEMPLATES.get(template_id)


def list_templates() -> list[GoalTemplate]:
    """List all available templates.

    Returns:
        List of all templates.
    """
    return list(TEMPLATES.values())


# ============================================================================
# Prompt Builder
# ============================================================================

def build_prompt_modifiers(config: PromptBuilderConfig) -> str:
    """Build prompt modifiers from configuration.

    Args:
        config: Prompt builder configuration.

    Returns:
        String with prompt modifiers to append.
    """
    modifiers = []

    # Output format
    if config.output_format == "bullets":
        modifiers.append("Use bullet points for all lists and sections.")
    elif config.output_format == "brief":
        modifiers.append("Keep responses concise and to the point.")
    # "structured" is the default, no modifier needed

    # Include links
    if config.include_links:
        modifiers.append("Always cite the source post URL when referencing specific content.")
    else:
        modifiers.append("Do not include URLs in the response.")

    # Output length
    if config.output_length == "short":
        modifiers.append("Keep the response brief (500-1000 words).")
    elif config.output_length == "long":
        modifiers.append("Provide comprehensive detail (2000+ words).")
    # "medium" is the default

    # Creativity
    if config.creativity == "low":
        modifiers.append("Stick closely to information explicitly stated in the posts.")
    elif config.creativity == "high":
        modifiers.append("Feel free to make creative connections and speculate on implications.")
    # "medium" is the default

    # Factuality
    if config.factuality == "strict":
        modifiers.append(
            "Be strict about factuality. Clearly distinguish facts from opinions. "
            "Note uncertainty and qualify speculative statements."
        )
    elif config.factuality == "exploratory":
        modifiers.append(
            "Take an exploratory approach. Feel free to hypothesize and "
            "make educated guesses based on the discussion."
        )
    # "balanced" is the default

    # Tone
    if config.tone == "direct":
        modifiers.append("Use a direct, no-nonsense tone.")
    elif config.tone == "playful":
        modifiers.append("Use a friendly, engaging tone with some personality.")
    # "neutral" is the default

    # Extra instructions
    if config.extra_instructions:
        modifiers.append(f"Additional instructions: {config.extra_instructions}")

    if modifiers:
        return "\n\nStyle Guidelines:\n- " + "\n- ".join(modifiers)
    return ""


def build_full_prompt(
    template: GoalTemplate,
    content: str,
    config: PromptBuilderConfig | None = None,
) -> tuple[str, str]:
    """Build the full prompt from template and content.

    Args:
        template: Goal template to use.
        content: Formatted post/comment content.
        config: Optional prompt builder config for modifiers.

    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    system_prompt = template.system_prompt

    if config:
        modifiers = build_prompt_modifiers(config)
        if modifiers:
            system_prompt += modifiers

    user_prompt = template.user_prompt_template.format(content=content)

    return system_prompt, user_prompt


# ============================================================================
# Content Formatting
# ============================================================================

def format_post_for_llm(
    title: str,
    body: str | None,
    url: str,
    subreddit: str,
    score: int,
    comments: list[dict],
    post_id: str = "",
) -> str:
    """Format a post and its comments for LLM input.

    Args:
        title: Post title.
        body: Post body text (may be None).
        url: Post URL.
        subreddit: Subreddit name.
        score: Post score.
        comments: List of comment dicts with 'body' and 'score' keys.
        post_id: Unique post identifier for reference.

    Returns:
        Formatted string for LLM.
    """
    # Extract short ID from URL for easy reference
    short_id = post_id or url.split("/comments/")[-1].split("/")[0] if "/comments/" in url else ""

    lines = [
        f"### [{short_id}] {title}",
        f"- Subreddit: r/{subreddit}",
        f"- Score: {score} points",
        f"- **SOURCE_URL**: {url}",
    ]

    if body:
        lines.append(f"\n{body}")

    if comments:
        lines.append("\n**Top Comments:**")
        for i, comment in enumerate(comments, 1):
            comment_preview = comment["body"][:500]
            if len(comment["body"]) > 500:
                comment_preview += "..."
            lines.append(f"\n{i}. [{comment['score']} points] {comment_preview}")

    # Repeat URL at end to reinforce it
    lines.append(f"\n[Source: {url}]")
    lines.append("\n---\n")

    return "\n".join(lines)


def format_posts_for_llm(posts_data: list[dict]) -> str:
    """Format multiple posts for LLM input.

    Args:
        posts_data: List of dicts with post info and comments.
            Each dict should have: title, body, url, subreddit, score, comments, post_id

    Returns:
        Combined formatted string.
    """
    formatted_posts = []

    for post in posts_data:
        formatted = format_post_for_llm(
            title=post["title"],
            body=post.get("body"),
            url=post["url"],
            subreddit=post["subreddit"],
            score=post["score"],
            comments=post.get("comments", []),
            post_id=post.get("post_id", ""),
        )
        formatted_posts.append(formatted)

    # Add URL reference index at the end for easy lookup
    url_index = ["\n\n## URL Reference (USE THESE EXACT URLS IN YOUR RESPONSE)"]
    for post in posts_data:
        short_id = post.get("post_id", "")[:8] if post.get("post_id") else ""
        url_index.append(f"- [{short_id}] {post['title'][:60]}... → {post['url']}")

    url_index.append("\nIMPORTANT: When citing sources, copy the URLs EXACTLY as shown above. Do NOT make up or modify URLs.")

    return "\n".join(formatted_posts) + "\n".join(url_index)


# ============================================================================
# Subreddit Discovery Prompt
# ============================================================================

SUBREDDIT_DISCOVERY_SYSTEM = """You are a Reddit expert who knows all major subreddits.
Your job is to suggest the most relevant subreddits for a specific research goal.

CRITICAL REQUIREMENTS:
1. **DIRECT RELEVANCE ONLY**: Every subreddit MUST be directly relevant to the goal's subject matter.
   - If the goal is about "legal regulations", suggest law/legal subreddits, NOT "lego" (sounds similar but unrelated)
   - If the goal is about "coding", suggest programming subreddits, NOT "coincollecting"
2. **Quality over quantity**: Only suggest subreddits you are confident are relevant
3. **Active communities**: Prefer subreddits with regular discussions
4. **Avoid NSFW or toxic communities**

VALIDATION: Before including a subreddit, ask yourself:
"Would posts in this subreddit actually contain information about the research goal?"
If the answer is no, DO NOT include it.

Return ONLY subreddit names, one per line, no descriptions or explanations."""

SUBREDDIT_DISCOVERY_TEMPLATE = """Research Goal: {goal}

Suggest exactly {count} subreddits that are DIRECTLY relevant to this research goal.

IMPORTANT:
- Each subreddit must contain discussions actually related to the research topic
- Do NOT suggest subreddits that merely sound similar to keywords in the goal
- Focus on topic relevance, not word similarity

Return as a simple list, one subreddit per line, without the r/ prefix:
"""


def build_subreddit_discovery_prompt(goal: str, count: int = 5) -> tuple[str, str]:
    """Build prompt for subreddit discovery.

    Args:
        goal: Research goal description.
        count: Number of subreddits to suggest.

    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    user_prompt = SUBREDDIT_DISCOVERY_TEMPLATE.format(goal=goal, count=count)
    return SUBREDDIT_DISCOVERY_SYSTEM, user_prompt
