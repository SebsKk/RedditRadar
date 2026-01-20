"""Post ranking and shortlisting for Reddit Radar.

This module handles local ranking of posts without using LLM,
to control costs and select the most relevant content.
"""

import math
import re
import time
from dataclasses import dataclass
from typing import Callable

from src.config import RankingConfig, get_config
from src.database import Post, Comment


@dataclass
class RankedPost:
    """A post with its ranking score and components."""
    post: Post
    total_score: float
    engagement_score: float
    recency_score: float
    keyword_score: float
    comments: list[Comment]


def tokenize(text: str) -> set[str]:
    """Tokenize text into lowercase words.

    Args:
        text: Text to tokenize.

    Returns:
        Set of lowercase word tokens.
    """
    if not text:
        return set()

    # Convert to lowercase and split on non-alphanumeric
    words = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
    return set(words)


def extract_keywords_from_goal(goal_prompt: str) -> set[str]:
    """Extract important keywords from goal prompt.

    Args:
        goal_prompt: The goal/analysis prompt.

    Returns:
        Set of keyword tokens.
    """
    # Common stop words to ignore
    stop_words = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
        'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
        'from', 'as', 'into', 'through', 'during', 'before', 'after', 'above',
        'below', 'between', 'under', 'again', 'further', 'then', 'once',
        'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each', 'few',
        'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
        'own', 'same', 'so', 'than', 'too', 'very', 'just', 'and', 'but',
        'if', 'or', 'because', 'until', 'while', 'this', 'that', 'these',
        'those', 'what', 'which', 'who', 'whom', 'any', 'both', 'each',
        'i', 'me', 'my', 'we', 'our', 'you', 'your', 'it', 'its', 'they',
        'them', 'their', 'find', 'identify', 'analyze', 'list', 'provide',
        'describe', 'explain', 'show', 'include', 'give', 'make', 'use',
    }

    tokens = tokenize(goal_prompt)
    keywords = tokens - stop_words

    # Filter out very short tokens
    keywords = {k for k in keywords if len(k) >= 3}

    return keywords


def compute_engagement_score(
    post_score: int,
    num_comments: int,
    weight_score: float = 1.0,
    weight_comments: float = 1.5,
) -> float:
    """Compute engagement score from post metrics.

    Uses log scale to prevent high-score posts from dominating.

    Args:
        post_score: Reddit score (upvotes - downvotes).
        num_comments: Number of comments.
        weight_score: Weight for score component.
        weight_comments: Weight for comments component.

    Returns:
        Engagement score (higher is better).
    """
    # Use log scale to compress range
    # Add 1 to avoid log(0)
    score_component = math.log1p(max(0, post_score)) * weight_score
    comment_component = math.log1p(max(0, num_comments)) * weight_comments

    return score_component + comment_component


def compute_recency_score(
    created_utc: float,
    half_life_hours: float = 24,
    now: float | None = None,
) -> float:
    """Compute recency score with exponential decay.

    Args:
        created_utc: Post creation timestamp.
        half_life_hours: Hours until score halves.
        now: Current timestamp (defaults to time.time()).

    Returns:
        Recency score between 0 and 1 (1 is most recent).
    """
    if now is None:
        now = time.time()

    age_hours = (now - created_utc) / 3600

    if age_hours <= 0:
        return 1.0

    # Exponential decay
    decay_rate = math.log(2) / half_life_hours
    score = math.exp(-decay_rate * age_hours)

    return score


def compute_keyword_score(
    title: str,
    body: str | None,
    keywords: set[str],
) -> float:
    """Compute keyword overlap score.

    Args:
        title: Post title.
        body: Post body (may be None).
        keywords: Set of target keywords.

    Returns:
        Keyword score (higher is better match).
    """
    if not keywords:
        return 0.0

    # Combine title and body
    text = title
    if body:
        text += " " + body

    post_tokens = tokenize(text)

    if not post_tokens:
        return 0.0

    # Count matching keywords
    matches = post_tokens & keywords
    match_count = len(matches)

    if match_count == 0:
        return 0.0

    # Normalize by sqrt of keyword count to not over-penalize short posts
    # Use Jaccard-like score
    score = match_count / math.sqrt(len(keywords))

    return score


def rank_post(
    post: Post,
    keywords: set[str],
    config: RankingConfig,
    now: float | None = None,
) -> tuple[float, float, float, float]:
    """Compute ranking scores for a single post.

    Args:
        post: Post to rank.
        keywords: Target keywords from goal.
        config: Ranking configuration.
        now: Current timestamp for recency calculation.

    Returns:
        Tuple of (total_score, engagement, recency, keyword).
    """
    engagement = compute_engagement_score(
        post.score,
        post.num_comments,
        config.weight_score,
        config.weight_comments,
    )

    recency = compute_recency_score(
        post.created_utc,
        config.recency_half_life,
        now,
    )

    keyword = compute_keyword_score(
        post.title,
        post.body,
        keywords,
    )

    # Weighted total
    total = (
        engagement +
        recency * config.weight_recency +
        keyword * config.weight_keywords
    )

    return total, engagement, recency, keyword


def rank_posts(
    posts: list[Post],
    goal_prompt: str,
    config: RankingConfig | None = None,
) -> list[RankedPost]:
    """Rank a list of posts by relevance to goal.

    Args:
        posts: Posts to rank.
        goal_prompt: Goal/analysis prompt for keyword extraction.
        config: Optional ranking configuration.

    Returns:
        List of RankedPost sorted by score (highest first).
    """
    if config is None:
        config = get_config().ranking

    keywords = extract_keywords_from_goal(goal_prompt)
    now = time.time()

    ranked = []
    for post in posts:
        total, engagement, recency, keyword = rank_post(
            post, keywords, config, now
        )

        ranked.append(RankedPost(
            post=post,
            total_score=total,
            engagement_score=engagement,
            recency_score=recency,
            keyword_score=keyword,
            comments=[],  # Comments added later
        ))

    # Sort by total score descending
    ranked.sort(key=lambda r: r.total_score, reverse=True)

    return ranked


def create_shortlist(
    posts: list[Post],
    goal_prompt: str,
    shortlist_size: int = 25,
    config: RankingConfig | None = None,
    min_per_subreddit: int = 1,
) -> list[RankedPost]:
    """Create a shortlist of top posts for LLM processing.

    Ensures minimum representation from each subreddit to avoid
    high-engagement subreddits dominating the analysis.

    Args:
        posts: All posts to consider.
        goal_prompt: Goal/analysis prompt.
        shortlist_size: Number of posts to include.
        config: Optional ranking configuration.
        min_per_subreddit: Minimum posts to include from each subreddit.

    Returns:
        List of top RankedPost.
    """
    ranked = rank_posts(posts, goal_prompt, config)

    if min_per_subreddit <= 0:
        return ranked[:shortlist_size]

    # Group by subreddit
    by_subreddit: dict[str, list[RankedPost]] = {}
    for r in ranked:
        sub = r.post.subreddit.lower()
        if sub not in by_subreddit:
            by_subreddit[sub] = []
        by_subreddit[sub].append(r)

    # First pass: ensure minimum from each subreddit
    shortlist = []
    used_post_ids = set()

    for sub, sub_posts in by_subreddit.items():
        for r in sub_posts[:min_per_subreddit]:
            if r.post.post_id not in used_post_ids:
                shortlist.append(r)
                used_post_ids.add(r.post.post_id)

    # Second pass: fill remaining slots with top-ranked posts
    remaining_slots = shortlist_size - len(shortlist)
    for r in ranked:
        if remaining_slots <= 0:
            break
        if r.post.post_id not in used_post_ids:
            shortlist.append(r)
            used_post_ids.add(r.post.post_id)
            remaining_slots -= 1

    # Sort final shortlist by score
    shortlist.sort(key=lambda r: r.total_score, reverse=True)

    return shortlist[:shortlist_size]


def attach_comments_to_ranked(
    ranked_posts: list[RankedPost],
    get_comments: Callable[[str], list[Comment]],
    comments_per_post: int = 5,
) -> None:
    """Attach comments to ranked posts in-place.

    Args:
        ranked_posts: List of ranked posts to update.
        get_comments: Function that returns comments for a post ID.
        comments_per_post: Number of comments to attach per post.
    """
    for ranked in ranked_posts:
        comments = get_comments(ranked.post.post_id)
        # Sort by score and take top N
        comments.sort(key=lambda c: c.score, reverse=True)
        ranked.comments = comments[:comments_per_post]


def ranked_to_llm_format(ranked_posts: list[RankedPost]) -> list[dict]:
    """Convert ranked posts to format expected by templates and analysis.

    Args:
        ranked_posts: List of ranked posts with comments.

    Returns:
        List of dicts ready for format_posts_for_llm and prepare_posts_for_analysis.
    """
    result = []

    for ranked in ranked_posts:
        post = ranked.post
        result.append({
            "post_id": post.post_id,
            "title": post.title,
            "body": post.body,
            "url": post.permalink,
            "subreddit": post.subreddit,
            "score": post.score,
            "num_comments": post.num_comments,
            "created_utc": post.created_utc,
            "comments": [
                {"body": c.body, "score": c.score}
                for c in ranked.comments
            ],
        })

    return result
