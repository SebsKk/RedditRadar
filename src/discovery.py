"""Smart subreddit discovery service for Reddit Radar.

This module provides intelligent subreddit discovery by combining:
- Reddit API search
- Subreddit validation and stats
- LLM-powered suggestions
"""

import logging
import re
from dataclasses import dataclass, asdict
from typing import Optional

from src.config import get_config

logger = logging.getLogger(__name__)


@dataclass
class SubredditCandidate:
    """A subreddit candidate with stats and relevance info."""
    name: str
    title: str
    description: str
    subscribers: int
    is_active: bool
    relevance_score: float
    source: str  # 'reddit_search', 'llm_suggestion', 'user_added'

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DiscoveryResult:
    """Result of subreddit discovery."""
    query: str
    subreddits: list[SubredditCandidate]
    llm_suggestions: list[str]
    error: Optional[str] = None


def calculate_activity_score(subscribers: int) -> float:
    """Calculate an activity score based on subscriber count.

    Higher subscribers generally means more activity.
    Uses logarithmic scale to prevent huge subs from dominating.
    """
    import math
    if subscribers <= 0:
        return 0.0
    # Log scale: 1K = 0.3, 10K = 0.4, 100K = 0.5, 1M = 0.6
    score = min(1.0, math.log10(subscribers) / 10)
    return round(score, 2)


def calculate_relevance_score(
    query_terms: list[str],
    name: str,
    title: str,
    description: str,
) -> float:
    """Calculate relevance score based on keyword matching.

    Uses whole-word matching to avoid false positives like "lego" matching "legal".

    Args:
        query_terms: List of search terms
        name: Subreddit name
        title: Subreddit title
        description: Subreddit description

    Returns:
        Score from 0.0 to 1.0
    """
    import re

    # Combine text for matching
    name_lower = name.lower()
    title_lower = title.lower()
    desc_lower = description.lower()

    # Filter out very short terms (less than 3 chars) to avoid false positives
    meaningful_terms = [t.lower() for t in query_terms if len(t) >= 3]

    if not meaningful_terms:
        return 0.3  # Default low score if no meaningful terms

    matches = 0
    total_weight = 0

    for term in meaningful_terms:
        # Use word boundary matching to avoid partial matches
        # e.g., "legal" should not match in "lego"
        pattern = rf'\b{re.escape(term)}\b'

        term_found = False

        # Exact match in name is strongest signal (weight 4)
        if re.search(pattern, name_lower):
            matches += 4
            term_found = True
        # Check if term is a substring of name (lower weight)
        elif term in name_lower and len(term) >= 4:
            matches += 2
            term_found = True

        # Match in title (weight 3)
        if re.search(pattern, title_lower):
            matches += 3
            term_found = True

        # Match in description (weight 2)
        if re.search(pattern, desc_lower):
            matches += 2
            term_found = True

        total_weight += 4 + 3 + 2  # Max possible for this term

    # Normalize to 0-1
    if total_weight == 0:
        return 0.0

    # Apply a minimum threshold - if very few terms matched, score low
    raw_score = matches / total_weight
    matched_terms = sum(1 for t in meaningful_terms if t in f"{name_lower} {title_lower} {desc_lower}")

    # Penalize if less than half of terms matched
    if len(meaningful_terms) > 2 and matched_terms < len(meaningful_terms) / 2:
        raw_score *= 0.5

    return min(1.0, raw_score)


def discover_subreddits_via_reddit(
    query: str,
    limit: int = 15,
    min_subscribers: int = 1000,
    exclude_names: set[str] | None = None,
) -> list[SubredditCandidate]:
    """Discover subreddits using Reddit's search API.

    Args:
        query: Search query (e.g., "machine learning", "startup ideas")
        limit: Maximum number of results
        min_subscribers: Minimum subscriber count for quality (default 1000)
        exclude_names: Set of subreddit names to exclude (lowercase)

    Returns:
        List of SubredditCandidate objects
    """
    from src.reddit_client import get_reddit_client, RedditClientError

    candidates = []
    query_terms = query.lower().split()
    exclude_names = exclude_names or set()

    try:
        client = get_reddit_client()

        # Search Reddit for subreddits - fetch 5x limit to account for filtering
        results = client.search_subreddits(query, limit=limit * 5)

        for sub_info in results:
            if not sub_info.public:
                continue

            # Skip already found subreddits
            if sub_info.name.lower() in exclude_names:
                continue

            # Calculate scores
            activity = calculate_activity_score(sub_info.subscribers)
            relevance = calculate_relevance_score(
                query_terms,
                sub_info.name,
                sub_info.title,
                sub_info.description,
            )

            # Combined score (weighted)
            combined_score = (relevance * 0.6) + (activity * 0.4)

            # Quality threshold: must have min_subscribers for active community
            if sub_info.subscribers >= min_subscribers:
                candidates.append(SubredditCandidate(
                    name=sub_info.name,
                    title=sub_info.title,
                    description=sub_info.description[:200] if sub_info.description else "",
                    subscribers=sub_info.subscribers,
                    is_active=sub_info.subscribers >= 5000,
                    relevance_score=round(combined_score, 2),
                    source="reddit_search",
                ))

    except RedditClientError as e:
        logger.error(f"Reddit search failed: {e}")
    except Exception as e:
        logger.error(f"Discovery error: {e}")

    # Sort by relevance score
    candidates.sort(key=lambda x: x.relevance_score, reverse=True)

    return candidates[:limit]


def discover_subreddits_via_llm(
    goal: str,
    count: int = 10,
) -> list[str]:
    """Use LLM to suggest relevant subreddits.

    Args:
        goal: Research goal description
        count: Number of suggestions

    Returns:
        List of subreddit names (without r/ prefix)
    """
    from src.llm_client import get_llm_client, LLMClientError
    from src.templates import build_subreddit_discovery_prompt

    try:
        client = get_llm_client()
        system_prompt, user_prompt = build_subreddit_discovery_prompt(goal, count)

        response = client.generate(user_prompt, system_prompt)

        # Parse response - expect one subreddit per line
        lines = response.content.strip().split('\n')
        suggestions = []

        for line in lines:
            # Clean the line
            name = line.strip()
            # Remove common prefixes
            name = re.sub(r'^[-*\d.)\s]+', '', name)  # Remove bullets, numbers
            name = re.sub(r'^r/', '', name, flags=re.IGNORECASE)  # Remove r/
            name = name.strip()

            # Validate format (alphanumeric with underscores)
            if name and re.match(r'^[a-zA-Z][a-zA-Z0-9_]{1,20}$', name):
                suggestions.append(name)

        return suggestions[:count]

    except LLMClientError as e:
        logger.error(f"LLM discovery failed: {e}")
        return []
    except Exception as e:
        logger.error(f"LLM discovery error: {e}")
        return []


def validate_subreddits(
    names: list[str],
    goal: str = "",
    min_subscribers: int = 1000,
    exclude_names: set[str] | None = None,
) -> list[SubredditCandidate]:
    """Validate a list of subreddit names and get their stats.

    Args:
        names: List of subreddit names to validate
        goal: Original goal for relevance scoring
        min_subscribers: Minimum subscriber count for quality (default 1000)
        exclude_names: Set of subreddit names to exclude (lowercase)

    Returns:
        List of validated SubredditCandidate objects
    """
    from src.reddit_client import get_reddit_client

    validated = []
    query_terms = goal.lower().split() if goal else []
    exclude_names = exclude_names or set()

    try:
        client = get_reddit_client()

        for name in names:
            # Skip already found subreddits
            if name.lower() in exclude_names:
                continue

            info = client.get_subreddit_info(name)
            # Quality threshold: must be public and have min_subscribers
            if info and info.public and info.subscribers >= min_subscribers:
                # Calculate relevance score for LLM suggestions
                if query_terms:
                    relevance = calculate_relevance_score(
                        query_terms,
                        info.name,
                        info.title,
                        info.description or "",
                    )
                    activity = calculate_activity_score(info.subscribers)
                    combined_score = (relevance * 0.6) + (activity * 0.4)
                else:
                    combined_score = 0.5  # Default if no goal provided

                validated.append(SubredditCandidate(
                    name=info.name,
                    title=info.title,
                    description=info.description[:200] if info.description else "",
                    subscribers=info.subscribers,
                    is_active=info.subscribers >= 5000,
                    relevance_score=round(combined_score, 2),
                    source="llm_suggestion",
                ))

    except Exception as e:
        logger.error(f"Validation error: {e}")

    return validated


def smart_discover(
    goal: str,
    target_count: int = 15,
    use_llm: bool = True,
    min_subscribers: int = 1000,
    max_iterations: int = 3,
) -> DiscoveryResult:
    """Perform smart subreddit discovery with iterative search until target is met.

    This function keeps searching until it finds the requested number of quality
    subreddits. It maintains quality standards (min_subscribers) while iterating
    with different strategies to find enough results.

    Strategies used:
    1. Reddit API search with the goal
    2. LLM suggestions (if enabled)
    3. Expanded Reddit search with broader terms
    4. Additional LLM rounds with context about what's missing

    Args:
        goal: Research goal or topic description
        target_count: Number of subreddits to find (will iterate until met)
        use_llm: Whether to use LLM for additional suggestions
        min_subscribers: Minimum subscriber count for quality (default 1000)
        max_iterations: Maximum search iterations to prevent infinite loops

    Returns:
        DiscoveryResult with ranked subreddit candidates
    """
    logger.info(f"[Discovery] Starting smart discovery for goal: '{goal}'")
    logger.info(f"[Discovery] Target: {target_count} subreddits, min_subscribers={min_subscribers}, use_llm={use_llm}")

    all_candidates: dict[str, SubredditCandidate] = {}
    llm_suggestions_raw: list[str] = []
    iteration = 0

    while len(all_candidates) < target_count and iteration < max_iterations:
        iteration += 1
        found_names = set(all_candidates.keys())
        needed = target_count - len(all_candidates)

        logger.info(f"[Discovery] === Iteration {iteration}/{max_iterations} === (have {len(all_candidates)}, need {needed} more)")

        # Strategy 1: Reddit API search
        if iteration == 1:
            # First iteration: direct search
            search_query = goal
        else:
            # Later iterations: try broader/related terms
            search_query = f"{goal} community discussion"

        logger.info(f"[Discovery] Searching Reddit for: '{search_query}'")
        reddit_results = discover_subreddits_via_reddit(
            search_query,
            limit=needed * 2,  # Request more to account for filtering
            min_subscribers=min_subscribers,
            exclude_names=found_names,
        )

        for candidate in reddit_results:
            key = candidate.name.lower()
            if key not in all_candidates:
                all_candidates[key] = candidate
                logger.info(f"[Discovery]   + r/{candidate.name} ({format_subscriber_count(candidate.subscribers)}) from Reddit")

        # Check if we have enough
        if len(all_candidates) >= target_count:
            logger.info(f"[Discovery] Target reached after Reddit search")
            break

        # Strategy 2: LLM suggestions (if enabled)
        if use_llm:
            found_names = set(all_candidates.keys())
            needed = target_count - len(all_candidates)

            # Build context for LLM about what we already have
            if iteration == 1:
                llm_prompt = goal
            else:
                existing = ", ".join(f"r/{name}" for name in list(all_candidates.keys())[:10])
                llm_prompt = f"{goal} (already found: {existing}, need {needed} more different ones)"

            logger.info(f"[Discovery] Asking LLM for {needed + 5} suggestions...")
            new_llm_suggestions = discover_subreddits_via_llm(llm_prompt, count=needed + 5)

            # Track all LLM suggestions
            llm_suggestions_raw.extend(new_llm_suggestions)

            # Filter to only new ones
            new_suggestions = [
                s for s in new_llm_suggestions
                if s.lower() not in found_names
            ]

            if new_suggestions:
                logger.info(f"[Discovery] Validating {len(new_suggestions)} LLM suggestions...")
                validated = validate_subreddits(
                    new_suggestions,
                    goal=goal,
                    min_subscribers=min_subscribers,
                    exclude_names=found_names,
                )

                for candidate in validated:
                    key = candidate.name.lower()
                    if key not in all_candidates:
                        all_candidates[key] = candidate
                        logger.info(f"[Discovery]   + r/{candidate.name} ({format_subscriber_count(candidate.subscribers)}) from LLM")

        # Check if we have enough
        if len(all_candidates) >= target_count:
            logger.info(f"[Discovery] Target reached after LLM suggestions")
            break

        # If still not enough, log and continue to next iteration
        if len(all_candidates) < target_count:
            logger.info(f"[Discovery] Still need {target_count - len(all_candidates)} more, continuing...")

    # Log final status
    if len(all_candidates) < target_count:
        logger.warning(f"[Discovery] Could only find {len(all_candidates)}/{target_count} subreddits after {iteration} iterations")

    # Step 3: Sort by relevance and activity
    final_candidates = sorted(
        all_candidates.values(),
        key=lambda x: (x.relevance_score, x.subscribers),
        reverse=True,
    )

    logger.info(f"[Discovery] Final result: {len(final_candidates)} subreddits")
    for i, c in enumerate(final_candidates[:10], 1):
        logger.info(f"[Discovery]   {i}. r/{c.name} ({format_subscriber_count(c.subscribers)}) - score={c.relevance_score:.2f}, source={c.source}")

    return DiscoveryResult(
        query=goal,
        subreddits=final_candidates,
        llm_suggestions=llm_suggestions_raw,
    )


def format_subscriber_count(count: int) -> str:
    """Format subscriber count for display.

    Args:
        count: Number of subscribers

    Returns:
        Formatted string (e.g., "1.2M", "45K", "500")
    """
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    elif count >= 1_000:
        return f"{count / 1_000:.0f}K"
    else:
        return str(count)
