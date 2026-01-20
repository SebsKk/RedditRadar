"""Reddit API client for Reddit Radar.

This module handles all Reddit API interactions using PRAW.
Uses application-only (read-only) OAuth authentication.
"""

import logging
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterator

logger = logging.getLogger(__name__)

# Suppress PRAW async environment warning (we're using sync PRAW in FastAPI's thread pool)
warnings.filterwarnings("ignore", message=".*asynchronous environment.*")

try:
    import praw
    from praw.models import Submission, Comment as PrawComment
    HAS_PRAW = True
except ImportError:
    HAS_PRAW = False
    praw = None

from src.config import RedditCredentials, get_config
from src.database import Post, Comment


@dataclass
class SubredditInfo:
    """Basic subreddit information."""
    name: str
    title: str
    description: str
    subscribers: int
    public: bool


class RedditClientError(Exception):
    """Base exception for Reddit client errors."""
    pass


class AuthenticationError(RedditClientError):
    """Raised when authentication fails."""
    pass


class RateLimitError(RedditClientError):
    """Raised when rate limit is exceeded."""
    pass


class RedditClient:
    """Reddit API client using PRAW for application-only OAuth.

    This client only requires client_id and client_secret (no username/password)
    and provides read-only access to public Reddit data.
    """

    def __init__(self, credentials: RedditCredentials | None = None):
        """Initialize Reddit client.

        Args:
            credentials: Reddit API credentials. If None, loads from config.

        Raises:
            ImportError: If PRAW is not installed.
            AuthenticationError: If credentials are invalid or missing.
        """
        if not HAS_PRAW:
            raise ImportError(
                "PRAW is required for Reddit API access. "
                "Install it with: pip install praw"
            )

        if credentials is None:
            config = get_config()
            credentials = config.reddit

        if not credentials.is_valid():
            raise AuthenticationError(
                "Reddit API credentials not configured. "
                "Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET environment variables."
            )

        self._credentials = credentials
        self._reddit = self._create_reddit_instance()

    def _create_reddit_instance(self) -> "praw.Reddit":
        """Create PRAW Reddit instance with application-only auth.

        Returns:
            Configured PRAW Reddit instance.
        """
        return praw.Reddit(
            client_id=self._credentials.client_id,
            client_secret=self._credentials.client_secret,
            user_agent=self._credentials.user_agent,
        )

    def verify_connection(self) -> bool:
        """Verify that the Reddit connection works.

        Returns:
            True if connection is successful.

        Raises:
            AuthenticationError: If authentication fails.
        """
        try:
            # Try to access a known subreddit
            self._reddit.subreddit("python").id
            return True
        except Exception as e:
            raise AuthenticationError(f"Failed to connect to Reddit: {e}")

    def get_subreddit_info(self, name: str) -> SubredditInfo | None:
        """Get information about a subreddit.

        Args:
            name: Subreddit name (without r/).

        Returns:
            SubredditInfo if found and public, None otherwise.
        """
        try:
            sub = self._reddit.subreddit(name)
            # Access an attribute to trigger the API call
            _ = sub.subscribers
            return SubredditInfo(
                name=sub.display_name,
                title=sub.title,
                description=sub.public_description or "",
                subscribers=sub.subscribers,
                public=sub.subreddit_type == "public",
            )
        except Exception:
            return None

    def subreddit_exists(self, name: str) -> bool:
        """Check if a subreddit exists and is accessible.

        Args:
            name: Subreddit name (without r/).

        Returns:
            True if subreddit exists and is public.
        """
        info = self.get_subreddit_info(name)
        return info is not None and info.public

    def search_subreddits(self, query: str, limit: int = 10) -> list[SubredditInfo]:
        """Search for subreddits by query.

        Args:
            query: Search query.
            limit: Maximum number of results.

        Returns:
            List of matching subreddits.
        """
        results = []
        try:
            for sub in self._reddit.subreddits.search(query, limit=limit):
                try:
                    if sub.subreddit_type == "public":
                        results.append(SubredditInfo(
                            name=sub.display_name,
                            title=sub.title,
                            description=sub.public_description or "",
                            subscribers=sub.subscribers,
                            public=True,
                        ))
                except Exception:
                    # Skip subreddits that cause errors
                    continue
        except Exception as e:
            # Log error but don't fail completely
            logger.warning(f" Subreddit search error: {e}")
        return results

    def get_top_posts(
        self,
        subreddit: str,
        time_filter: str = "day",
        limit: int = 10,
    ) -> list[Post]:
        """Get top posts from a subreddit.

        Args:
            subreddit: Subreddit name (without r/).
            time_filter: Time filter (hour, day, week, month, year, all).
            limit: Maximum number of posts.

        Returns:
            List of Post objects.
        """
        posts = []
        now = time.time()

        try:
            sub = self._reddit.subreddit(subreddit)
            for submission in sub.top(time_filter=time_filter, limit=limit):
                posts.append(self._submission_to_post(submission, now))
        except Exception as e:
            logger.warning(f" Failed to fetch posts from r/{subreddit}: {e}")

        return posts

    def get_rising_posts(self, subreddit: str, limit: int = 10) -> list[Post]:
        """Get rising posts from a subreddit.

        Args:
            subreddit: Subreddit name (without r/).
            limit: Maximum number of posts.

        Returns:
            List of Post objects.
        """
        posts = []
        now = time.time()

        try:
            sub = self._reddit.subreddit(subreddit)
            for submission in sub.rising(limit=limit):
                posts.append(self._submission_to_post(submission, now))
        except Exception as e:
            logger.warning(f" Failed to fetch rising posts from r/{subreddit}: {e}")

        return posts

    def get_hot_posts(self, subreddit: str, limit: int = 10) -> list[Post]:
        """Get hot posts from a subreddit.

        Args:
            subreddit: Subreddit name (without r/).
            limit: Maximum number of posts.

        Returns:
            List of Post objects.
        """
        posts = []
        now = time.time()

        try:
            sub = self._reddit.subreddit(subreddit)
            for submission in sub.hot(limit=limit):
                posts.append(self._submission_to_post(submission, now))
        except Exception as e:
            logger.warning(f" Failed to fetch hot posts from r/{subreddit}: {e}")

        return posts

    def get_new_posts(self, subreddit: str, limit: int = 10) -> list[Post]:
        """Get new posts from a subreddit.

        Args:
            subreddit: Subreddit name (without r/).
            limit: Maximum number of posts.

        Returns:
            List of Post objects.
        """
        posts = []
        now = time.time()

        try:
            sub = self._reddit.subreddit(subreddit)
            for submission in sub.new(limit=limit):
                posts.append(self._submission_to_post(submission, now))
        except Exception as e:
            logger.warning(f" Failed to fetch new posts from r/{subreddit}: {e}")

        return posts

    def get_posts_by_time_window(
        self,
        subreddit: str,
        hours: int = 24,
        feed_mode: str = "top",
        limit: int = 100,
    ) -> list[Post]:
        """Get posts from subreddit within a time window.

        Args:
            subreddit: Subreddit name (without r/).
            hours: Time window in hours (24, 72, 144).
            feed_mode: Feed mode (top, rising, hot, new).
            limit: Maximum number of posts to fetch (before filtering).

        Returns:
            List of posts within the time window.
        """
        # Calculate cutoff time
        cutoff = time.time() - (hours * 3600)

        # Determine time filter for top posts
        if hours <= 24:
            time_filter = "day"
        elif hours <= 72:
            time_filter = "week"
        else:
            time_filter = "week"  # Use week for up to 144h

        # Fetch posts based on feed mode
        if feed_mode == "top":
            posts = self.get_top_posts(subreddit, time_filter=time_filter, limit=limit)
        elif feed_mode == "rising":
            posts = self.get_rising_posts(subreddit, limit=limit)
        elif feed_mode == "hot":
            posts = self.get_hot_posts(subreddit, limit=limit)
        elif feed_mode == "new":
            posts = self.get_new_posts(subreddit, limit=limit)
        else:
            posts = self.get_top_posts(subreddit, time_filter=time_filter, limit=limit)

        # Filter by time window
        filtered = [p for p in posts if p.created_utc >= cutoff]
        return filtered

    def get_post_comments(
        self,
        post_id: str,
        limit: int = 5,
        sort: str = "top",
    ) -> list[Comment]:
        """Get top comments for a post.

        Args:
            post_id: Reddit post ID (without t3_ prefix).
            limit: Maximum number of comments.
            sort: Comment sort order (top, best, new, controversial).

        Returns:
            List of Comment objects.
        """
        comments = []
        now = time.time()

        try:
            # Handle post_id with or without t3_ prefix
            if post_id.startswith("t3_"):
                post_id = post_id[3:]

            submission = self._reddit.submission(id=post_id)
            submission.comment_sort = sort
            submission.comments.replace_more(limit=0)  # Don't load "more comments"

            for comment in submission.comments[:limit]:
                if hasattr(comment, "body"):  # Skip MoreComments objects
                    comments.append(Comment(
                        comment_id=comment.id,
                        post_id=post_id,
                        body=comment.body,
                        score=comment.score,
                        created_utc=comment.created_utc,
                    ))
        except Exception as e:
            logger.warning(f" Failed to fetch comments for post {post_id}: {e}")

        return comments

    def fetch_subreddit_data(
        self,
        subreddit: str,
        posts_limit: int = 10,
        comments_per_post: int = 5,
        time_window_hours: int = 24,
        feed_mode: str = "top",
    ) -> tuple[list[Post], list[Comment]]:
        """Fetch posts and comments from a subreddit.

        This is a convenience method that fetches posts and their comments
        in a single call.

        Args:
            subreddit: Subreddit name (without r/).
            posts_limit: Maximum number of posts.
            comments_per_post: Number of comments per post.
            time_window_hours: Time window in hours.
            feed_mode: Feed mode (top, rising, hot, new).

        Returns:
            Tuple of (posts, comments).
        """
        # Fetch posts
        posts = self.get_posts_by_time_window(
            subreddit=subreddit,
            hours=time_window_hours,
            feed_mode=feed_mode,
            limit=posts_limit * 2,  # Fetch extra to account for filtering
        )[:posts_limit]

        # Fetch comments for each post
        all_comments = []
        for post in posts:
            comments = self.get_post_comments(
                post_id=post.post_id,
                limit=comments_per_post,
            )
            all_comments.extend(comments)

        return posts, all_comments

    def _submission_to_post(self, submission: "Submission", fetched_at: float) -> Post:
        """Convert PRAW Submission to Post dataclass.

        Args:
            submission: PRAW Submission object.
            fetched_at: Timestamp when fetched.

        Returns:
            Post dataclass.
        """
        # Get body text (selftext for text posts, empty for links)
        body = submission.selftext if submission.is_self else None

        return Post(
            post_id=submission.id,
            subreddit=submission.subreddit.display_name,
            title=submission.title,
            body=body,
            url=submission.url,
            permalink=f"https://reddit.com{submission.permalink}",
            score=submission.score,
            num_comments=submission.num_comments,
            created_utc=submission.created_utc,
            fetched_at=fetched_at,
        )


def get_reddit_client(credentials: RedditCredentials | None = None) -> RedditClient:
    """Get a Reddit client instance.

    Args:
        credentials: Optional credentials. If None, loads from config.

    Returns:
        RedditClient instance.
    """
    return RedditClient(credentials)


# Helper functions for common operations
def fetch_posts_from_subreddits(
    subreddits: list[str],
    posts_per_sub: int = 10,
    time_window_hours: int = 24,
    feed_mode: str = "top",
    comments_per_post: int = 5,
) -> tuple[list[Post], list[Comment]]:
    """Fetch posts and comments from multiple subreddits.

    Args:
        subreddits: List of subreddit names.
        posts_per_sub: Posts to fetch per subreddit.
        time_window_hours: Time window in hours.
        feed_mode: Feed mode (top, rising, hot, new).
        comments_per_post: Comments to fetch per post.

    Returns:
        Tuple of (all_posts, all_comments).
    """
    client = get_reddit_client()

    all_posts = []
    all_comments = []

    for subreddit in subreddits:
        posts, comments = client.fetch_subreddit_data(
            subreddit=subreddit,
            posts_limit=posts_per_sub,
            comments_per_post=comments_per_post,
            time_window_hours=time_window_hours,
            feed_mode=feed_mode,
        )
        all_posts.extend(posts)
        all_comments.extend(comments)

    return all_posts, all_comments
