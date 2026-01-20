"""Tests for the Reddit client module."""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False

try:
    import praw
    HAS_PRAW = True
except ImportError:
    HAS_PRAW = False

from src.database import Post, Comment


def test_imports():
    """Test that all imports work."""
    from src.reddit_client import (
        RedditClient,
        RedditClientError,
        AuthenticationError,
        SubredditInfo,
    )
    assert RedditClient is not None
    assert SubredditInfo is not None


def test_reddit_client_without_praw():
    """Test that RedditClient raises ImportError without PRAW."""
    if HAS_PRAW:
        # PRAW is installed, this test doesn't apply
        assert True
        return

    from src.reddit_client import RedditClient
    from src.config import RedditCredentials

    credentials = RedditCredentials(
        client_id="test",
        client_secret="test",
        user_agent="test",
    )
    with pytest.raises(ImportError):
        RedditClient(credentials)


def test_reddit_client_with_praw():
    """Test RedditClient with PRAW installed (requires valid credentials)."""
    if not HAS_PRAW:
        pytest.skip("PRAW not installed")

    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()

    from src.reddit_client import RedditClient
    from src.config import RedditCredentials

    client_id = os.getenv("REDDIT_CLIENT_ID", "")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET", "")

    if not client_id or not client_secret:
        pytest.skip("Reddit credentials not configured")

    credentials = RedditCredentials(
        client_id=client_id,
        client_secret=client_secret,
        user_agent="RedditRadar/1.0 (test)",
    )

    client = RedditClient(credentials)
    assert client is not None

    # Test connection
    assert client.verify_connection()

    # Test subreddit info
    info = client.get_subreddit_info("python")
    assert info is not None
    assert info.name == "Python"

    # Test subreddit search
    results = client.search_subreddits("programming", limit=3)
    assert len(results) >= 1

    # Test fetching posts
    posts = client.get_top_posts("python", time_filter="day", limit=3)
    assert isinstance(posts, list)

    # Test fetching comments (if posts exist)
    if posts:
        comments = client.get_post_comments(posts[0].post_id, limit=3)
        assert isinstance(comments, list)


def test_post_dataclass():
    """Test Post dataclass creation."""
    import time

    post = Post(
        post_id="abc123",
        subreddit="python",
        title="Test Post",
        body="Test body",
        url="https://reddit.com/test",
        permalink="https://reddit.com/r/python/abc123",
        score=100,
        num_comments=10,
        created_utc=time.time(),
        fetched_at=time.time(),
    )

    assert post.post_id == "abc123"
    assert post.subreddit == "python"
    assert post.score == 100


def test_comment_dataclass():
    """Test Comment dataclass creation."""
    import time

    comment = Comment(
        comment_id="xyz789",
        post_id="abc123",
        body="Test comment",
        score=50,
        created_utc=time.time(),
    )

    assert comment.comment_id == "xyz789"
    assert comment.post_id == "abc123"
    assert comment.score == 50


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Reddit Client Tests")
    print("=" * 60)

    tests = [
        ("Imports", test_imports),
        ("Post dataclass", test_post_dataclass),
        ("Comment dataclass", test_comment_dataclass),
        ("Reddit client (no PRAW)", test_reddit_client_without_praw),
        ("Reddit client (with PRAW)", test_reddit_client_with_praw),
    ]

    results = []
    for name, test_func in tests:
        print(f"\n--- {name} ---")
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {status}: {name}")

    all_passed = all(r for _, r in results)
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
