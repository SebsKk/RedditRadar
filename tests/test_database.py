"""Tests for the database module."""

import os
import sys
import tempfile
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
    pytest = None

from src.database import (
    Comment,
    Database,
    Digest,
    LLMCache,
    Post,
    Run,
    RunItem,
)


# Only define pytest fixtures if pytest is available
if HAS_PYTEST:
    @pytest.fixture
    def temp_db():
        """Create a temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        db = Database(db_path)
        db.initialize()

        yield db

        # Cleanup
        os.unlink(db_path)

    @pytest.fixture
    def sample_post() -> Post:
        """Create a sample post for testing."""
        return Post(
            post_id="abc123",
            subreddit="python",
            title="Test Post Title",
            body="This is the body of the test post.",
            url="https://reddit.com/r/python/comments/abc123",
            permalink="/r/python/comments/abc123/test_post",
            score=100,
            num_comments=25,
            created_utc=time.time(),
            fetched_at=time.time(),
        )

    @pytest.fixture
    def sample_comment() -> Comment:
        """Create a sample comment for testing."""
        return Comment(
            comment_id="comment123",
            post_id="abc123",
            body="This is a test comment.",
            score=50,
            created_utc=time.time(),
        )


class TestDatabaseInitialization:
    """Tests for database initialization."""

    def test_create_database(self, temp_db):
        """Test database creation."""
        assert temp_db.db_path.exists()

    def test_schema_created(self, temp_db):
        """Test that all tables are created."""
        stats = temp_db.get_stats()
        expected_tables = ["posts", "comments", "runs", "run_items", "digests", "llm_cache"]

        for table in expected_tables:
            assert table in stats
            assert stats[table] == 0


class TestPostOperations:
    """Tests for post CRUD operations."""

    def test_insert_post(self, temp_db, sample_post):
        """Test inserting a post."""
        temp_db.insert_post(sample_post)

        retrieved = temp_db.get_post(sample_post.post_id)
        assert retrieved is not None
        assert retrieved.post_id == sample_post.post_id
        assert retrieved.title == sample_post.title

    def test_insert_posts_batch(self, temp_db):
        """Test batch inserting posts."""
        posts = [
            Post(
                post_id=f"post_{i}",
                subreddit="python",
                title=f"Post {i}",
                body=f"Body {i}",
                url=f"https://reddit.com/r/python/comments/post_{i}",
                permalink=f"/r/python/comments/post_{i}",
                score=100 - i,
                num_comments=10,
                created_utc=time.time(),
                fetched_at=time.time(),
            )
            for i in range(5)
        ]

        temp_db.insert_posts(posts)
        stats = temp_db.get_stats()
        assert stats["posts"] == 5

    def test_post_exists(self, temp_db, sample_post):
        """Test checking if post exists."""
        assert not temp_db.post_exists(sample_post.post_id)

        temp_db.insert_post(sample_post)
        assert temp_db.post_exists(sample_post.post_id)

    def test_get_posts_by_subreddit(self, temp_db):
        """Test getting posts by subreddit."""
        posts = [
            Post(
                post_id=f"post_{i}",
                subreddit="python" if i < 3 else "javascript",
                title=f"Post {i}",
                body=None,
                url=f"https://reddit.com/r/test/comments/post_{i}",
                permalink=f"/r/test/comments/post_{i}",
                score=100 - i,
                num_comments=10,
                created_utc=time.time(),
                fetched_at=time.time(),
            )
            for i in range(5)
        ]
        temp_db.insert_posts(posts)

        python_posts = temp_db.get_posts_by_subreddit("python")
        assert len(python_posts) == 3

        js_posts = temp_db.get_posts_by_subreddit("javascript")
        assert len(js_posts) == 2

    def test_get_posts_with_limit(self, temp_db):
        """Test limiting post results."""
        posts = [
            Post(
                post_id=f"post_{i}",
                subreddit="python",
                title=f"Post {i}",
                body=None,
                url=f"https://reddit.com/r/python/comments/post_{i}",
                permalink=f"/r/python/comments/post_{i}",
                score=100 - i,
                num_comments=10,
                created_utc=time.time(),
                fetched_at=time.time(),
            )
            for i in range(10)
        ]
        temp_db.insert_posts(posts)

        limited = temp_db.get_posts_by_subreddit("python", limit=5)
        assert len(limited) == 5

    def test_upsert_post(self, temp_db, sample_post):
        """Test that inserting duplicate post updates it."""
        temp_db.insert_post(sample_post)

        # Modify and reinsert
        sample_post.score = 200
        temp_db.insert_post(sample_post)

        retrieved = temp_db.get_post(sample_post.post_id)
        assert retrieved.score == 200

        # Should still be only 1 post
        stats = temp_db.get_stats()
        assert stats["posts"] == 1


class TestCommentOperations:
    """Tests for comment CRUD operations."""

    def test_insert_comment(self, temp_db, sample_post, sample_comment):
        """Test inserting a comment."""
        temp_db.insert_post(sample_post)
        temp_db.insert_comment(sample_comment)

        comments = temp_db.get_comments_for_post(sample_post.post_id)
        assert len(comments) == 1
        assert comments[0].body == sample_comment.body

    def test_insert_comments_batch(self, temp_db, sample_post):
        """Test batch inserting comments."""
        temp_db.insert_post(sample_post)

        comments = [
            Comment(
                comment_id=f"comment_{i}",
                post_id=sample_post.post_id,
                body=f"Comment {i}",
                score=50 - i,
                created_utc=time.time(),
            )
            for i in range(5)
        ]
        temp_db.insert_comments(comments)

        retrieved = temp_db.get_comments_for_post(sample_post.post_id)
        assert len(retrieved) == 5

    def test_comments_ordered_by_score(self, temp_db, sample_post):
        """Test that comments are ordered by score."""
        temp_db.insert_post(sample_post)

        comments = [
            Comment(
                comment_id=f"comment_{i}",
                post_id=sample_post.post_id,
                body=f"Comment {i}",
                score=i * 10,  # Increasing scores
                created_utc=time.time(),
            )
            for i in range(5)
        ]
        temp_db.insert_comments(comments)

        retrieved = temp_db.get_comments_for_post(sample_post.post_id)
        # Should be descending by score
        assert retrieved[0].score > retrieved[-1].score


class TestRunOperations:
    """Tests for run CRUD operations."""

    def test_insert_run(self, temp_db):
        """Test inserting a run."""
        run = Run(
            run_id="run_123",
            run_date="2025-01-15",
            goal_template="content_ideas",
            goal_prompt_hash="abc123hash",
            settings_json='{"key": "value"}',
            subreddits_json='["python", "startup"]',
            created_at=time.time(),
        )
        temp_db.insert_run(run)

        retrieved = temp_db.get_run(run.run_id)
        assert retrieved is not None
        assert retrieved.goal_template == "content_ideas"

    def test_get_runs_by_date(self, temp_db):
        """Test getting runs by date."""
        runs = [
            Run(
                run_id=f"run_{i}",
                run_date="2025-01-15" if i < 2 else "2025-01-16",
                goal_template="content_ideas",
                goal_prompt_hash=f"hash_{i}",
                settings_json="{}",
                subreddits_json="[]",
                created_at=time.time() + i,
            )
            for i in range(4)
        ]
        for run in runs:
            temp_db.insert_run(run)

        jan15_runs = temp_db.get_runs_by_date("2025-01-15")
        assert len(jan15_runs) == 2

        jan16_runs = temp_db.get_runs_by_date("2025-01-16")
        assert len(jan16_runs) == 2

    def test_get_recent_runs(self, temp_db):
        """Test getting recent runs."""
        for i in range(5):
            run = Run(
                run_id=f"run_{i}",
                run_date="2025-01-15",
                goal_template="content_ideas",
                goal_prompt_hash=f"hash_{i}",
                settings_json="{}",
                subreddits_json="[]",
                created_at=time.time() + i,
            )
            temp_db.insert_run(run)

        recent = temp_db.get_recent_runs(limit=3)
        assert len(recent) == 3
        # Most recent first
        assert recent[0].run_id == "run_4"


class TestRunItemOperations:
    """Tests for run item operations."""

    def test_insert_run_items(self, temp_db, sample_post):
        """Test inserting run items."""
        temp_db.insert_post(sample_post)

        run = Run(
            run_id="run_123",
            run_date="2025-01-15",
            goal_template="content_ideas",
            goal_prompt_hash="hash",
            settings_json="{}",
            subreddits_json="[]",
            created_at=time.time(),
        )
        temp_db.insert_run(run)

        items = [RunItem(run_id="run_123", post_id=sample_post.post_id, rank_score=0.85)]
        temp_db.insert_run_items(items)

        retrieved = temp_db.get_run_items("run_123")
        assert len(retrieved) == 1
        assert retrieved[0].rank_score == 0.85

    def test_get_posts_for_run(self, temp_db):
        """Test getting full post data for a run."""
        posts = [
            Post(
                post_id=f"post_{i}",
                subreddit="python",
                title=f"Post {i}",
                body=None,
                url=f"https://reddit.com/r/python/comments/post_{i}",
                permalink=f"/r/python/comments/post_{i}",
                score=100,
                num_comments=10,
                created_utc=time.time(),
                fetched_at=time.time(),
            )
            for i in range(3)
        ]
        temp_db.insert_posts(posts)

        run = Run(
            run_id="run_123",
            run_date="2025-01-15",
            goal_template="content_ideas",
            goal_prompt_hash="hash",
            settings_json="{}",
            subreddits_json="[]",
            created_at=time.time(),
        )
        temp_db.insert_run(run)

        items = [
            RunItem(run_id="run_123", post_id=f"post_{i}", rank_score=1.0 - i * 0.1)
            for i in range(3)
        ]
        temp_db.insert_run_items(items)

        run_posts = temp_db.get_posts_for_run("run_123")
        assert len(run_posts) == 3


class TestDigestOperations:
    """Tests for digest operations."""

    def test_insert_digest(self, temp_db):
        """Test inserting a digest."""
        run = Run(
            run_id="run_123",
            run_date="2025-01-15",
            goal_template="content_ideas",
            goal_prompt_hash="hash",
            settings_json="{}",
            subreddits_json="[]",
            created_at=time.time(),
        )
        temp_db.insert_run(run)

        digest = Digest(
            run_id="run_123",
            model_mode="single",
            models_json='["deepseek-chat"]',
            markdown="# Test Digest\n\nContent here.",
            card_json='{"title": "Test"}',
            created_at=time.time(),
        )
        temp_db.insert_digest(digest)

        retrieved = temp_db.get_digest("run_123")
        assert retrieved is not None
        assert "Test Digest" in retrieved.markdown


class TestCacheOperations:
    """Tests for LLM cache operations."""

    def test_compute_cache_key(self):
        """Test cache key computation."""
        key1 = Database.compute_cache_key(
            goal_prompt="Find startup ideas",
            post_ids=["post_1", "post_2"],
            comment_ids=["comment_1"],
            model="deepseek-chat",
        )

        # Same inputs should give same key
        key2 = Database.compute_cache_key(
            goal_prompt="Find startup ideas",
            post_ids=["post_2", "post_1"],  # Different order
            comment_ids=["comment_1"],
            model="deepseek-chat",
        )
        assert key1 == key2

        # Different inputs should give different key
        key3 = Database.compute_cache_key(
            goal_prompt="Find trends",  # Different prompt
            post_ids=["post_1", "post_2"],
            comment_ids=["comment_1"],
            model="deepseek-chat",
        )
        assert key1 != key3

    def test_cache_response(self, temp_db):
        """Test caching LLM response."""
        cache_key = "test_cache_key_123"
        temp_db.cache_response(
            cache_key=cache_key,
            model="deepseek-chat",
            input_hash="input_hash_123",
            output_markdown="# Cached Response",
        )

        cached = temp_db.get_cached_response(cache_key)
        assert cached is not None
        assert cached.output_markdown == "# Cached Response"

    def test_cache_miss(self, temp_db):
        """Test cache miss returns None."""
        cached = temp_db.get_cached_response("nonexistent_key")
        assert cached is None


class TestUtilityOperations:
    """Tests for utility operations."""

    def test_get_stats(self, temp_db, sample_post, sample_comment):
        """Test getting database statistics."""
        temp_db.insert_post(sample_post)
        temp_db.insert_comment(sample_comment)

        stats = temp_db.get_stats()
        assert stats["posts"] == 1
        assert stats["comments"] == 1
        assert stats["runs"] == 0

    def test_vacuum(self, temp_db):
        """Test vacuum runs without error."""
        temp_db.vacuum()  # Should not raise


# Simple test that can run without pytest
def test_basic_database():
    """Basic database test that runs without pytest."""
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        db = Database(db_path)
        db.initialize()

        # Test post insert
        post = Post(
            post_id="test123",
            subreddit="python",
            title="Test",
            body="Body",
            url="https://reddit.com/test",
            permalink="/r/python/test",
            score=100,
            num_comments=10,
            created_utc=time.time(),
            fetched_at=time.time(),
        )
        db.insert_post(post)

        # Test retrieval
        retrieved = db.get_post("test123")
        assert retrieved is not None
        assert retrieved.title == "Test"

        # Test stats
        stats = db.get_stats()
        assert stats["posts"] == 1

        print("All basic database tests passed!")

    finally:
        os.unlink(db_path)


if __name__ == "__main__":
    test_basic_database()
