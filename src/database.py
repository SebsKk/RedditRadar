"""SQLite database layer for Reddit Radar.

This module handles all database operations including:
- Schema creation and migrations
- CRUD operations for posts, comments, runs, and digests
- Caching for LLM responses
"""

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Generator


@dataclass
class Post:
    """Reddit post data."""
    post_id: str
    subreddit: str
    title: str
    body: str | None
    url: str
    permalink: str
    score: int
    num_comments: int
    created_utc: float
    fetched_at: float


@dataclass
class Comment:
    """Reddit comment data."""
    comment_id: str
    post_id: str
    body: str
    score: int
    created_utc: float


@dataclass
class Run:
    """Analysis run metadata."""
    run_id: str
    run_date: str
    goal_template: str
    goal_prompt_hash: str
    settings_json: str
    subreddits_json: str
    created_at: float


@dataclass
class RunItem:
    """Post included in a run."""
    run_id: str
    post_id: str
    rank_score: float


@dataclass
class Digest:
    """Generated digest."""
    run_id: str
    model_mode: str
    models_json: str
    markdown: str
    card_json: str | None
    created_at: float


@dataclass
class LLMCache:
    """Cached LLM response."""
    cache_key: str
    model: str
    input_hash: str
    output_markdown: str
    created_at: float


@dataclass
class RunCluster:
    """Cluster metrics for a run (for Radar tracking)."""
    run_id: str
    cluster_id: str  # Stable ID for comparison (e.g., "marketing_growth")
    cluster_name: str  # Display name (e.g., "Marketing & Growth")
    frequency_posts: int
    engagement_score_sum: int
    engagement_comments_sum: int
    avg_age_hours: float  # Renamed from recency_avg_hours (it's actually age, not freshness)
    intensity_score: float
    proxy_score: float
    top_urls_json: str
    value_score: float = 0.0  # LLM-assigned value score (0-10) for heatmap


@dataclass
class SubredditPreset:
    """Saved subreddit combination preset."""
    preset_id: str
    name: str
    description: str
    subreddits_json: str  # JSON array of subreddit names
    template: str  # Default template to use with this preset
    created_at: float
    updated_at: float


@dataclass
class Schedule:
    """Scheduled analysis job."""
    schedule_id: str
    name: str
    preset_id: str | None  # Reference to subreddit preset
    template: str
    subreddits_json: str  # Fallback if no preset
    posts_per_sub: int
    time_window_hours: int
    cron_expression: str  # e.g., "0 9 * * *" for daily at 9am
    enabled: bool
    last_run_at: float | None
    next_run_at: float | None
    created_at: float


@dataclass
class Notification:
    """Notification configuration."""
    notification_id: str
    name: str
    notification_type: str  # "webhook" or "email"
    config_json: str  # {"url": "..."} or {"email": "...", "smtp_host": ...}
    trigger_on: str  # "run_complete", "run_failed", "pattern_detected"
    enabled: bool
    created_at: float


# SQL Schema
SCHEMA = """
-- Posts fetched from Reddit
CREATE TABLE IF NOT EXISTS posts (
    post_id TEXT PRIMARY KEY,
    subreddit TEXT NOT NULL,
    title TEXT NOT NULL,
    body TEXT,
    url TEXT NOT NULL,
    permalink TEXT NOT NULL,
    score INTEGER DEFAULT 0,
    num_comments INTEGER DEFAULT 0,
    created_utc REAL NOT NULL,
    fetched_at REAL NOT NULL
);

-- Comments for posts
CREATE TABLE IF NOT EXISTS comments (
    comment_id TEXT PRIMARY KEY,
    post_id TEXT NOT NULL,
    body TEXT NOT NULL,
    score INTEGER DEFAULT 0,
    created_utc REAL NOT NULL,
    FOREIGN KEY (post_id) REFERENCES posts(post_id)
);

-- Analysis runs
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    run_date TEXT NOT NULL,
    goal_template TEXT NOT NULL,
    goal_prompt_hash TEXT NOT NULL,
    settings_json TEXT NOT NULL,
    subreddits_json TEXT NOT NULL,
    created_at REAL NOT NULL
);

-- Posts used in each run
CREATE TABLE IF NOT EXISTS run_items (
    run_id TEXT NOT NULL,
    post_id TEXT NOT NULL,
    rank_score REAL,
    PRIMARY KEY (run_id, post_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id),
    FOREIGN KEY (post_id) REFERENCES posts(post_id)
);

-- Generated digests
CREATE TABLE IF NOT EXISTS digests (
    run_id TEXT PRIMARY KEY,
    model_mode TEXT NOT NULL,
    models_json TEXT NOT NULL,
    markdown TEXT NOT NULL,
    card_json TEXT,
    created_at REAL NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

-- LLM response cache
CREATE TABLE IF NOT EXISTS llm_cache (
    cache_key TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    input_hash TEXT NOT NULL,
    output_markdown TEXT NOT NULL,
    created_at REAL NOT NULL
);

-- Cluster metrics per run (for Radar tracking and heatmap)
CREATE TABLE IF NOT EXISTS run_clusters (
    run_id TEXT NOT NULL,
    cluster_id TEXT NOT NULL,
    cluster_name TEXT NOT NULL,
    frequency_posts INTEGER DEFAULT 0,
    engagement_score_sum INTEGER DEFAULT 0,
    engagement_comments_sum INTEGER DEFAULT 0,
    avg_age_hours REAL DEFAULT 0,
    intensity_score REAL DEFAULT 0,
    proxy_score REAL DEFAULT 0,
    value_score REAL DEFAULT 0,
    top_urls_json TEXT,
    PRIMARY KEY (run_id, cluster_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

-- Subreddit presets (saved subreddit combinations)
CREATE TABLE IF NOT EXISTS subreddit_presets (
    preset_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    subreddits_json TEXT NOT NULL,
    template TEXT NOT NULL DEFAULT 'content_ideas',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

-- Scheduled analysis jobs
CREATE TABLE IF NOT EXISTS schedules (
    schedule_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    preset_id TEXT,
    template TEXT NOT NULL,
    subreddits_json TEXT NOT NULL,
    posts_per_sub INTEGER DEFAULT 10,
    time_window_hours INTEGER DEFAULT 24,
    cron_expression TEXT NOT NULL,
    enabled INTEGER DEFAULT 1,
    last_run_at REAL,
    next_run_at REAL,
    created_at REAL NOT NULL,
    FOREIGN KEY (preset_id) REFERENCES subreddit_presets(preset_id)
);

-- Notification configurations
CREATE TABLE IF NOT EXISTS notifications (
    notification_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    notification_type TEXT NOT NULL,
    config_json TEXT NOT NULL,
    trigger_on TEXT NOT NULL DEFAULT 'run_complete',
    enabled INTEGER DEFAULT 1,
    created_at REAL NOT NULL
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_posts_subreddit ON posts(subreddit);
CREATE INDEX IF NOT EXISTS idx_posts_created_utc ON posts(created_utc);
CREATE INDEX IF NOT EXISTS idx_comments_post_id ON comments(post_id);
CREATE INDEX IF NOT EXISTS idx_runs_run_date ON runs(run_date);
CREATE INDEX IF NOT EXISTS idx_run_clusters_run_id ON run_clusters(run_id);
CREATE INDEX IF NOT EXISTS idx_schedules_enabled ON schedules(enabled);
CREATE INDEX IF NOT EXISTS idx_schedules_next_run ON schedules(next_run_at);
"""


class Database:
    """SQLite database manager for Reddit Radar."""

    def __init__(self, db_path: str | Path = "./reddit_radar.db"):
        """Initialize database connection.

        Args:
            db_path: Path to SQLite database file.
        """
        self.db_path = Path(db_path)
        self._ensure_directory()

    def _ensure_directory(self) -> None:
        """Ensure the database directory exists."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for database connections.

        Yields:
            SQLite connection with row factory set.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def initialize(self) -> None:
        """Initialize database schema and run migrations."""
        with self.connection() as conn:
            conn.executescript(SCHEMA)
        self._run_migrations()

    def _run_migrations(self) -> None:
        """Run database migrations for schema updates.

        This handles adding new columns to existing tables without
        requiring users to delete their database.
        """
        with self.connection() as conn:
            # Get current columns in run_clusters
            cursor = conn.execute("PRAGMA table_info(run_clusters)")
            columns = {row["name"] for row in cursor.fetchall()}

            # Migration 1: Add cluster_id column if missing
            if "cluster_id" not in columns:
                conn.execute("""
                    ALTER TABLE run_clusters
                    ADD COLUMN cluster_id TEXT DEFAULT ''
                """)
                conn.execute("""
                    UPDATE run_clusters
                    SET cluster_id = LOWER(REPLACE(cluster_name, ' ', '_'))
                    WHERE cluster_id = '' OR cluster_id IS NULL
                """)

            # Migration 2: Add value_score column if missing (for heatmap)
            if "value_score" not in columns:
                conn.execute("""
                    ALTER TABLE run_clusters
                    ADD COLUMN value_score REAL DEFAULT 0
                """)

            # Migration 3: Add avg_age_hours if missing (renamed from recency_avg_hours)
            if "avg_age_hours" not in columns:
                conn.execute("""
                    ALTER TABLE run_clusters
                    ADD COLUMN avg_age_hours REAL DEFAULT 0
                """)
                # Copy data from old column if it exists
                if "recency_avg_hours" in columns:
                    conn.execute("""
                        UPDATE run_clusters
                        SET avg_age_hours = recency_avg_hours
                        WHERE avg_age_hours = 0 OR avg_age_hours IS NULL
                    """)

    # -------------------------------------------------------------------------
    # Post operations
    # -------------------------------------------------------------------------

    def insert_post(self, post: Post) -> None:
        """Insert or update a post.

        Args:
            post: Post data to insert.
        """
        with self.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO posts
                (post_id, subreddit, title, body, url, permalink, score,
                 num_comments, created_utc, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    post.post_id,
                    post.subreddit,
                    post.title,
                    post.body,
                    post.url,
                    post.permalink,
                    post.score,
                    post.num_comments,
                    post.created_utc,
                    post.fetched_at,
                ),
            )

    def insert_posts(self, posts: list[Post]) -> None:
        """Insert multiple posts.

        Args:
            posts: List of posts to insert.
        """
        with self.connection() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO posts
                (post_id, subreddit, title, body, url, permalink, score,
                 num_comments, created_utc, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        p.post_id,
                        p.subreddit,
                        p.title,
                        p.body,
                        p.url,
                        p.permalink,
                        p.score,
                        p.num_comments,
                        p.created_utc,
                        p.fetched_at,
                    )
                    for p in posts
                ],
            )

    def get_post(self, post_id: str) -> Post | None:
        """Get a post by ID.

        Args:
            post_id: Reddit post ID.

        Returns:
            Post if found, None otherwise.
        """
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM posts WHERE post_id = ?", (post_id,)
            ).fetchone()
            if row:
                return Post(**dict(row))
            return None

    def get_posts_by_subreddit(
        self,
        subreddit: str,
        since_utc: float | None = None,
        limit: int | None = None,
    ) -> list[Post]:
        """Get posts from a subreddit.

        Args:
            subreddit: Subreddit name (without r/).
            since_utc: Only return posts created after this timestamp.
            limit: Maximum number of posts to return.

        Returns:
            List of posts.
        """
        query = "SELECT * FROM posts WHERE subreddit = ?"
        params: list = [subreddit]

        if since_utc is not None:
            query += " AND created_utc >= ?"
            params.append(since_utc)

        query += " ORDER BY score DESC"

        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)

        with self.connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [Post(**dict(row)) for row in rows]

    def get_posts_since(self, since_utc: float, limit: int | None = None) -> list[Post]:
        """Get all posts since a timestamp.

        Args:
            since_utc: Unix timestamp.
            limit: Maximum number of posts.

        Returns:
            List of posts.
        """
        query = "SELECT * FROM posts WHERE created_utc >= ? ORDER BY created_utc DESC"
        params: list = [since_utc]

        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)

        with self.connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [Post(**dict(row)) for row in rows]

    def post_exists(self, post_id: str) -> bool:
        """Check if a post exists in the database.

        Args:
            post_id: Reddit post ID.

        Returns:
            True if post exists.
        """
        with self.connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM posts WHERE post_id = ?", (post_id,)
            ).fetchone()
            return row is not None

    # -------------------------------------------------------------------------
    # Comment operations
    # -------------------------------------------------------------------------

    def insert_comment(self, comment: Comment) -> None:
        """Insert or update a comment.

        Args:
            comment: Comment data to insert.
        """
        with self.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO comments
                (comment_id, post_id, body, score, created_utc)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    comment.comment_id,
                    comment.post_id,
                    comment.body,
                    comment.score,
                    comment.created_utc,
                ),
            )

    def insert_comments(self, comments: list[Comment]) -> None:
        """Insert multiple comments.

        Args:
            comments: List of comments to insert.
        """
        with self.connection() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO comments
                (comment_id, post_id, body, score, created_utc)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (c.comment_id, c.post_id, c.body, c.score, c.created_utc)
                    for c in comments
                ],
            )

    def get_comments_for_post(
        self, post_id: str, limit: int | None = None
    ) -> list[Comment]:
        """Get comments for a post.

        Args:
            post_id: Reddit post ID.
            limit: Maximum number of comments.

        Returns:
            List of comments ordered by score.
        """
        query = "SELECT * FROM comments WHERE post_id = ? ORDER BY score DESC"
        params: list = [post_id]

        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)

        with self.connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [Comment(**dict(row)) for row in rows]

    # -------------------------------------------------------------------------
    # Run operations
    # -------------------------------------------------------------------------

    def insert_run(self, run: Run) -> None:
        """Insert a new run.

        Args:
            run: Run metadata to insert.
        """
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO runs
                (run_id, run_date, goal_template, goal_prompt_hash,
                 settings_json, subreddits_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.run_id,
                    run.run_date,
                    run.goal_template,
                    run.goal_prompt_hash,
                    run.settings_json,
                    run.subreddits_json,
                    run.created_at,
                ),
            )

    def get_run(self, run_id: str) -> Run | None:
        """Get a run by ID.

        Args:
            run_id: Run ID.

        Returns:
            Run if found, None otherwise.
        """
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row:
                return Run(**dict(row))
            return None

    def get_runs_by_date(self, run_date: str) -> list[Run]:
        """Get all runs for a specific date.

        Args:
            run_date: Date string (YYYY-MM-DD).

        Returns:
            List of runs.
        """
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM runs WHERE run_date = ? ORDER BY created_at DESC",
                (run_date,),
            ).fetchall()
            return [Run(**dict(row)) for row in rows]

    def get_recent_runs(self, limit: int = 10) -> list[Run]:
        """Get most recent runs.

        Args:
            limit: Maximum number of runs.

        Returns:
            List of runs.
        """
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
            return [Run(**dict(row)) for row in rows]

    # -------------------------------------------------------------------------
    # Run items operations
    # -------------------------------------------------------------------------

    def insert_run_items(self, items: list[RunItem]) -> None:
        """Insert posts associated with a run.

        Args:
            items: List of run items.
        """
        with self.connection() as conn:
            conn.executemany(
                """
                INSERT INTO run_items (run_id, post_id, rank_score)
                VALUES (?, ?, ?)
                """,
                [(i.run_id, i.post_id, i.rank_score) for i in items],
            )

    def get_run_items(self, run_id: str) -> list[RunItem]:
        """Get all posts for a run.

        Args:
            run_id: Run ID.

        Returns:
            List of run items.
        """
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM run_items WHERE run_id = ? ORDER BY rank_score DESC",
                (run_id,),
            ).fetchall()
            return [RunItem(**dict(row)) for row in rows]

    def get_posts_for_run(self, run_id: str) -> list[Post]:
        """Get full post data for a run.

        Args:
            run_id: Run ID.

        Returns:
            List of posts with their data.
        """
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT p.* FROM posts p
                JOIN run_items ri ON p.post_id = ri.post_id
                WHERE ri.run_id = ?
                ORDER BY ri.rank_score DESC
                """,
                (run_id,),
            ).fetchall()
            return [Post(**dict(row)) for row in rows]

    # -------------------------------------------------------------------------
    # Digest operations
    # -------------------------------------------------------------------------

    def insert_digest(self, digest: Digest) -> None:
        """Insert a digest.

        Args:
            digest: Digest to insert.
        """
        with self.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO digests
                (run_id, model_mode, models_json, markdown, card_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    digest.run_id,
                    digest.model_mode,
                    digest.models_json,
                    digest.markdown,
                    digest.card_json,
                    digest.created_at,
                ),
            )

    def get_digest(self, run_id: str) -> Digest | None:
        """Get digest for a run.

        Args:
            run_id: Run ID.

        Returns:
            Digest if found, None otherwise.
        """
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM digests WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row:
                return Digest(**dict(row))
            return None

    def get_recent_digests(self, limit: int = 10) -> list[Digest]:
        """Get most recent digests.

        Args:
            limit: Maximum number of digests.

        Returns:
            List of digests.
        """
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM digests ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
            return [Digest(**dict(row)) for row in rows]

    # -------------------------------------------------------------------------
    # LLM cache operations
    # -------------------------------------------------------------------------

    @staticmethod
    def compute_cache_key(
        goal_prompt: str,
        post_ids: list[str],
        comment_ids: list[str],
        model: str,
    ) -> str:
        """Compute a cache key for LLM response.

        Args:
            goal_prompt: The goal prompt used.
            post_ids: List of post IDs included.
            comment_ids: List of comment IDs included.
            model: Model identifier.

        Returns:
            SHA256 hash as cache key.
        """
        data = {
            "goal_prompt": goal_prompt,
            "post_ids": sorted(post_ids),
            "comment_ids": sorted(comment_ids),
            "model": model,
        }
        content = json.dumps(data, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def get_cached_response(self, cache_key: str) -> LLMCache | None:
        """Get cached LLM response.

        Args:
            cache_key: Cache key.

        Returns:
            Cached response if found, None otherwise.
        """
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM llm_cache WHERE cache_key = ?", (cache_key,)
            ).fetchone()
            if row:
                return LLMCache(**dict(row))
            return None

    def cache_response(
        self,
        cache_key: str,
        model: str,
        input_hash: str,
        output_markdown: str,
    ) -> None:
        """Cache an LLM response.

        Args:
            cache_key: Cache key.
            model: Model identifier.
            input_hash: Hash of input content.
            output_markdown: Generated markdown.
        """
        with self.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO llm_cache
                (cache_key, model, input_hash, output_markdown, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (cache_key, model, input_hash, output_markdown, datetime.now().timestamp()),
            )

    # -------------------------------------------------------------------------
    # Run cluster operations (for Radar tracking)
    # -------------------------------------------------------------------------

    def insert_run_clusters(self, clusters: list[RunCluster]) -> None:
        """Insert cluster metrics for a run.

        Args:
            clusters: List of RunCluster objects.
        """
        with self.connection() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO run_clusters
                (run_id, cluster_id, cluster_name, frequency_posts, engagement_score_sum,
                 engagement_comments_sum, avg_age_hours, intensity_score,
                 proxy_score, value_score, top_urls_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        c.run_id, c.cluster_id, c.cluster_name, c.frequency_posts,
                        c.engagement_score_sum, c.engagement_comments_sum,
                        c.avg_age_hours, c.intensity_score, c.proxy_score,
                        c.value_score, c.top_urls_json,
                    )
                    for c in clusters
                ],
            )

    def get_run_clusters(self, run_id: str) -> list[RunCluster]:
        """Get cluster metrics for a run.

        Args:
            run_id: Run ID.

        Returns:
            List of RunCluster objects.
        """
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM run_clusters WHERE run_id = ?
                ORDER BY engagement_score_sum DESC
                """,
                (run_id,),
            ).fetchall()
            # Map DB columns to Python fields, handling old schemas
            result = []
            for row in rows:
                d = dict(row)
                # Handle old column name
                if "recency_avg_hours" in d and "avg_age_hours" not in d:
                    d["avg_age_hours"] = d.pop("recency_avg_hours", 0)
                elif "recency_avg_hours" in d:
                    d.pop("recency_avg_hours", None)
                # Handle old data without cluster_id
                if "cluster_id" not in d or not d["cluster_id"]:
                    d["cluster_id"] = d.get("cluster_name", "unknown")
                # Handle old data without value_score
                if "value_score" not in d:
                    d["value_score"] = 0.0
                result.append(RunCluster(**d))
            return result

    def get_historical_clusters(
        self,
        days: int = 7,
        goal_template: str | None = None,
        goal_prompt_hash: str | None = None,
    ) -> list[dict]:
        """Get historical cluster data for Radar analysis.

        Args:
            days: Number of days to look back.
            goal_template: Filter by template type (e.g., "content_ideas", "custom").
            goal_prompt_hash: Filter by exact goal prompt hash (for custom runs).

        Returns:
            List of dicts with cluster data from recent runs.
        """
        cutoff = datetime.now().timestamp() - (days * 24 * 60 * 60)

        # Build query with optional filters
        query = """
            SELECT rc.*, r.created_at as run_created_at, r.goal_template, r.goal_prompt_hash
            FROM run_clusters rc
            JOIN runs r ON rc.run_id = r.run_id
            WHERE r.created_at >= ?
        """
        params: list = [cutoff]

        # Filter by template type to only compare similar analyses
        if goal_template:
            query += " AND r.goal_template = ?"
            params.append(goal_template)

        # For custom templates, also filter by goal_prompt_hash for exact match
        if goal_prompt_hash:
            query += " AND r.goal_prompt_hash = ?"
            params.append(goal_prompt_hash)

        query += " ORDER BY r.created_at DESC"

        with self.connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    def get_heatmap_data(self, days: int = 30, max_runs: int = 20) -> dict:
        """Get cluster data formatted for heatmap visualization.

        Args:
            days: Number of days to look back.
            max_runs: Maximum number of runs to include.

        Returns:
            Dict with structure for Chart.js heatmap:
            {
                "runs": [{"run_id": "...", "date": "2025-01-20", "label": "Jan 20"}],
                "clusters": ["cluster_id1", "cluster_id2", ...],
                "cluster_labels": {"cluster_id1": "Display Name", ...},
                "data": [{"x": run_idx, "y": cluster_idx, "v": value_score}, ...]
            }
        """
        cutoff = datetime.now().timestamp() - (days * 24 * 60 * 60)
        with self.connection() as conn:
            # Get runs in time range
            runs = conn.execute(
                """
                SELECT DISTINCT r.run_id, r.created_at, r.run_date
                FROM runs r
                JOIN run_clusters rc ON r.run_id = rc.run_id
                WHERE r.created_at >= ?
                ORDER BY r.created_at ASC
                LIMIT ?
                """,
                (cutoff, max_runs),
            ).fetchall()

            if not runs:
                return {"runs": [], "clusters": [], "cluster_labels": {}, "data": []}

            run_list = []
            run_id_to_idx = {}
            for idx, row in enumerate(runs):
                run_id = row["run_id"]
                created_at = row["created_at"]
                run_date = row["run_date"]
                # Format date for display
                dt = datetime.fromtimestamp(created_at)
                label = dt.strftime("%b %d %H:%M")
                run_list.append({
                    "run_id": run_id,
                    "date": run_date,
                    "label": label,
                })
                run_id_to_idx[run_id] = idx

            # Get all cluster data for these runs
            run_ids = [r["run_id"] for r in runs]
            placeholders = ",".join("?" * len(run_ids))
            rows = conn.execute(
                f"""
                SELECT rc.run_id, rc.cluster_id, rc.cluster_name,
                       rc.frequency_posts, rc.engagement_score_sum, rc.value_score
                FROM run_clusters rc
                WHERE rc.run_id IN ({placeholders})
                ORDER BY rc.engagement_score_sum DESC
                """,
                run_ids,
            ).fetchall()

            # Build cluster list (unique, ordered by total engagement)
            cluster_engagement = {}
            cluster_labels = {}
            for row in rows:
                cid = row["cluster_id"]
                cluster_labels[cid] = row["cluster_name"]
                cluster_engagement[cid] = cluster_engagement.get(cid, 0) + row["engagement_score_sum"]

            # Sort clusters by total engagement
            sorted_clusters = sorted(cluster_engagement.keys(),
                                     key=lambda c: cluster_engagement[c], reverse=True)
            cluster_to_idx = {cid: idx for idx, cid in enumerate(sorted_clusters)}

            # Build data points for heatmap
            data = []
            for row in rows:
                run_idx = run_id_to_idx.get(row["run_id"])
                cluster_idx = cluster_to_idx.get(row["cluster_id"])
                if run_idx is not None and cluster_idx is not None:
                    # Use value_score if available, else compute from engagement
                    value = row["value_score"] if row["value_score"] else 0
                    # If no LLM score, use normalized engagement as fallback
                    if value == 0:
                        eng = row["engagement_score_sum"] + row["frequency_posts"] * 100
                        value = min(10, eng / 500)  # Normalize to ~0-10 scale
                    data.append({
                        "x": run_idx,
                        "y": cluster_idx,
                        "v": round(value, 2),
                        "freq": row["frequency_posts"],
                        "engagement": row["engagement_score_sum"],
                    })

            return {
                "runs": run_list,
                "clusters": sorted_clusters,
                "cluster_labels": cluster_labels,
                "data": data,
            }

    # -------------------------------------------------------------------------
    # Subreddit preset operations
    # -------------------------------------------------------------------------

    def insert_preset(self, preset: SubredditPreset) -> None:
        """Insert or update a subreddit preset."""
        with self.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO subreddit_presets
                (preset_id, name, description, subreddits_json, template, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    preset.preset_id, preset.name, preset.description,
                    preset.subreddits_json, preset.template,
                    preset.created_at, preset.updated_at,
                ),
            )

    def get_preset(self, preset_id: str) -> SubredditPreset | None:
        """Get a preset by ID."""
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM subreddit_presets WHERE preset_id = ?", (preset_id,)
            ).fetchone()
            if row:
                return SubredditPreset(**dict(row))
            return None

    def get_all_presets(self) -> list[SubredditPreset]:
        """Get all subreddit presets."""
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM subreddit_presets ORDER BY name"
            ).fetchall()
            return [SubredditPreset(**dict(row)) for row in rows]

    def delete_preset(self, preset_id: str) -> bool:
        """Delete a preset. Returns True if deleted."""
        with self.connection() as conn:
            cursor = conn.execute(
                "DELETE FROM subreddit_presets WHERE preset_id = ?", (preset_id,)
            )
            return cursor.rowcount > 0

    # -------------------------------------------------------------------------
    # Schedule operations
    # -------------------------------------------------------------------------

    def insert_schedule(self, schedule: Schedule) -> None:
        """Insert or update a schedule."""
        with self.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO schedules
                (schedule_id, name, preset_id, template, subreddits_json,
                 posts_per_sub, time_window_hours, cron_expression, enabled,
                 last_run_at, next_run_at, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    schedule.schedule_id, schedule.name, schedule.preset_id,
                    schedule.template, schedule.subreddits_json,
                    schedule.posts_per_sub, schedule.time_window_hours,
                    schedule.cron_expression, 1 if schedule.enabled else 0,
                    schedule.last_run_at, schedule.next_run_at, schedule.created_at,
                ),
            )

    def get_schedule(self, schedule_id: str) -> Schedule | None:
        """Get a schedule by ID."""
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM schedules WHERE schedule_id = ?", (schedule_id,)
            ).fetchone()
            if row:
                d = dict(row)
                d['enabled'] = bool(d['enabled'])
                return Schedule(**d)
            return None

    def get_all_schedules(self) -> list[Schedule]:
        """Get all schedules."""
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM schedules ORDER BY name"
            ).fetchall()
            result = []
            for row in rows:
                d = dict(row)
                d['enabled'] = bool(d['enabled'])
                result.append(Schedule(**d))
            return result

    def get_enabled_schedules(self) -> list[Schedule]:
        """Get all enabled schedules."""
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM schedules WHERE enabled = 1 ORDER BY next_run_at"
            ).fetchall()
            result = []
            for row in rows:
                d = dict(row)
                d['enabled'] = bool(d['enabled'])
                result.append(Schedule(**d))
            return result

    def get_due_schedules(self) -> list[Schedule]:
        """Get schedules that are due to run now."""
        now = datetime.now().timestamp()
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM schedules
                WHERE enabled = 1 AND next_run_at IS NOT NULL AND next_run_at <= ?
                ORDER BY next_run_at
                """,
                (now,),
            ).fetchall()
            result = []
            for row in rows:
                d = dict(row)
                d['enabled'] = bool(d['enabled'])
                result.append(Schedule(**d))
            return result

    def update_schedule_run_times(
        self, schedule_id: str, last_run_at: float, next_run_at: float
    ) -> None:
        """Update schedule's last and next run times."""
        with self.connection() as conn:
            conn.execute(
                """
                UPDATE schedules
                SET last_run_at = ?, next_run_at = ?
                WHERE schedule_id = ?
                """,
                (last_run_at, next_run_at, schedule_id),
            )

    def delete_schedule(self, schedule_id: str) -> bool:
        """Delete a schedule. Returns True if deleted."""
        with self.connection() as conn:
            cursor = conn.execute(
                "DELETE FROM schedules WHERE schedule_id = ?", (schedule_id,)
            )
            return cursor.rowcount > 0

    # -------------------------------------------------------------------------
    # Notification operations
    # -------------------------------------------------------------------------

    def insert_notification(self, notification: Notification) -> None:
        """Insert or update a notification config."""
        with self.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO notifications
                (notification_id, name, notification_type, config_json, trigger_on, enabled, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    notification.notification_id, notification.name,
                    notification.notification_type, notification.config_json,
                    notification.trigger_on, 1 if notification.enabled else 0,
                    notification.created_at,
                ),
            )

    def get_notification(self, notification_id: str) -> Notification | None:
        """Get a notification config by ID."""
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM notifications WHERE notification_id = ?", (notification_id,)
            ).fetchone()
            if row:
                d = dict(row)
                d['enabled'] = bool(d['enabled'])
                return Notification(**d)
            return None

    def get_all_notifications(self) -> list[Notification]:
        """Get all notification configs."""
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM notifications ORDER BY name"
            ).fetchall()
            result = []
            for row in rows:
                d = dict(row)
                d['enabled'] = bool(d['enabled'])
                result.append(Notification(**d))
            return result

    def get_enabled_notifications(self, trigger_on: str) -> list[Notification]:
        """Get enabled notifications for a specific trigger."""
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM notifications WHERE enabled = 1 AND trigger_on = ?",
                (trigger_on,),
            ).fetchall()
            result = []
            for row in rows:
                d = dict(row)
                d['enabled'] = bool(d['enabled'])
                result.append(Notification(**d))
            return result

    def delete_notification(self, notification_id: str) -> bool:
        """Delete a notification. Returns True if deleted."""
        with self.connection() as conn:
            cursor = conn.execute(
                "DELETE FROM notifications WHERE notification_id = ?", (notification_id,)
            )
            return cursor.rowcount > 0

    # -------------------------------------------------------------------------
    # Utility operations
    # -------------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Get database statistics.

        Returns:
            Dictionary with counts for each table.
        """
        stats = {}
        tables = [
            "posts", "comments", "runs", "run_items", "digests", "llm_cache",
            "run_clusters", "subreddit_presets", "schedules", "notifications"
        ]

        with self.connection() as conn:
            for table in tables:
                try:
                    row = conn.execute(f"SELECT COUNT(*) as count FROM {table}").fetchone()
                    stats[table] = row["count"]
                except Exception:
                    stats[table] = 0  # Table may not exist yet

        return stats

    def vacuum(self) -> None:
        """Optimize database by running VACUUM."""
        with self.connection() as conn:
            conn.execute("VACUUM")

    def clear_old_cache(self, days: int = 30) -> int:
        """Clear LLM cache entries older than N days.

        Args:
            days: Number of days to keep.

        Returns:
            Number of entries deleted.
        """
        cutoff = datetime.now().timestamp() - (days * 24 * 60 * 60)
        with self.connection() as conn:
            cursor = conn.execute(
                "DELETE FROM llm_cache WHERE created_at < ?", (cutoff,)
            )
            return cursor.rowcount


def get_database(db_path: str | Path | None = None) -> Database:
    """Get a database instance.

    Args:
        db_path: Optional path to database file.

    Returns:
        Database instance.
    """
    if db_path is None:
        from src.config import get_config
        config = get_config()
        db_path = config.database.path

    return Database(db_path)
