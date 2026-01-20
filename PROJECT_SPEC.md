# Reddit Radar - Project Specification

> A local app that turns Reddit into a daily, goal-specific research radar.

## Overview

Reddit Radar helps users extract actionable insights from Reddit by:
1. Discovering relevant subreddits via LLM
2. Fetching and ranking posts/comments
3. Generating goal-specific summaries using AI
4. Presenting results in clean Markdown + card UI

## Core User Flow

```
1. Choose goal template (Startup Ideas, Trend Radar, etc.)
2. [Optional] Refine via Prompt Builder
3. LLM suggests subreddits → User reviews/edits list
4. Configure: posts per sub, time window, shortlist size
5. Run → Get Markdown report + Card view
```

---

## Features

### 1. Goal Templates (Presets)

| Template | Purpose |
|----------|---------|
| **Startup Ideas** | Extract pain points, propose top 3 ideas, risks, next steps |
| **Trend Radar** | List emerging trends, signals, predicted direction, why it matters |
| **Industry Intel** | What's changing, key debates, who cares, actionable takeaways |
| **Career/Job Intel** | Hot skills/tools, what to learn, what to build |
| **Deep Research** | Summarize evidence, highlight uncertainties, link to sources |

### 2. Prompt Builder (Optional)

Guided wizard that refines the goal template:

| Option | Values |
|--------|--------|
| Output format | bullets / brief / structured report |
| Include links | yes / no |
| Output length | short / medium / long |
| Creativity | low / medium / high |
| Factuality | strict / balanced / exploratory |
| Tone | neutral / direct / playful |
| Extra instructions | free text |

### 3. Subreddit Discovery (LLM-Powered)

- User selects goal + number of subreddits (3 / 5 / 10)
- LLM searches/proposes relevant subreddits
- User can:
  - Remove suggested subreddits
  - Add custom subreddits
- Final list is saved for the run

### 4. Data Collection Settings

| Setting | Options | Default |
|---------|---------|---------|
| Subreddits | 3 / 5 / 10 | 5 |
| Posts per subreddit | 5 / 10 / 20 | 10 |
| Comments per post | Always 5 | 5 |
| Time window | 24h / 72h / 144h | 24h |
| Post feed mode | top / rising | top |

### 5. Ranking & Shortlisting (No LLM)

Before LLM processing, posts are ranked locally:

```python
score = (
    post_score * weight_score +
    num_comments * weight_comments +
    recency_decay(created_utc) +
    keyword_overlap(title + body, goal_prompt) * weight_keywords
)
```

- Global shortlist: 20-30 posts (configurable)
- Top 5 comments attached per shortlisted post

### 6. LLM Summary Output

#### Single Model Mode (v1 Default)

Input: goal template + prompt builder output + shortlisted posts/comments

Output sections:
1. **Key Themes** (5-10 bullets)
2. **Top Insights/Ideas** (top 3 aligned to goal)
3. **Evidence & Links** (post links grouped by insight)
4. **Reasoning/Justification** (why selected, what signals)
5. **Next Actions** (3-5 actionable items)

#### Multi-Model Mode (Future)

- User selects 3 models (e.g., GPT + Claude + Gemini)
- Each produces draft digest
- Synthesis step merges into final output
- Debate transcript optional

### 7. Output Formats

- **Markdown file**: `/output/YYYY-MM-DD.md`
- **Card view**: Web UI with tiles showing:
  - Date + Goal
  - Top 3 insights
  - Links to full report and sources

---

## Technical Architecture

### Directory Structure

```
reddit-radar/
├── src/
│   ├── __init__.py
│   ├── config.py          # Configuration management
│   ├── database.py        # SQLite operations
│   ├── reddit_client.py   # Reddit API wrapper
│   ├── ranker.py          # Post ranking/shortlisting
│   ├── llm_client.py      # LLM API wrapper
│   ├── summarizer.py      # Summary generation
│   ├── templates.py       # Goal templates
│   └── ui/
│       ├── __init__.py
│       ├── web.py         # FastAPI web UI
│       └── static/        # CSS, JS assets
├── tests/
│   ├── __init__.py
│   ├── test_database.py
│   ├── test_reddit_client.py
│   ├── test_ranker.py
│   └── test_llm_client.py
├── output/                # Generated reports
├── config/
│   └── default.yaml       # Default configuration
├── .env.example           # Environment variables template
├── .gitignore
├── requirements.txt
├── README.md
├── LICENSE
├── PROJECT_SPEC.md        # This file
└── main.py                # CLI entry point
```

### SQLite Schema

```sql
-- Posts fetched from Reddit
CREATE TABLE posts (
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
CREATE TABLE comments (
    comment_id TEXT PRIMARY KEY,
    post_id TEXT NOT NULL,
    body TEXT NOT NULL,
    score INTEGER DEFAULT 0,
    created_utc REAL NOT NULL,
    FOREIGN KEY (post_id) REFERENCES posts(post_id)
);

-- Analysis runs
CREATE TABLE runs (
    run_id TEXT PRIMARY KEY,
    run_date TEXT NOT NULL,
    goal_template TEXT NOT NULL,
    goal_prompt_hash TEXT NOT NULL,
    settings_json TEXT NOT NULL,
    subreddits_json TEXT NOT NULL,
    created_at REAL NOT NULL
);

-- Posts used in each run
CREATE TABLE run_items (
    run_id TEXT NOT NULL,
    post_id TEXT NOT NULL,
    rank_score REAL,
    PRIMARY KEY (run_id, post_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id),
    FOREIGN KEY (post_id) REFERENCES posts(post_id)
);

-- Generated digests
CREATE TABLE digests (
    run_id TEXT PRIMARY KEY,
    model_mode TEXT NOT NULL,
    models_json TEXT NOT NULL,
    markdown TEXT NOT NULL,
    card_json TEXT,
    created_at REAL NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

-- LLM response cache
CREATE TABLE llm_cache (
    cache_key TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    input_hash TEXT NOT NULL,
    output_markdown TEXT NOT NULL,
    created_at REAL NOT NULL
);

-- Indexes for performance
CREATE INDEX idx_posts_subreddit ON posts(subreddit);
CREATE INDEX idx_posts_created_utc ON posts(created_utc);
CREATE INDEX idx_comments_post_id ON comments(post_id);
CREATE INDEX idx_runs_run_date ON runs(run_date);
```

### Configuration

#### config/default.yaml

```yaml
# Goal settings
goal_template: "startup_ideas"
prompt_builder:
  output_format: "structured"  # bullets / brief / structured
  include_links: true
  output_length: "medium"      # short / medium / long
  creativity: "medium"         # low / medium / high
  factuality: "balanced"       # strict / balanced / exploratory
  tone: "neutral"              # neutral / direct / playful
  extra_instructions: ""

# Data collection
collection:
  subreddit_count: 5           # 3 / 5 / 10
  posts_per_subreddit: 10      # 5 / 10 / 20
  comments_per_post: 5         # Always 5
  time_window_hours: 24        # 24 / 72 / 144
  feed_mode: "top"             # top / rising

# Ranking
ranking:
  shortlist_size: 25
  weight_score: 1.0
  weight_comments: 1.5
  weight_recency: 0.5
  weight_keywords: 2.0

# LLM settings
llm:
  mode: "single"               # single / multi
  provider: "deepseek"
  model: "deepseek-chat"
  temperature: 0.7
  max_tokens: 4000

# Output
output:
  directory: "./output"
  date_format: "%Y-%m-%d"
```

#### Environment Variables (.env)

```bash
# Reddit API (Application-only OAuth)
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=RedditRadar/1.0

# LLM API
DEEPSEEK_API_KEY=your_api_key

# Optional: Multi-model mode
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GOOGLE_API_KEY=
```

### API Integrations

#### Reddit API (PRAW - Application-Only)

```python
import praw

reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT
)
# No username/password needed for read-only access
```

#### DeepSeek API (OpenAI-Compatible)

```python
from openai import OpenAI

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[...],
    temperature=0.7
)
```

---

## CLI Commands

| Command | Description |
|---------|-------------|
| `reddit-radar init` | Create config + database |
| `reddit-radar run` | Run daily analysis (default) |
| `reddit-radar discover` | LLM subreddit discovery only |
| `reddit-radar backfill --days N` | Backfill last N days |
| `reddit-radar history --days N` | Show run history |
| `reddit-radar trends --days N` | Show topic trends |
| `reddit-radar serve` | Launch web UI |

---

## Development Phases

### Phase 1: Foundation
- [x] Project specification (this file)
- [x] Project structure (GitHub-ready)
- [x] Requirements and dependencies
- [x] Configuration system
- [x] SQLite database layer
- [x] Tests for database layer

### Phase 2: Data Collection
- [x] Reddit API client
- [x] Post/comment fetching
- [x] Subreddit discovery (LLM)
- [x] Tests for Reddit client

### Phase 3: Processing Pipeline
- [x] Post ranking algorithm
- [x] Shortlisting logic
- [x] LLM client (DeepSeek)
- [x] Summary generation
- [x] Tests for pipeline

### Phase 4: Output & UI
- [x] Markdown report generation
- [x] Web UI (FastAPI)
- [x] Card view component
- [x] CLI commands
- [x] Run from browser

### Phase 5: Polish
- [x] LLM response caching
- [x] Backfill command
- [x] Trends command
- [ ] Multi-model mode (future-ready)
- [x] Documentation
- [ ] Example outputs

---

## Non-Goals (v1)

- No production SaaS deployment
- No heavy scraping outside official API
- No user accounts/authentication
- No mobile app
- No real-time updates

---

## Cost Control

- Local ranking reduces LLM calls
- SQLite caching avoids reprocessing
- Configurable shortlist size
- Multi-model mode has smaller shortlist by default

---

## Implementation Status

### COMPLETED Features

| Feature | Status | Notes |
|---------|--------|-------|
| **Project Structure** | ✅ Done | GitHub-ready with README, LICENSE, .gitignore |
| **Configuration System** | ✅ Done | YAML config + env vars, dataclass-based |
| **SQLite Database** | ✅ Done | Full schema with 6 tables, indexes, CRUD operations |
| **Reddit API Client** | ✅ Done | PRAW-based, application-only OAuth, verified working |
| **LLM Client (DeepSeek)** | ✅ Done | OpenAI-compatible, tested with real API |
| **Subreddit Discovery** | ✅ Done | LLM suggests + Reddit API verifies subreddits |
| **Post Ranking** | ✅ Done | Engagement + recency + keyword scoring |
| **Shortlisting** | ✅ Done | Configurable size, attaches comments |
| **Goal Templates (5)** | ✅ Done | startup_ideas, trend_radar, industry_intel, career_intel, deep_research |
| **Prompt Builder** | ✅ Done | Format, length, creativity, factuality, tone modifiers |
| **Markdown Reports** | ✅ Done | Full reports with headers, links, metadata |
| **CLI Commands** | ✅ Done | init, run, discover, templates, history, stats, check |
| **Test Suite** | ✅ Done | 28 tests passing (pytest) |

### COMPLETED (Session 2 - Jan 19, 2026)

| Feature | Status | Notes |
|---------|--------|-------|
| **LLM Response Caching** | ✅ Done | Uses `llm_cache` table, `--no-cache` flag to bypass |
| **Backfill Command** | ✅ Done | Re-analyze historical posts with different templates |
| **Trends Command** | ✅ Done | Keyword frequency analysis over time |
| **Clear Cache Command** | ✅ Done | Clear old LLM cache entries |
| **Run from Browser** | ✅ Done | Background job execution with progress tracking |
| **Test Warning Fixes** | ✅ Done | Fixed all pytest warnings |
| **CLI Tests** | ✅ Done | 11 new tests for CLI commands |

### COMPLETED (Session 1 - Jan 15, 2026)

| Feature | Status | Notes |
|---------|--------|-------|
| **Web UI - FastAPI Backend** | ✅ Done | `src/ui/web.py` with all routes |
| **Web UI - Base Template** | ✅ Done | Beautiful minimalist CSS, Inter font, cards/tiles |
| **Web UI - Dashboard** | ✅ Done | Shows recent runs with insight previews |
| **Web UI - Run Detail** | ✅ Done | Full markdown report with sidebar |
| **Web UI - History Page** | ✅ Done | List of all past runs |
| **Web UI - New Run Form** | ✅ Done | Template selection, settings (CLI note for now) |
| **Serve Command** | ✅ Done | `python main.py serve` launches web UI |

### TODO (Remaining)

| Feature | Priority | Notes |
|---------|----------|-------|
| **Multi-Model Mode** | Low | 3-model debate/synthesis (future) |
| **Export Formats** | Low | JSON, Notion, Obsidian export |
| **Example Outputs** | Low | Sample reports for documentation |

### Verified Working (Live Tests)

```
✓ Reddit API connection (application-only OAuth)
✓ Subreddit discovery for multiple topics (finance, coding, smart home, remote work)
✓ Post/comment fetching from r/startups, r/SaaS
✓ Local ranking algorithm (engagement, recency, keywords)
✓ DeepSeek LLM analysis (9,806 tokens used in test run)
✓ Markdown report generation with source links
✓ Database storage and retrieval
✓ CLI commands (all 7 working)
✓ 28/28 pytest tests passing
```

### Files Created

```
main.py                    # CLI entry point (500+ lines)
src/config.py              # Configuration management
src/database.py            # SQLite layer (600+ lines)
src/reddit_client.py       # Reddit API client
src/llm_client.py          # LLM client (DeepSeek)
src/templates.py           # 5 goal templates + prompt builder
src/ranker.py              # Post ranking algorithm
tests/test_database.py     # 22 database tests
tests/test_reddit_client.py # 6 Reddit client tests
config/default.yaml        # Default configuration
.env.example               # Environment template
requirements.txt           # Python dependencies
pyproject.toml             # Modern Python packaging
README.md                  # User documentation
LICENSE                    # MIT License
PROJECT_SPEC.md            # This specification
```

---

## Session Notes

### Session 1 Summary (Jan 15, 2026)

**What was accomplished:**
1. Full project structure created (GitHub-ready)
2. All core modules implemented and tested:
   - Config system (YAML + env vars)
   - SQLite database with 6 tables
   - Reddit API client (PRAW, application-only OAuth)
   - LLM client (DeepSeek via OpenAI-compatible API)
   - Post ranker (engagement + recency + keywords)
   - 5 goal templates with prompt builder
3. CLI fully working: `init`, `run`, `discover`, `templates`, `history`, `stats`, `check`, `serve`
4. Live tested: fetched real Reddit posts, generated real LLM analysis
5. Web UI created with minimalist professional design (not yet browser-tested)

**To resume tomorrow:**
```bash
cd /home/skaczmarczyk/code/RedditTrends

# Activate environment (pip is at ~/.local/bin)
export PATH="$HOME/.local/bin:$PATH"

# Test the web UI
python main.py serve

# Open browser to http://127.0.0.1:8000
```

**Next steps:**
1. Verify web UI renders correctly in browser
2. Test all pages: Dashboard, History, Run Detail, New Run
3. Optionally: Add ability to run analysis from web form
4. Optionally: Add LLM response caching

---

## License

MIT License - Open source from day one.
