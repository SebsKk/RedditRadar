# Reddit Radar

> Turn Reddit into your daily, goal-specific research radar.

Reddit Radar is a local-first application that helps you extract actionable insights from Reddit. Pick a goal (Content Ideas, Trend Radar, etc.), let AI discover relevant subreddits, and get a clean daily digest with the most important discussions.

## Features

### Core Analysis
- **Goal Templates**: Pre-built analysis modes (Content Ideas, Trend Radar, Industry Intel, Career Intel, Deep Research)
- **Smart Subreddit Discovery**: LLM suggests relevant subreddits based on your goal
- **Local Ranking**: Intelligent post ranking without LLM costs
- **AI-Powered Summaries**: Get key themes, top insights, and actionable next steps
- **Structured Analysis**: TF-IDF clustering, scorecards, and radar tracking

### Intelligent Insights
- **"So What?" Summary**: LLM-generated 1-liner actionable insight at the top of every report
- **Smart Cluster Filtering**: "General" bucket only appears in Top 3 when truly dominant (>30% posts AND >1.5x engagement)
- **Clean Key Signals**: TF-IDF extraction with document frequency filtering (requires 2+ posts per term)
- **Topic-Aware Radar**: Only compares runs with the same goal/template (gaming analysis won't pollute startup insights)

### Trend Tracking
- **Radar Categories**: New, Continuing, Repeating, and Rising clusters
- **Percentage-Based Rising Detection**: +2 posts OR +10% engagement/comments (not noise from tiny absolute changes)
- **Interactive Heatmap**: Chart.js visualization of cluster value over time with LLM-assigned scores (0-10)
- **Historical Analysis**: Track recurring patterns and trends across runs via web UI

### Automation & Management
- **Subreddit Presets**: Save and reuse your favorite subreddit combinations
- **Scheduled Runs**: Cron-like scheduling for automated daily/weekly analyses
- **Notifications**: Get webhook or email alerts when analyses complete
- **Subreddit Diversity**: Ensures all searched subreddits are represented in analysis (min 2 posts each)

### Performance & Cost Control
- **SQLite Caching**: Never reprocess the same content
- **LLM Response Caching**: Identical queries return cached results
- **Configurable Shortlist**: Control how much content goes to LLM
- **Clean Output**: Markdown reports + web-based card view

## Quick Start

### Prerequisites

- Python 3.10+
- Reddit API credentials ([get them here](https://www.reddit.com/prefs/apps))
- DeepSeek API key ([get it here](https://platform.deepseek.com/))

### Installation

```bash
# Clone the repository
git clone https://github.com/skaczmarczyk/reddit-radar.git
cd reddit-radar

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment variables
cp .env.example .env
# Edit .env with your API credentials
```

### Configuration

Edit `config/default.yaml` to customize:

```yaml
goal:
  template: "content_ideas"  # or: trend_radar, industry_intel, career_intel, deep_research
collection:
  subreddit_count: 5
  posts_per_subreddit: 10
  time_window_hours: 24
```

### Usage

```bash
# Initialize database
python main.py init

# Discover subreddits for your goal
python main.py discover --goal "content marketing strategies"

# Run daily analysis
python main.py run --template content_ideas --subreddits "content_marketing,blogging"

# Run with LLM caching disabled
python main.py run --template content_ideas --no-cache

# Backfill analysis using cached posts
python main.py backfill --template trend_radar --days 7

# Show keyword trends
python main.py trends --days 7 --top 20

# Clear old LLM cache
python main.py clear-cache --days 30

# Launch web UI (includes run-from-browser)
python main.py serve

# View history
python main.py history --days 7

# Check configuration
python main.py check
```

## Goal Templates

| Template | What You Get |
|----------|--------------|
| **Content Ideas** | Hot topics, content formats, engagement patterns, action plan |
| **Trend Radar** | Emerging trends, signals, predictions |
| **Industry Intel** | Key debates, changes, actionable takeaways |
| **Career Intel** | Hot skills, what to learn, what to build |
| **Deep Research** | Evidence summary, uncertainties, sources |

## How It Works

1. **Discover**: LLM suggests subreddits based on your goal
2. **Fetch**: Collect top posts and comments from selected subreddits
3. **Rank**: Score posts locally (engagement, recency, keyword relevance) with subreddit diversity
4. **Cluster**: TF-IDF clustering groups posts into themes with smart signal extraction
5. **Score**: Generate scorecards with signal strength, WTP, complexity, reachability, moat
6. **Radar**: Track New/Continuing/Repeating/Rising patterns across runs (topic-aware)
7. **Summarize**: LLM generates "So What?" summary + detailed analysis
8. **Output**: Markdown report + card view in web UI + interactive heatmap

## Report Structure

Each analysis produces a structured report:

```
## So What?
*This week's strongest theme: [theme]. Quick validation: [specific action].*

## Radar
- New (vs previous run): clusters appearing for first time
- Continuing: clusters active in both current and previous run
- Repeating: clusters seen in 2+ of last 3 runs
- Rising: clusters with +2 posts OR +10% engagement/comments

## Cluster Summary
Topic clusters with post counts, engagement, and intensity

## Top 3 Items
Scorecards with signal strength, WTP, complexity, reachability, moat

## Evidence & Links
Source URLs grouped by cluster

## Next Actions
Prioritized recommendations based on scorecard dimensions
```

## Project Structure

```
reddit-radar/
├── src/
│   ├── config.py          # Configuration management
│   ├── database.py        # SQLite operations + migrations
│   ├── reddit_client.py   # Reddit API wrapper (PRAW)
│   ├── ranker.py          # Post ranking with subreddit diversity
│   ├── llm_client.py      # LLM API wrapper (DeepSeek/OpenAI)
│   ├── templates.py       # Goal templates & prompt builder
│   ├── discovery.py       # Subreddit discovery
│   ├── scheduler.py       # Cron-like scheduled runs
│   ├── analysis/          # Structured analysis module
│   │   ├── __init__.py    # Module exports
│   │   ├── clustering.py  # TF-IDF clustering + signal extraction
│   │   ├── scoring.py     # Template-specific scoring + LLM cluster scoring
│   │   ├── radar.py       # New/Repeating/Rising tracking (percentage-based)
│   │   ├── report.py      # Report generation + "So What?" summary
│   │   └── historical.py  # Historical pattern analysis
│   └── ui/
│       ├── __init__.py    # Module exports
│       ├── web.py         # FastAPI web server + API endpoints
│       └── templates/     # Jinja2 HTML templates
│           ├── base.html       # Base layout
│           ├── dashboard.html  # Main dashboard
│           ├── historical.html # Historical trends + heatmap
│           ├── history.html    # Run history list
│           ├── job_status.html # Job progress tracking
│           ├── new_run.html    # Create new analysis
│           ├── run_detail.html # Single run view
│           └── settings.html   # Presets, schedules, notifications
├── tests/                 # Test suite (39 tests)
│   ├── test_cli.py        # CLI command tests
│   ├── test_database.py   # Database operation tests
│   └── test_reddit_client.py # Reddit client tests
├── config/
│   └── default.yaml       # Default configuration
├── output/                # Generated reports
└── main.py                # CLI entry point
```

## API Endpoints

The web UI exposes these endpoints:

### Pages
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Dashboard with recent runs |
| `/history` | GET | List all past runs |
| `/historical` | GET | Historical trends analysis + heatmap |
| `/settings` | GET | Settings (presets, schedules, notifications) |
| `/run/{run_id}` | GET | View a specific run's report |
| `/new` | GET/POST | Create and run new analysis |

### Analysis API
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/discover` | POST | Discover subreddits for a goal |
| `/api/job/{job_id}` | GET | Check analysis job status |
| `/api/historical-analysis` | POST | LLM-powered historical analysis |
| `/api/historical-summary` | GET | Fast historical summary (no LLM) |
| `/api/heatmap` | GET | Heatmap data for cluster value visualization |

### Management API
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/presets` | GET/POST/DELETE | Manage subreddit presets |
| `/api/schedules` | GET/POST/DELETE | Manage scheduled runs |
| `/api/notifications` | GET/POST/DELETE | Manage notifications |

## Cost Control

Reddit Radar is designed to minimize API costs:

- **Local ranking**: No LLM calls for filtering/sorting
- **SQLite caching**: Never reprocess the same content
- **Configurable shortlist**: Control how much content goes to LLM (default: 25 posts)
- **Response caching**: Identical queries return cached results
- **Smart LLM calls**: "So What?" summary uses minimal tokens (~150 max)

## Development

```bash
# Run tests
pytest

# Run tests with verbose output
pytest -v

# Run with verbose logging
python main.py run --verbose

# Development mode (auto-reload)
python main.py serve --reload
```

## Roadmap

- [x] Core pipeline (fetch → rank → summarize)
- [x] SQLite caching
- [x] Web UI with card view
- [x] Run analysis from web browser
- [x] LLM response caching
- [x] Backfill command (re-analyze cached data)
- [x] Trend analysis (keyword frequency over time)
- [x] Structured analysis (TF-IDF clustering, scorecards)
- [x] Historical pattern tracking
- [x] Historical analysis UI in web interface
- [x] Subreddit presets (save/reuse combinations)
- [x] Scheduled runs (cron-like automation)
- [x] Webhook/email notifications
- [x] Interactive heatmap (Chart.js + LLM value scoring)
- [x] "So What?" actionable summary
- [x] Topic-aware Radar (only compare same-goal runs)
- [x] Smart cluster filtering (General bucket handling)
- [x] Percentage-based Rising detection
- [ ] Multi-model debate mode
- [ ] Export to Notion/Obsidian/JSON

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit PRs.

## License

MIT License - see [LICENSE](LICENSE) for details.

---

Built with Python, PRAW, FastAPI, Chart.js, and DeepSeek.
