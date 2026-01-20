# Reddit Radar

> Turn Reddit into your daily, goal-specific research radar.

Reddit Radar is a local-first application that helps you extract actionable insights from Reddit. Pick a goal (Startup Ideas, Trend Radar, etc.), let AI discover relevant subreddits, and get a clean daily digest with the most important discussions.

## Features

- **Goal Templates**: Pre-built analysis modes (Startup Ideas, Trend Radar, Industry Intel, Career Intel, Deep Research)
- **Smart Subreddit Discovery**: LLM suggests relevant subreddits based on your goal
- **Local Ranking**: Intelligent post ranking without LLM costs
- **AI-Powered Summaries**: Get key themes, top insights, and actionable next steps
- **Caching**: SQLite storage avoids reprocessing the same content
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
goal_template: "startup_ideas"
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
python main.py discover --goal "startup ideas"

# Run daily analysis
python main.py run --template startup_ideas --subreddits "startups,SaaS"

# Run with LLM caching disabled
python main.py run --template startup_ideas --no-cache

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
| **Startup Ideas** | Pain points, top 3 ideas, risks, next steps |
| **Trend Radar** | Emerging trends, signals, predictions |
| **Industry Intel** | Key debates, changes, actionable takeaways |
| **Career Intel** | Hot skills, what to learn, what to build |
| **Deep Research** | Evidence summary, uncertainties, sources |

## How It Works

1. **Discover**: LLM suggests subreddits based on your goal
2. **Fetch**: Collect top posts and comments from selected subreddits
3. **Rank**: Score posts locally (engagement, recency, keyword relevance)
4. **Summarize**: LLM analyzes shortlisted content against your goal
5. **Output**: Markdown report + card view in web UI

## Project Structure

```
reddit-radar/
├── src/
│   ├── config.py          # Configuration management
│   ├── database.py        # SQLite operations
│   ├── reddit_client.py   # Reddit API wrapper
│   ├── ranker.py          # Post ranking algorithm
│   ├── llm_client.py      # LLM API wrapper (DeepSeek/OpenAI)
│   ├── templates.py       # Goal templates & prompt builder
│   └── ui/
│       ├── web.py         # FastAPI web server
│       └── templates/     # Jinja2 HTML templates
├── tests/                 # Test suite
├── config/                # YAML configuration
├── output/                # Generated reports
└── main.py                # CLI entry point
```

## Cost Control

Reddit Radar is designed to minimize API costs:

- **Local ranking**: No LLM calls for filtering/sorting
- **SQLite caching**: Never reprocess the same content
- **Configurable shortlist**: Control how much content goes to LLM
- **Response caching**: Identical queries return cached results

## Development

```bash
# Run tests
pytest

# Run with verbose output
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
- [ ] Multi-model debate mode
- [ ] Export to Notion/Obsidian

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit PRs.

## License

MIT License - see [LICENSE](LICENSE) for details.

---

Built with Python, PRAW, FastAPI, and DeepSeek.
