# Session Progress - January 19, 2025

## What Was Done

### 1. Structured Analysis Module (`src/analysis/`)

Created a complete analysis system with:

#### Clustering (`clustering.py`)
- **TF-IDF cosine similarity** for cluster assignment (not substring matching)
- **10 core clusters** + `uncategorized` fallback:
  - Execution/Focus, Distribution/Traction, Ops/Systems, Web Presence/Brand
  - Lead Gen/Sales, Pricing/Monetization, Tools/Tech, Career/Skills
  - Research/Uncertainty, Info Management, Uncategorized
- **Multi-label support** - posts can belong to secondary clusters
- **Soft saturation** for intensity/proxy scores (exponential decay, not linear)
- **Precomputed** cluster token sets and TF-IDF vectors for performance
- **Fixed keyword_overlap** - now divides by `min(top_n, goal_tokens, 20)` instead of full goal length

#### Scoring (`scoring.py`)
- **Template-specific weights** for all 5 templates + custom
- **Scorecards** with 5 metrics: Signal Strength, Impact/WTP, Reachability, Complexity, Moat
- Star ratings (★★★☆☆ format)

#### Radar (`radar.py`)
- **New**: Clusters not seen in past N days
- **Repeating**: Clusters appearing 2+ times
- **Rising**: Clusters with >20% engagement increase

#### Report (`report.py`)
- Combines all sections into structured markdown
- Prepares posts for LLM with cluster context

#### Historical Analysis (`historical.py`) - NEW
- Retrieves past runs with **full context** (digests, posts, clusters)
- Extracts ideas from past digest markdown
- Aggregates recurring patterns across runs
- Formats historical data for LLM consumption
- Generates LLM-powered trend insights

### 2. Database Updates (`src/database.py`)

- Added `run_clusters` table for Radar tracking
- Added `RunCluster` dataclass
- Methods: `insert_run_clusters()`, `get_run_clusters()`, `get_historical_clusters()`

### 3. Web Integration (`src/ui/web.py`)

- Integrated structured analysis into `run_analysis_job()`
- Cluster data stored after each run
- Combined structured sections with LLM output
- Added API endpoints:
  - `POST /api/historical-analysis` - Full LLM-powered historical analysis
  - `GET /api/historical-summary` - Fast summary without LLM

### 4. Feedback Fixes Applied

| Issue | Fix |
|-------|-----|
| `keyword_overlap` ~0 | Divides by `min(top_n, len(goal_tokens), 20)` |
| Substring false-positives | Exact token matching via precomputed sets |
| Default to `tools_tech` | Added `uncategorized` cluster |
| Single-label only | Multi-label with `secondary_clusters` |
| `recency_avg_hours` naming | Renamed to `avg_age_hours` + backwards compat |
| Aggressive saturation | `soft_saturate()` with exp decay |
| TF-IDF cosine similarity | Full implementation |
| Performance | Precomputed `CLUSTER_TOKEN_SETS` & `CLUSTER_TFIDF_VECTORS` |

## Test Status

- **39 tests pass**
- All modules import correctly
- Integration tests pass
- Historical analysis module tested

## Session 2 Updates (January 20, 2025)

### New Features Implemented

#### 1. Historical Analysis UI (`/historical`)
- Full-featured historical trends page in web UI
- Quick summary mode (no LLM cost)
- Deep analysis mode (LLM-powered insights)
- Recurring ideas and clusters visualization

#### 2. Subreddit Presets
- Save subreddit combinations for reuse
- Associate default template with each preset
- Full CRUD via Settings page and API

#### 3. Scheduled Runs
- Cron-like scheduling for automated analyses
- Common schedules: daily, weekdays, weekly, every 6/12 hours
- Enable/disable schedules
- Track last run and next run times

#### 4. Notifications
- Webhook notifications (Slack, Discord, etc.)
- Email notifications (SMTP)
- Trigger on: run_complete, run_failed
- Test notification functionality

#### 5. Settings Page (`/settings`)
- Tabs for Presets, Schedules, Notifications
- Create/delete management UI
- Toggle enable/disable for schedules and notifications

### Database Updates
- Added `subreddit_presets` table
- Added `schedules` table
- Added `notifications` table
- Updated `get_stats()` with new tables

### New Files Created
- `src/scheduler.py` - Cron parsing and scheduler thread
- `src/ui/templates/settings.html` - Settings page
- `src/ui/templates/historical.html` - Historical analysis page

### Template Change
- Replaced `startup_ideas` template with `content_ideas`
- Updated all references across codebase

### Documentation Updates
- Updated README.md with new features
- Updated API endpoints documentation
- Updated roadmap

## What Remains To Do

### 1. Potential Future Enhancements
- [ ] Multi-model debate mode (3 LLMs compare results)
- [ ] Export to Notion/Obsidian/JSON
- [ ] CLI command for historical analysis (`python main.py trends --llm`)
- [ ] Visualize cluster trends over time in charts
- [ ] Date range picker for historical analysis

## Files Modified/Created

### Created
- `src/analysis/clustering.py` - Complete rewrite with TF-IDF
- `src/analysis/scoring.py` - Template weights and scorecards
- `src/analysis/radar.py` - New/Repeating/Rising tracking
- `src/analysis/report.py` - Structured report generation
- `src/analysis/historical.py` - Historical analysis with LLM
- `src/analysis/__init__.py` - Module exports

### Modified
- `src/database.py` - Added run_clusters table and methods
- `src/ui/web.py` - Integrated analysis, added historical endpoints

## How To Test

```bash
# Run all tests
python3 -m pytest tests/ -v

# Test clustering
python3 -c "from src.analysis.clustering import assign_cluster; print(assign_cluster('AI sales automation tool', []))"

# Test historical (needs existing runs in DB)
python3 -c "
from src.database import get_database
from src.analysis.historical import get_historical_runs
db = get_database()
db.initialize()
runs = get_historical_runs(db, days=7)
print(f'Found {len(runs)} historical runs')
"
```

## API Endpoints

```
POST /api/historical-analysis
  - days: int (default 7)
  - template: str (optional filter)
  - Returns: LLM-generated insights about recurring patterns

GET /api/historical-summary?days=7&template=content_ideas
  - Fast, no LLM cost
  - Returns: recurring_ideas, recurring_clusters, counts
```
