"""FastAPI Web UI for Reddit Radar.

A minimalist, professional web interface for running analyses
and viewing results in a clean card/tile format.
"""

import json
import logging
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, Form, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import markdown as md

from src.config import get_config
from src.database import get_database, Run, Digest, Post, Comment, RunItem, Database, RunCluster
from src.templates import list_templates, get_template, build_full_prompt, format_posts_for_llm, GoalTemplate
from src.discovery import smart_discover, format_subscriber_count, SubredditCandidate
from src.analysis import (
    prepare_posts_for_analysis,
    analyze_posts,
    generate_structured_report,
    clusters_to_db_format,
    format_posts_for_llm_structured,
    apply_llm_value_scores,
)

logger = logging.getLogger(__name__)


# Create FastAPI app
app = FastAPI(
    title="Reddit Radar",
    description="Turn Reddit into your research radar",
    version="0.1.0",
)

# Get paths
UI_DIR = Path(__file__).parent
STATIC_DIR = UI_DIR / "static"
TEMPLATES_DIR = UI_DIR / "templates"

# Ensure directories exist
STATIC_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR.mkdir(exist_ok=True)

# Setup templates
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Add markdown filter
def render_markdown(text: str) -> str:
    """Convert markdown to HTML."""
    if not text:
        return ""
    return md.markdown(text, extensions=['tables', 'fenced_code', 'nl2br'])


def timestamp_to_date(ts: float | None) -> str:
    """Convert Unix timestamp to readable date."""
    if ts is None:
        return "Never"
    return datetime.fromtimestamp(ts).strftime("%b %d, %Y at %H:%M")


def fromjson(text: str) -> list | dict:
    """Parse JSON string."""
    try:
        return json.loads(text) if text else []
    except (json.JSONDecodeError, TypeError):
        return []


templates.env.filters["markdown"] = render_markdown
templates.env.filters["timestamp_to_date"] = timestamp_to_date
templates.env.filters["fromjson"] = fromjson

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ============================================================================
# Background Job Management
# ============================================================================

class JobStatus:
    """Track status of background analysis jobs."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# In-memory job storage (simple for local app)
_jobs: dict[str, dict] = {}


def create_job(
    template: str,
    subreddits: list[str],
    posts_per_sub: int,
    time_window: int,
    comments_per_post: int = 5,
    custom_goal: str = "",
    llm_focus: str = "",
) -> str:
    """Create a new job and return its ID."""
    job_id = f"job_{uuid.uuid4().hex[:12]}"
    _jobs[job_id] = {
        "id": job_id,
        "status": JobStatus.PENDING,
        "template": template,
        "subreddits": subreddits,
        "posts_per_sub": posts_per_sub,
        "time_window": time_window,
        "comments_per_post": comments_per_post,
        "custom_goal": custom_goal,
        "llm_focus": llm_focus,
        "progress": "Initializing...",
        "run_id": None,
        "error": None,
        "created_at": time.time(),
    }
    return job_id


def get_job(job_id: str) -> Optional[dict]:
    """Get job status."""
    return _jobs.get(job_id)


def update_job(job_id: str, **kwargs):
    """Update job status."""
    if job_id in _jobs:
        _jobs[job_id].update(kwargs)


def run_analysis_job(job_id: str):
    """Execute an analysis job in the background."""
    job = get_job(job_id)
    if not job:
        return

    try:
        update_job(job_id, status=JobStatus.RUNNING, progress="Loading configuration...")
        logger.info(f"[Analysis] Starting job {job_id}")
        logger.info(f"[Analysis] Template: {job['template']}")
        logger.info(f"[Analysis] Subreddits from request: {job['subreddits']}")
        logger.info(f"[Analysis] Settings: {job['posts_per_sub']} posts/sub, {job.get('comments_per_post', 5)} comments/post, {job['time_window']}h window")
        if job.get('custom_goal'):
            logger.info(f"[Analysis] Custom goal: {job['custom_goal'][:100]}...")
        if job.get('llm_focus'):
            logger.info(f"[Analysis] LLM focus: {job['llm_focus'][:100]}...")

        config = get_config()
        db = get_database()
        db.initialize()

        # Get template or build custom goal
        custom_goal = job.get("custom_goal", "").strip()
        llm_focus = job.get("llm_focus", "").strip()

        if job["template"] == "custom" and custom_goal:
            # Build a custom template from user input
            goal_template = GoalTemplate(
                id="custom",
                name="Custom Analysis",
                description=custom_goal[:100],
                system_prompt=f"""You are an expert analyst. Your job is to analyze Reddit discussions
based on this research goal: {custom_goal}

Focus on extracting actionable insights, patterns, and key discussions.
Be specific and cite evidence from the posts.""",
                user_prompt_template="""Analyze these Reddit posts and comments based on the research goal.

Research Goal: """ + custom_goal + """

{content}

Provide a comprehensive analysis with:
## Key Findings
Main insights and patterns discovered.

## Top Discussions
Most relevant posts and why they matter.

## Actionable Insights
Specific recommendations or takeaways.

## Additional Observations
Other notable patterns or information.
""",
                suggested_subreddits=[],
            )
        else:
            goal_template = get_template(job["template"])
            if not goal_template:
                raise ValueError(f"Unknown template: {job['template']}")

        # Get subreddits
        subreddit_list = job["subreddits"]
        if not subreddit_list:
            logger.warning(f"[Analysis] No subreddits provided, using template defaults")
            if goal_template.suggested_subreddits:
                subreddit_list = goal_template.suggested_subreddits[:config.collection.subreddit_count]
                logger.info(f"[Analysis] Using template suggested subreddits: {subreddit_list}")
            else:
                raise ValueError("No subreddits specified. Please discover or add subreddits first.")
        else:
            logger.info(f"[Analysis] Using {len(subreddit_list)} user-selected subreddits: {subreddit_list}")

        update_job(job_id, progress=f"Fetching posts from {len(subreddit_list)} subreddits...")

        # Import clients
        from src.reddit_client import get_reddit_client
        from src.llm_client import get_llm_client
        from src.ranker import create_shortlist, attach_comments_to_ranked, ranked_to_llm_format

        reddit = get_reddit_client()
        llm = get_llm_client()

        # Fetch data
        all_posts: list[Post] = []
        all_comments: list[Comment] = []
        posts_per_sub = job["posts_per_sub"]
        time_window = job["time_window"]
        comments_per_post = job.get("comments_per_post", config.collection.comments_per_post)

        logger.info(f"[Analysis] Fetching {posts_per_sub} posts from each of {len(subreddit_list)} subreddits, {comments_per_post} comments each...")

        for i, sub in enumerate(subreddit_list, 1):
            update_job(job_id, progress=f"Fetching r/{sub} ({i}/{len(subreddit_list)})...")
            logger.info(f"[Analysis] Fetching r/{sub} ({i}/{len(subreddit_list)})...")

            sub_posts, sub_comments = reddit.fetch_subreddit_data(
                subreddit=sub,
                posts_limit=posts_per_sub,
                comments_per_post=comments_per_post,
                time_window_hours=time_window,
                feed_mode=config.collection.feed_mode,
            )
            logger.info(f"[Analysis]   -> Got {len(sub_posts)} posts, {len(sub_comments)} comments from r/{sub}")
            all_posts.extend(sub_posts)
            all_comments.extend(sub_comments)

        logger.info(f"[Analysis] Total fetched: {len(all_posts)} posts, {len(all_comments)} comments")
        update_job(job_id, progress=f"Fetched {len(all_posts)} posts, storing in database...")

        # Store in database
        db.insert_posts(all_posts)
        db.insert_comments(all_comments)

        # Rank and shortlist
        update_job(job_id, progress="Ranking and shortlisting posts...")

        # Use custom goal for ranking if provided, otherwise use template's default
        ranking_goal = custom_goal if custom_goal else goal_template.user_prompt_template.format(content="")
        # Use min_per_subreddit=2 to ensure all searched subreddits are represented
        shortlist = create_shortlist(
            all_posts,
            ranking_goal,
            config.ranking.shortlist_size,
            min_per_subreddit=2,
        )

        # Log subreddit distribution in shortlist for debugging
        shortlist_subs = {}
        for r in shortlist:
            sub = r.post.subreddit
            shortlist_subs[sub] = shortlist_subs.get(sub, 0) + 1
        logger.info(f"[Analysis] Shortlist subreddit distribution: {shortlist_subs}")

        # Attach comments
        attach_comments_to_ranked(
            shortlist,
            lambda post_id: db.get_comments_for_post(post_id, comments_per_post)
        )

        update_job(job_id, progress=f"Running structured analysis on {len(shortlist)} posts...")
        logger.info(f"[Analysis] Running structured analysis on {len(shortlist)} posts...")

        # Prepare posts for analysis (with truncation)
        posts_data = ranked_to_llm_format(shortlist)

        # Build comments lookup for analysis
        comments_by_post = {}
        for r in shortlist:
            post_id = r.post.post_id
            comments = db.get_comments_for_post(post_id, comments_per_post)
            comments_by_post[post_id] = [
                {"body": c.body, "score": c.score} for c in comments
            ]

        # Prepare posts with truncation for analysis
        prepared_posts = prepare_posts_for_analysis(
            posts_data,
            comments_by_post,
            max_body_chars=500,
            max_comment_chars=250,
            max_comments=5,
        )

        # Get goal prompt for keyword matching
        goal_prompt = goal_template.user_prompt_template.format(content="")
        if custom_goal:
            goal_prompt = custom_goal

        # Compute goal prompt hash for filtering historical runs
        import hashlib
        goal_prompt_hash = hashlib.sha256(goal_prompt.encode()).hexdigest()[:16]

        # Get historical cluster data for Radar - filtered by template and goal
        # This ensures we only compare similar analyses (e.g., gaming with gaming, not with startup)
        template_id = job["template"]
        historical_clusters = db.get_historical_clusters(
            days=7,
            goal_template=template_id,
            # For custom template, also filter by goal_prompt_hash so different custom goals don't mix
            goal_prompt_hash=goal_prompt_hash if template_id == "custom" else None,
        )
        logger.info(f"[Analysis] Found {len(historical_clusters)} historical cluster records for Radar (template={template_id})")

        # Generate structured report
        update_job(job_id, progress="Generating structured report (clusters, scores, radar)...")
        structured_report = generate_structured_report(
            posts=prepared_posts,
            goal_prompt=goal_prompt,
            template_id=job["template"],
            historical_clusters=historical_clusters,
            llm_content="",  # Will add LLM content after
        )

        logger.info(f"[Analysis] Structured report generated:")
        logger.info(f"[Analysis]   - {len(structured_report.clusters)} clusters found")
        logger.info(f"[Analysis]   - {len(structured_report.scorecards)} scorecards generated")
        logger.info(f"[Analysis]   - Radar: {len(structured_report.radar.new_clusters)} new, {len(structured_report.radar.repeating_clusters)} repeating, {len(structured_report.radar.rising_clusters)} rising")

        update_job(job_id, progress=f"Generating LLM analysis with {config.llm.model}...")

        # Format posts with cluster context for better LLM analysis
        content = format_posts_for_llm_structured(
            prepared_posts,
            structured_report.post_features,
            structured_report.clusters,
            max_posts=15,
        )

        # Build prompts
        system_prompt, user_prompt = build_full_prompt(
            goal_template,
            content,
            config.prompt_builder,
        )

        # Add LLM focus instructions if provided
        if llm_focus:
            system_prompt += f"\n\nAdditional Focus Areas:\n{llm_focus}"

        # Generate with LLM
        response = llm.generate(user_prompt, system_prompt)
        llm_output = response.content

        # Combine structured sections with LLM output
        markdown_output = combine_structured_and_llm_output(
            structured_report,
            llm_output,
            goal_template,
        )

        update_job(job_id, progress="Saving results...")

        # Generate run ID
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        run_date = datetime.now().strftime("%Y-%m-%d")

        # Hash prompt
        import hashlib
        goal_prompt_text = custom_goal if custom_goal else goal_template.user_prompt_template
        prompt_hash = hashlib.sha256(goal_prompt_text.encode()).hexdigest()[:16]

        # Save run record
        run_record = Run(
            run_id=run_id,
            run_date=run_date,
            goal_template=job["template"],
            goal_prompt_hash=prompt_hash,
            settings_json=json.dumps({
                "posts_per_sub": posts_per_sub,
                "time_window_hours": time_window,
                "comments_per_post": comments_per_post,
                "source": "web_ui",
            }),
            subreddits_json=json.dumps(subreddit_list),
            created_at=time.time(),
        )
        db.insert_run(run_record)

        # Save run items
        run_items = [
            RunItem(run_id=run_id, post_id=r.post.post_id, rank_score=r.total_score)
            for r in shortlist
        ]
        db.insert_run_items(run_items)

        # Apply LLM value scores to clusters for heatmap visualization
        goal_prompt_for_scoring = custom_goal if custom_goal else goal_template.user_prompt_template
        scored_clusters = apply_llm_value_scores(
            structured_report.clusters,
            goal_prompt_for_scoring,
            job["template"],
        )

        # Save cluster data for Radar tracking (with LLM value scores)
        run_clusters = clusters_to_db_format(scored_clusters, run_id)
        db.insert_run_clusters(run_clusters)
        logger.info(f"[Analysis] Saved {len(run_clusters)} cluster records for Radar tracking")

        # Save digest
        digest = Digest(
            run_id=run_id,
            model_mode=config.llm.mode,
            models_json=json.dumps([config.llm.model]),
            markdown=markdown_output,
            card_json=None,
            created_at=time.time(),
        )
        db.insert_digest(digest)

        # Write output file
        output_dir = Path(config.output.directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{run_date}_{job['template']}.md"

        # Build summary stats
        cluster_names = [c.cluster_name for c in structured_report.clusters[:5]]
        cluster_summary = ", ".join(cluster_names) if cluster_names else "None"

        full_markdown = f"""# Reddit Radar Report

**Date:** {run_date}
**Template:** {goal_template.name}
**Subreddits:** {', '.join(f'r/{s}' for s in subreddit_list)}
**Posts analyzed:** {len(shortlist)}
**Top clusters:** {cluster_summary}

---

{markdown_output}

---

*Generated by Reddit Radar Web UI using {config.llm.model}*
*Tokens used: {response.total_tokens}*
"""
        output_path.write_text(full_markdown, encoding='utf-8')

        update_job(
            job_id,
            status=JobStatus.COMPLETED,
            progress="Analysis complete!",
            run_id=run_id,
        )

    except Exception as e:
        logger.exception(f"Job {job_id} failed")
        update_job(
            job_id,
            status=JobStatus.FAILED,
            progress="Analysis failed",
            error=str(e),
        )


# ============================================================================
# Helper functions
# ============================================================================

def combine_structured_and_llm_output(
    structured_report,
    llm_output: str,
    goal_template: GoalTemplate,
) -> str:
    """Combine structured analysis sections with LLM output.

    The final report structure:
    1. Radar (New/Repeating/Rising)
    2. Cluster Summary
    3. Top 3 Items with Scorecards
    4. LLM Detailed Analysis
    5. Evidence & Links
    6. Next Actions
    """
    from src.analysis import (
        format_radar_section,
        format_cluster_summary,
        format_scorecards,
        get_template_weights,
    )

    weights = get_template_weights(structured_report.template_id)
    sections = []

    # 1. Radar
    sections.append(format_radar_section(structured_report.radar))

    # 2. Cluster Summary
    sections.append(format_cluster_summary(structured_report.clusters))

    # 3. Top Items with Scorecards
    sections.append(format_scorecards(structured_report.scorecards, weights.output_name))

    # 4. LLM Detailed Analysis
    if llm_output:
        sections.append("## Detailed Analysis\n")
        sections.append(llm_output)
        sections.append("")

    # 5. Evidence section (from clusters)
    evidence_lines = ["## Evidence & Links\n"]
    url_by_cluster = {}
    for c in structured_report.clusters:
        if c.top_urls:
            url_by_cluster[c.cluster_name] = c.top_urls

    if url_by_cluster:
        for cluster_name, urls in url_by_cluster.items():
            evidence_lines.append(f"**{cluster_name}:**")
            for url in urls[:3]:
                display_url = url[:60] + "..." if len(url) > 60 else url
                evidence_lines.append(f"- [{display_url}]({url})")
            evidence_lines.append("")
    else:
        evidence_lines.append("*No external links found in analyzed posts.*\n")

    sections.append("\n".join(evidence_lines))

    # 6. Next Actions based on top scorecards
    action_lines = ["## Next Actions\n"]
    if structured_report.scorecards:
        if weights.template_id == "content_ideas":
            action_lines.append("Content creation priorities:\n")
            for i, sc in enumerate(structured_report.scorecards[:3], 1):
                if sc.signal_strength >= 4:
                    action_lines.append(f"{i}. **Create content about {sc.title}**: High engagement signals strong audience interest.")
                elif sc.impact_wtp >= 4:
                    action_lines.append(f"{i}. **Explore {sc.title}**: Strong demand indicators. Consider a content series.")
                else:
                    action_lines.append(f"{i}. **Monitor {sc.title}**: Track this topic's engagement to confirm trend.")
        elif weights.template_id == "trend_radar":
            action_lines.append("Trend tracking recommendations:\n")
            for i, sc in enumerate(structured_report.scorecards[:3], 1):
                action_lines.append(f"{i}. **{sc.title}**: Set up alerts for this topic. Consider early experiments if signal strengthens.")
        elif weights.template_id == "career_intel":
            action_lines.append("Career development priorities:\n")
            for i, sc in enumerate(structured_report.scorecards[:3], 1):
                action_lines.append(f"{i}. **{sc.title}**: Allocate learning time to this area. Look for projects or side work to build portfolio evidence.")
        else:
            action_lines.append("Recommended follow-ups:\n")
            for i, sc in enumerate(structured_report.scorecards[:3], 1):
                action_lines.append(f"{i}. **{sc.title}**: Continue monitoring. Look for validation opportunities.")
    else:
        action_lines.append("*Run more analyses to generate actionable insights.*\n")

    sections.append("\n".join(action_lines))

    return "\n".join(sections)


def get_recent_runs_with_digests(limit: int = 10) -> list[dict]:
    """Get recent runs with their digest data."""
    db = get_database()
    db.initialize()

    runs = db.get_recent_runs(limit=limit)
    results = []

    for run in runs:
        digest = db.get_digest(run.run_id)
        template = get_template(run.goal_template)

        # Parse subreddits
        try:
            subreddits = json.loads(run.subreddits_json)
        except (json.JSONDecodeError, TypeError):
            subreddits = []

        # Extract insights from markdown (simple parsing)
        insights = []
        if digest and digest.markdown:
            lines = digest.markdown.split('\n')
            for i, line in enumerate(lines):
                if line.strip().startswith('###') and 'Idea' in line:
                    # Found an idea heading
                    idea_title = line.replace('###', '').strip()
                    insights.append(idea_title[:100])
                elif line.strip().startswith('1.') or line.strip().startswith('2.') or line.strip().startswith('3.'):
                    if '**' in line:
                        # Extract bold text as insight
                        parts = line.split('**')
                        if len(parts) >= 2:
                            insights.append(parts[1][:100])

                if len(insights) >= 3:
                    break

        results.append({
            "run": run,
            "digest": digest,
            "template": template,
            "subreddits": subreddits,
            "insights": insights[:3],
            "date_formatted": datetime.fromtimestamp(run.created_at).strftime("%b %d, %Y at %H:%M"),
        })

    return results


def parse_markdown_to_sections(markdown: str) -> dict:
    """Parse markdown into sections."""
    sections = {}
    current_section = "intro"
    current_content = []

    for line in markdown.split('\n'):
        if line.startswith('## '):
            if current_content:
                sections[current_section] = '\n'.join(current_content)
            current_section = line[3:].strip().lower().replace(' ', '_')
            current_content = []
        else:
            current_content.append(line)

    if current_content:
        sections[current_section] = '\n'.join(current_content)

    return sections


# ============================================================================
# Routes
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard view."""
    runs = get_recent_runs_with_digests(limit=6)
    goal_templates = list_templates()
    config = get_config()

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "runs": runs,
        "templates": goal_templates,
        "config": config,
    })


@app.get("/run/{run_id}", response_class=HTMLResponse)
async def view_run(request: Request, run_id: str):
    """View a specific run's results."""
    db = get_database()
    db.initialize()

    run = db.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    digest = db.get_digest(run_id)
    template = get_template(run.goal_template)

    # Parse data
    try:
        subreddits = json.loads(run.subreddits_json)
        settings = json.loads(run.settings_json)
    except (json.JSONDecodeError, TypeError):
        subreddits = []
        settings = {}

    # Parse markdown sections
    sections = {}
    if digest and digest.markdown:
        sections = parse_markdown_to_sections(digest.markdown)

    return templates.TemplateResponse("run_detail.html", {
        "request": request,
        "run": run,
        "digest": digest,
        "template": template,
        "subreddits": subreddits,
        "settings": settings,
        "sections": sections,
        "markdown": digest.markdown if digest else "",
        "date_formatted": datetime.fromtimestamp(run.created_at).strftime("%B %d, %Y at %H:%M"),
    })


@app.get("/new", response_class=HTMLResponse)
async def new_run_form(request: Request, template: Optional[str] = None):
    """Form to create a new analysis run."""
    goal_templates = list_templates()
    config = get_config()

    return templates.TemplateResponse("new_run.html", {
        "request": request,
        "templates": goal_templates,
        "config": config,
        "selected_template": template,
        "cli_command": None,
    })


@app.post("/new", response_class=HTMLResponse)
async def new_run_submit(
    request: Request,
    template: str = Form(...),
    subreddits: str = Form(""),
    posts_per_sub: int = Form(10),
    time_window: int = Form(24),
    subreddit_count: int = Form(5),
    run_now: bool = Form(False),
):
    """Handle form submission - run analysis or generate CLI command."""
    goal_templates = list_templates()
    config = get_config()

    # Parse subreddits
    sub_list = []
    if subreddits.strip():
        sub_list = [s.strip() for s in subreddits.split(",") if s.strip()]

    if run_now:
        # Create and start background job
        job_id = create_job(template, sub_list, posts_per_sub, time_window)

        # Start job in background thread
        thread = threading.Thread(target=run_analysis_job, args=(job_id,))
        thread.daemon = True
        thread.start()

        # Redirect to job status page
        return RedirectResponse(url=f"/job/{job_id}", status_code=303)

    # Generate CLI command only
    cmd_parts = ["python main.py run", f"--template {template}"]

    if sub_list:
        subs = ",".join(sub_list)
        cmd_parts.append(f'--subreddits "{subs}"')

    if posts_per_sub != 10:
        cmd_parts.append(f"--posts {posts_per_sub}")

    if time_window != 24:
        cmd_parts.append(f"--hours {time_window}")

    cli_command = " ".join(cmd_parts)

    return templates.TemplateResponse("new_run.html", {
        "request": request,
        "templates": goal_templates,
        "config": config,
        "selected_template": template,
        "cli_command": cli_command,
        "form_subreddits": subreddits,
        "form_posts_per_sub": posts_per_sub,
        "form_time_window": time_window,
    })


@app.get("/job/{job_id}", response_class=HTMLResponse)
async def view_job(request: Request, job_id: str):
    """View job status page."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    goal_template = get_template(job["template"])

    return templates.TemplateResponse("job_status.html", {
        "request": request,
        "job": job,
        "template": goal_template,
    })


@app.get("/history", response_class=HTMLResponse)
async def history(request: Request):
    """View run history."""
    runs = get_recent_runs_with_digests(limit=50)

    return templates.TemplateResponse("history.html", {
        "request": request,
        "runs": runs,
    })


@app.post("/api/run")
async def api_run(
    template: str = Form(...),
    subreddits: str = Form(""),
    posts_per_sub: int = Form(10),
    time_window: int = Form(24),
    comments_per_post: int = Form(5),
    subreddit_count: int = Form(5),
    run_now: bool = Form(False),
    custom_goal: str = Form(""),
    llm_focus: str = Form(""),
):
    """API endpoint to start an analysis job."""
    # Parse subreddits
    sub_list = []
    if subreddits.strip():
        sub_list = [s.strip() for s in subreddits.split(",") if s.strip()]

    # Create and start background job
    job_id = create_job(template, sub_list, posts_per_sub, time_window, comments_per_post, custom_goal, llm_focus)

    # Start job in background thread
    thread = threading.Thread(target=run_analysis_job, args=(job_id,))
    thread.daemon = True
    thread.start()

    return {"job_id": job_id, "status": "started"}


@app.post("/api/discover")
async def api_discover(
    goal: str = Form(...),
    use_llm: bool = Form(True),
    count: int = Form(15),
):
    """API endpoint to discover relevant subreddits.

    This combines Reddit search with optional LLM suggestions
    to find the most relevant subreddits for a given goal.
    Uses iterative discovery to meet the requested count.
    """
    try:
        # Ensure count is reasonable (between 5 and 30)
        target_count = max(5, min(30, count))

        result = smart_discover(
            goal=goal,
            target_count=target_count,
            use_llm=use_llm,
            min_subscribers=1000,  # Quality threshold
            max_iterations=3,  # Will keep searching up to 3 rounds
        )

        # Format for frontend
        subreddits = []
        for sub in result.subreddits:
            subreddits.append({
                "name": sub.name,
                "title": sub.title,
                "description": sub.description,
                "subscribers": sub.subscribers,
                "subscribers_formatted": format_subscriber_count(sub.subscribers),
                "is_active": sub.is_active,
                "relevance_score": sub.relevance_score,
                "source": sub.source,
            })

        return {
            "success": True,
            "query": result.query,
            "subreddits": subreddits,
            "llm_suggestions": result.llm_suggestions,
        }

    except Exception as e:
        logger.exception("Discovery failed")
        return {
            "success": False,
            "error": str(e),
            "subreddits": [],
        }


@app.get("/api/subreddit/{name}")
async def api_subreddit_info(name: str):
    """Get info about a specific subreddit."""
    try:
        from src.reddit_client import get_reddit_client

        client = get_reddit_client()
        info = client.get_subreddit_info(name)

        if info:
            return {
                "success": True,
                "subreddit": {
                    "name": info.name,
                    "title": info.title,
                    "description": info.description,
                    "subscribers": info.subscribers,
                    "subscribers_formatted": format_subscriber_count(info.subscribers),
                    "is_public": info.public,
                }
            }
        else:
            return {
                "success": False,
                "error": f"Subreddit r/{name} not found or is private",
            }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


@app.get("/api/job/{job_id}")
async def api_job_status(job_id: str):
    """API endpoint for job status."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/api/stats")
async def api_stats():
    """API endpoint for database stats."""
    db = get_database()
    db.initialize()
    return db.get_stats()


@app.get("/api/runs")
async def api_runs(limit: int = 10):
    """API endpoint for recent runs."""
    runs = get_recent_runs_with_digests(limit=limit)
    return [
        {
            "run_id": r["run"].run_id,
            "date": r["date_formatted"],
            "template": r["run"].goal_template,
            "subreddits": r["subreddits"],
            "insights": r["insights"],
        }
        for r in runs
    ]


@app.post("/api/historical-analysis")
async def api_historical_analysis(
    days: int = Form(7),
    template: str = Form(""),
):
    """Run historical analysis across past runs.

    Analyzes past runs to find:
    - Recurring ideas and themes
    - Persistent opportunities
    - Trend patterns over time

    Returns LLM-generated insights based on historical data.
    """
    try:
        from src.llm_client import get_llm_client
        from src.analysis.historical import (
            get_historical_runs,
            run_historical_analysis_sync,
            format_historical_report,
        )

        logger.info(f"[Historical] Starting analysis for past {days} days...")

        db = get_database()
        db.initialize()

        # Check if we have any historical data
        runs = get_historical_runs(db, days=days, template_filter=template or None)

        if not runs:
            return {
                "success": False,
                "error": f"No runs found in the past {days} days. Run some analyses first.",
                "runs_found": 0,
            }

        logger.info(f"[Historical] Found {len(runs)} runs to analyze")

        # Get LLM client
        llm = get_llm_client()

        # Run analysis
        result = run_historical_analysis_sync(
            db=db,
            llm_client=llm,
            days=days,
            template_filter=template or None,
        )

        # Format report
        report_markdown = format_historical_report(result)

        return {
            "success": True,
            "days_analyzed": result.days_analyzed,
            "runs_analyzed": result.runs_analyzed,
            "recurring_ideas": result.recurring_ideas[:10],
            "recurring_clusters": result.recurring_clusters[:10],
            "llm_analysis": result.llm_analysis,
            "report_markdown": report_markdown,
        }

    except Exception as e:
        logger.exception("Historical analysis failed")
        return {
            "success": False,
            "error": str(e),
        }


@app.get("/api/historical-summary")
async def api_historical_summary(days: int = 7, template: str = ""):
    """Get historical summary without LLM (faster, no API cost).

    Returns recurring ideas and clusters from past runs.
    """
    try:
        from src.analysis.historical import (
            get_historical_runs,
            aggregate_recurring_patterns,
        )

        db = get_database()
        db.initialize()

        runs = get_historical_runs(db, days=days, template_filter=template or None)

        if not runs:
            return {
                "success": True,
                "days_analyzed": days,
                "runs_found": 0,
                "recurring_ideas": [],
                "recurring_clusters": [],
                "message": "No historical data found. Run more analyses to build history.",
            }

        patterns = aggregate_recurring_patterns(runs)

        # Format recurring ideas
        recurring_ideas = []
        for idea, count in patterns["idea_frequency"].items():
            if count >= 2:
                dates = patterns["idea_timeline"].get(idea, [])
                recurring_ideas.append({
                    "idea": idea,
                    "frequency": count,
                    "dates": dates,
                })
        recurring_ideas.sort(key=lambda x: x["frequency"], reverse=True)

        # Format recurring clusters
        recurring_clusters = []
        for name, stats in patterns["cluster_stats"].items():
            recurring_clusters.append({
                "cluster": name,
                "appearances": stats["count"],
                "total_engagement": stats["total_engagement"],
            })
        recurring_clusters.sort(key=lambda x: x["appearances"], reverse=True)

        return {
            "success": True,
            "days_analyzed": days,
            "runs_found": len(runs),
            "recurring_ideas": recurring_ideas[:20],
            "recurring_clusters": recurring_clusters[:10],
            "all_ideas_count": len(patterns["idea_frequency"]),
        }

    except Exception as e:
        logger.exception("Historical summary failed")
        return {
            "success": False,
            "error": str(e),
        }


@app.get("/api/heatmap")
async def api_heatmap(days: int = 30, max_runs: int = 20):
    """Get cluster heatmap data for visualization.

    Returns data formatted for Chart.js matrix/heatmap plugin.
    """
    try:
        db = get_database()
        db.initialize()

        heatmap_data = db.get_heatmap_data(days=days, max_runs=max_runs)

        return {
            "success": True,
            "days": days,
            **heatmap_data,
        }

    except Exception as e:
        logger.exception("Heatmap data fetch failed")
        return {
            "success": False,
            "error": str(e),
            "runs": [],
            "clusters": [],
            "cluster_labels": {},
            "data": [],
        }


# ============================================================================
# Subreddit Presets API
# ============================================================================

@app.get("/api/presets")
async def api_list_presets():
    """List all subreddit presets."""
    db = get_database()
    db.initialize()
    presets = db.get_all_presets()
    return {
        "success": True,
        "presets": [
            {
                "preset_id": p.preset_id,
                "name": p.name,
                "description": p.description,
                "subreddits": json.loads(p.subreddits_json),
                "template": p.template,
                "created_at": p.created_at,
            }
            for p in presets
        ],
    }


@app.post("/api/presets")
async def api_create_preset(
    name: str = Form(...),
    description: str = Form(""),
    subreddits: str = Form(...),
    template: str = Form("content_ideas"),
):
    """Create a new subreddit preset."""
    from src.database import SubredditPreset
    import uuid

    # Parse subreddits
    sub_list = [s.strip() for s in subreddits.split(",") if s.strip()]
    if not sub_list:
        return {"success": False, "error": "At least one subreddit is required"}

    now = time.time()
    preset = SubredditPreset(
        preset_id=f"preset_{uuid.uuid4().hex[:12]}",
        name=name,
        description=description,
        subreddits_json=json.dumps(sub_list),
        template=template,
        created_at=now,
        updated_at=now,
    )

    db = get_database()
    db.initialize()
    db.insert_preset(preset)

    return {
        "success": True,
        "preset_id": preset.preset_id,
        "message": f"Preset '{name}' created with {len(sub_list)} subreddits",
    }


@app.delete("/api/presets/{preset_id}")
async def api_delete_preset(preset_id: str):
    """Delete a subreddit preset."""
    db = get_database()
    db.initialize()

    if db.delete_preset(preset_id):
        return {"success": True, "message": "Preset deleted"}
    return {"success": False, "error": "Preset not found"}


# ============================================================================
# Schedules API
# ============================================================================

@app.get("/api/schedules")
async def api_list_schedules():
    """List all schedules."""
    from src.scheduler import describe_cron, get_next_run_time

    db = get_database()
    db.initialize()
    schedules = db.get_all_schedules()

    return {
        "success": True,
        "schedules": [
            {
                "schedule_id": s.schedule_id,
                "name": s.name,
                "template": s.template,
                "subreddits": json.loads(s.subreddits_json) if s.subreddits_json else [],
                "preset_id": s.preset_id,
                "cron_expression": s.cron_expression,
                "cron_description": describe_cron(s.cron_expression),
                "posts_per_sub": s.posts_per_sub,
                "time_window_hours": s.time_window_hours,
                "enabled": s.enabled,
                "last_run_at": s.last_run_at,
                "next_run_at": s.next_run_at,
            }
            for s in schedules
        ],
    }


@app.post("/api/schedules")
async def api_create_schedule(
    name: str = Form(...),
    template: str = Form("content_ideas"),
    subreddits: str = Form(""),
    preset_id: str = Form(""),
    cron_expression: str = Form(...),
    posts_per_sub: int = Form(10),
    time_window_hours: int = Form(24),
    enabled: bool = Form(True),
):
    """Create a new schedule."""
    from src.database import Schedule
    from src.scheduler import get_next_run_time, parse_cron
    import uuid

    # Validate cron expression
    try:
        parse_cron(cron_expression)
    except ValueError as e:
        return {"success": False, "error": str(e)}

    # Parse subreddits
    sub_list = [s.strip() for s in subreddits.split(",") if s.strip()]

    # Must have either subreddits or preset
    if not sub_list and not preset_id:
        return {"success": False, "error": "Either subreddits or a preset is required"}

    now = time.time()
    next_run = get_next_run_time(cron_expression).timestamp() if enabled else None

    schedule = Schedule(
        schedule_id=f"sched_{uuid.uuid4().hex[:12]}",
        name=name,
        preset_id=preset_id if preset_id else None,
        template=template,
        subreddits_json=json.dumps(sub_list),
        posts_per_sub=posts_per_sub,
        time_window_hours=time_window_hours,
        cron_expression=cron_expression,
        enabled=enabled,
        last_run_at=None,
        next_run_at=next_run,
        created_at=now,
    )

    db = get_database()
    db.initialize()
    db.insert_schedule(schedule)

    return {
        "success": True,
        "schedule_id": schedule.schedule_id,
        "next_run_at": next_run,
        "message": f"Schedule '{name}' created",
    }


@app.put("/api/schedules/{schedule_id}/toggle")
async def api_toggle_schedule(schedule_id: str):
    """Enable or disable a schedule."""
    from src.scheduler import get_next_run_time

    db = get_database()
    db.initialize()

    schedule = db.get_schedule(schedule_id)
    if not schedule:
        return {"success": False, "error": "Schedule not found"}

    # Toggle enabled state
    new_enabled = not schedule.enabled
    next_run = get_next_run_time(schedule.cron_expression).timestamp() if new_enabled else None

    # Update in database
    schedule.enabled = new_enabled
    schedule.next_run_at = next_run
    db.insert_schedule(schedule)

    return {
        "success": True,
        "enabled": new_enabled,
        "next_run_at": next_run,
    }


@app.delete("/api/schedules/{schedule_id}")
async def api_delete_schedule(schedule_id: str):
    """Delete a schedule."""
    db = get_database()
    db.initialize()

    if db.delete_schedule(schedule_id):
        return {"success": True, "message": "Schedule deleted"}
    return {"success": False, "error": "Schedule not found"}


# ============================================================================
# Notifications API
# ============================================================================

@app.get("/api/notifications")
async def api_list_notifications():
    """List all notification configurations."""
    db = get_database()
    db.initialize()
    notifications = db.get_all_notifications()

    return {
        "success": True,
        "notifications": [
            {
                "notification_id": n.notification_id,
                "name": n.name,
                "type": n.notification_type,
                "trigger_on": n.trigger_on,
                "enabled": n.enabled,
                "config": json.loads(n.config_json),
            }
            for n in notifications
        ],
    }


@app.post("/api/notifications")
async def api_create_notification(
    name: str = Form(...),
    notification_type: str = Form(...),  # "webhook" or "email"
    trigger_on: str = Form("run_complete"),
    config: str = Form(...),  # JSON string
    enabled: bool = Form(True),
):
    """Create a new notification configuration."""
    from src.database import Notification
    import uuid

    # Validate type
    if notification_type not in ["webhook", "email"]:
        return {"success": False, "error": "Type must be 'webhook' or 'email'"}

    # Validate trigger
    valid_triggers = ["run_complete", "run_failed", "pattern_detected"]
    if trigger_on not in valid_triggers:
        return {"success": False, "error": f"Trigger must be one of: {valid_triggers}"}

    # Validate config JSON
    try:
        config_dict = json.loads(config)
    except json.JSONDecodeError:
        return {"success": False, "error": "Invalid config JSON"}

    # Validate required config fields
    if notification_type == "webhook" and "url" not in config_dict:
        return {"success": False, "error": "Webhook requires 'url' in config"}
    if notification_type == "email" and "to" not in config_dict:
        return {"success": False, "error": "Email requires 'to' in config"}

    notification = Notification(
        notification_id=f"notif_{uuid.uuid4().hex[:12]}",
        name=name,
        notification_type=notification_type,
        config_json=config,
        trigger_on=trigger_on,
        enabled=enabled,
        created_at=time.time(),
    )

    db = get_database()
    db.initialize()
    db.insert_notification(notification)

    return {
        "success": True,
        "notification_id": notification.notification_id,
        "message": f"Notification '{name}' created",
    }


@app.put("/api/notifications/{notification_id}/toggle")
async def api_toggle_notification(notification_id: str):
    """Enable or disable a notification."""
    db = get_database()
    db.initialize()

    notification = db.get_notification(notification_id)
    if not notification:
        return {"success": False, "error": "Notification not found"}

    notification.enabled = not notification.enabled
    db.insert_notification(notification)

    return {"success": True, "enabled": notification.enabled}


@app.delete("/api/notifications/{notification_id}")
async def api_delete_notification(notification_id: str):
    """Delete a notification configuration."""
    db = get_database()
    db.initialize()

    if db.delete_notification(notification_id):
        return {"success": True, "message": "Notification deleted"}
    return {"success": False, "error": "Notification not found"}


@app.post("/api/notifications/{notification_id}/test")
async def api_test_notification(notification_id: str):
    """Send a test notification."""
    from src.scheduler import send_notification

    db = get_database()
    db.initialize()

    notification = db.get_notification(notification_id)
    if not notification:
        return {"success": False, "error": "Notification not found"}

    try:
        send_notification(notification, {
            "event": "test",
            "message": "This is a test notification from Reddit Radar",
            "timestamp": datetime.now().isoformat(),
        })
        return {"success": True, "message": "Test notification sent"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================================================
# Settings & Historical Analysis Pages
# ============================================================================

@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    """Settings page for presets, schedules, and notifications."""
    db = get_database()
    db.initialize()

    presets = db.get_all_presets()
    schedules = db.get_all_schedules()
    notifications = db.get_all_notifications()
    goal_templates = list_templates()

    return templates.TemplateResponse("settings.html", {
        "request": request,
        "presets": presets,
        "schedules": schedules,
        "notifications": notifications,
        "templates": goal_templates,
    })


@app.get("/historical", response_class=HTMLResponse)
async def historical_page(request: Request):
    """Historical analysis page."""
    goal_templates = list_templates()

    return templates.TemplateResponse("historical.html", {
        "request": request,
        "templates": goal_templates,
    })


# ============================================================================
# Run the app
# ============================================================================

def run_server(host: str = "127.0.0.1", port: int = 8000, reload: bool = False):
    """Run the web server."""
    import uvicorn

    # Configure logging to show INFO level for our modules
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Set our modules to INFO level
    logging.getLogger("src.discovery").setLevel(logging.INFO)
    logging.getLogger("src.ui.web").setLevel(logging.INFO)

    uvicorn.run(
        "src.ui.web:app",
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    run_server()
