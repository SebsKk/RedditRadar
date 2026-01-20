#!/usr/bin/env python3
"""Reddit Radar - CLI Entry Point.

Turn Reddit into your daily, goal-specific research radar.
"""

import json
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

# Check for required dependencies
def check_dependencies():
    """Check if required dependencies are installed."""
    missing = []

    try:
        import click
    except ImportError:
        missing.append("click")

    try:
        import praw
    except ImportError:
        missing.append("praw")

    try:
        import openai
    except ImportError:
        missing.append("openai")

    try:
        import yaml
    except ImportError:
        missing.append("pyyaml")

    try:
        from dotenv import load_dotenv
    except ImportError:
        missing.append("python-dotenv")

    if missing:
        print("Missing required dependencies:")
        for dep in missing:
            print(f"  - {dep}")
        print("\nInstall them with:")
        print(f"  pip install {' '.join(missing)}")
        print("\nOr install all dependencies:")
        print("  pip install -r requirements.txt")
        sys.exit(1)


# Only import heavy dependencies after checking
check_dependencies()

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown

from src.config import get_config, reload_config, ensure_directories
from src.database import Database, Post, Comment, Run, RunItem, Digest, LLMCache, get_database
from src.reddit_client import get_reddit_client, RedditClient
from src.llm_client import get_llm_client, LLMClient
from src.templates import (
    get_template, list_templates, build_full_prompt,
    build_subreddit_discovery_prompt, format_posts_for_llm,
)
from src.ranker import (
    create_shortlist, attach_comments_to_ranked, ranked_to_llm_format,
)


console = Console()


def generate_run_id() -> str:
    """Generate a unique run ID."""
    return f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def hash_prompt(prompt: str) -> str:
    """Generate a hash for a prompt."""
    import hashlib
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]


def get_ids_from_shortlist(shortlist: list) -> tuple[list[str], list[str]]:
    """Extract post IDs and comment IDs from a shortlist.

    Args:
        shortlist: List of RankedPost objects.

    Returns:
        Tuple of (post_ids, comment_ids).
    """
    post_ids = [r.post.post_id for r in shortlist]
    comment_ids = []
    for r in shortlist:
        comment_ids.extend([c.comment_id for c in r.comments])
    return post_ids, comment_ids


@click.group()
@click.version_option(version="0.1.0", prog_name="reddit-radar")
def cli():
    """Reddit Radar - Turn Reddit into your research radar."""
    pass


@cli.command()
def init():
    """Initialize Reddit Radar (create database and directories)."""
    console.print("[bold]Initializing Reddit Radar...[/bold]\n")

    config = get_config()

    # Ensure output directory exists
    ensure_directories(config)
    console.print(f"[green]✓[/green] Created output directory: {config.output.directory}")

    # Initialize database
    db = get_database()
    db.initialize()
    console.print(f"[green]✓[/green] Initialized database: {config.database.path}")

    # Check credentials
    if config.reddit.is_valid():
        console.print("[green]✓[/green] Reddit credentials configured")
    else:
        console.print("[yellow]![/yellow] Reddit credentials not set (set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET)")

    if config.llm_credentials.has_key_for_provider(config.llm.provider):
        console.print(f"[green]✓[/green] LLM credentials configured ({config.llm.provider})")
    else:
        console.print(f"[yellow]![/yellow] LLM credentials not set for {config.llm.provider}")

    console.print("\n[bold green]Initialization complete![/bold green]")


@cli.command()
def templates():
    """List available goal templates."""
    console.print("\n[bold]Available Goal Templates[/bold]\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Description")

    for template in list_templates():
        table.add_row(template.id, template.name, template.description)

    console.print(table)
    console.print("\nUse with: [cyan]reddit-radar run --template <id>[/cyan]")


@cli.command()
@click.option("--goal", "-g", default=None, help="Research goal description")
@click.option("--count", "-c", default=5, type=click.Choice(["3", "5", "10"]), help="Number of subreddits")
def discover(goal: str | None, count: str):
    """Discover relevant subreddits using LLM."""
    config = get_config()

    if goal is None:
        template = get_template(config.goal.template)
        if template:
            goal = template.description
        else:
            goal = "general research"

    console.print(f"\n[bold]Discovering subreddits for:[/bold] {goal}\n")

    try:
        llm = get_llm_client()
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        return

    system_prompt, user_prompt = build_subreddit_discovery_prompt(goal, int(count))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Asking LLM for suggestions...", total=None)

        response = llm.generate(user_prompt, system_prompt, temperature=0.5)

    # Parse response
    lines = response.content.strip().split("\n")
    subreddits = [line.strip().replace("r/", "") for line in lines if line.strip()]
    subreddits = [s for s in subreddits if s and not s.startswith("#")]

    console.print("\n[bold green]Suggested Subreddits:[/bold green]\n")

    # Verify subreddits exist
    try:
        reddit = get_reddit_client()
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("#", style="dim")
        table.add_column("Subreddit", style="cyan")
        table.add_column("Status")
        table.add_column("Subscribers", justify="right")

        for i, sub in enumerate(subreddits[:int(count)], 1):
            info = reddit.get_subreddit_info(sub)
            if info:
                table.add_row(
                    str(i),
                    f"r/{info.name}",
                    "[green]✓ Valid[/green]",
                    f"{info.subscribers:,}"
                )
            else:
                table.add_row(str(i), f"r/{sub}", "[red]✗ Not found[/red]", "-")

        console.print(table)

    except Exception as e:
        console.print(f"[yellow]Warning:[/yellow] Could not verify subreddits: {e}")
        for i, sub in enumerate(subreddits[:int(count)], 1):
            console.print(f"  {i}. r/{sub}")

    console.print(f"\n[dim]Tokens used: {response.total_tokens}[/dim]")


@cli.command()
@click.option("--template", "-t", default=None, help="Goal template ID")
@click.option("--subreddits", "-s", default=None, help="Comma-separated subreddit list")
@click.option("--posts", "-p", default=None, type=click.Choice(["5", "10", "20"]), help="Posts per subreddit")
@click.option("--hours", "-h", default=None, type=click.Choice(["24", "72", "144"]), help="Time window in hours")
@click.option("--output", "-o", default=None, help="Output file path")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option("--no-cache", is_flag=True, help="Bypass LLM response cache")
def run(
    template: str | None,
    subreddits: str | None,
    posts: str | None,
    hours: str | None,
    output: str | None,
    verbose: bool,
    no_cache: bool,
):
    """Run a Reddit analysis."""
    config = get_config()

    # Determine template
    template_id = template or config.goal.template
    goal_template = get_template(template_id)
    if not goal_template:
        console.print(f"[red]Error:[/red] Unknown template: {template_id}")
        console.print("Use [cyan]reddit-radar templates[/cyan] to see available templates.")
        return

    console.print(Panel(
        f"[bold]{goal_template.name}[/bold]\n{goal_template.description}",
        title="Running Analysis",
        border_style="cyan"
    ))

    # Determine subreddits
    if subreddits:
        subreddit_list = [s.strip() for s in subreddits.split(",")]
    else:
        subreddit_list = goal_template.suggested_subreddits[:config.collection.subreddit_count]

    console.print(f"\n[bold]Subreddits:[/bold] {', '.join(f'r/{s}' for s in subreddit_list)}")

    # Settings
    posts_per_sub = int(posts) if posts else config.collection.posts_per_subreddit
    time_window = int(hours) if hours else config.collection.time_window_hours
    comments_per_post = config.collection.comments_per_post

    console.print(f"[bold]Settings:[/bold] {posts_per_sub} posts/sub, {time_window}h window, {comments_per_post} comments/post\n")

    # Initialize
    db = get_database()
    db.initialize()

    try:
        reddit = get_reddit_client()
        llm = get_llm_client()
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        return

    # Generate run ID
    run_id = generate_run_id()
    run_date = datetime.now().strftime("%Y-%m-%d")

    all_posts: list[Post] = []
    all_comments: list[Comment] = []

    # Fetch data
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        for sub in subreddit_list:
            task = progress.add_task(f"Fetching r/{sub}...", total=None)

            sub_posts, sub_comments = reddit.fetch_subreddit_data(
                subreddit=sub,
                posts_limit=posts_per_sub,
                comments_per_post=comments_per_post,
                time_window_hours=time_window,
                feed_mode=config.collection.feed_mode,
            )

            all_posts.extend(sub_posts)
            all_comments.extend(sub_comments)

            progress.update(task, description=f"[green]✓[/green] r/{sub} ({len(sub_posts)} posts)")

    console.print(f"\n[green]Fetched {len(all_posts)} posts and {len(all_comments)} comments[/green]")

    # Store in database
    db.insert_posts(all_posts)
    db.insert_comments(all_comments)

    if verbose:
        console.print("[dim]Stored data in database[/dim]")

    # Rank and shortlist
    goal_prompt = goal_template.user_prompt_template.format(content="")
    shortlist = create_shortlist(all_posts, goal_prompt, config.ranking.shortlist_size)

    # Attach comments
    attach_comments_to_ranked(
        shortlist,
        lambda post_id: db.get_comments_for_post(post_id, comments_per_post)
    )

    console.print(f"[green]Shortlisted {len(shortlist)} posts for analysis[/green]")

    if verbose:
        console.print("\n[bold]Top 5 posts by score:[/bold]")
        for i, ranked in enumerate(shortlist[:5], 1):
            console.print(f"  {i}. [{ranked.post.score}] {ranked.post.title[:60]}...")

    # Format for LLM
    posts_data = ranked_to_llm_format(shortlist)
    content = format_posts_for_llm(posts_data)

    # Build prompts
    system_prompt, user_prompt = build_full_prompt(
        goal_template,
        content,
        config.prompt_builder,
    )

    # Check cache before calling LLM
    post_ids, comment_ids = get_ids_from_shortlist(shortlist)
    cache_key = Database.compute_cache_key(
        goal_prompt=user_prompt,
        post_ids=post_ids,
        comment_ids=comment_ids,
        model=config.llm.model,
    )

    cached_response = None if no_cache else db.get_cached_response(cache_key)
    tokens_used = 0
    cache_hit = False

    if cached_response:
        # Use cached response
        markdown_output = cached_response.output_markdown
        cache_hit = True
        console.print("\n[bold green]Using cached analysis[/bold green] (use --no-cache to regenerate)")
    else:
        # Generate summary
        console.print("\n[bold]Generating analysis...[/bold]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Analyzing with LLM...", total=None)

            response = llm.generate(user_prompt, system_prompt)

            progress.update(task, description="[green]✓[/green] Analysis complete")

        markdown_output = response.content
        tokens_used = response.total_tokens

        # Cache the response
        db.cache_response(
            cache_key=cache_key,
            model=config.llm.model,
            input_hash=hash_prompt(user_prompt),
            output_markdown=markdown_output,
        )
        if verbose:
            console.print("[dim]Response cached for future use[/dim]")

    # Save to database
    run_record = Run(
        run_id=run_id,
        run_date=run_date,
        goal_template=template_id,
        goal_prompt_hash=hash_prompt(goal_prompt),
        settings_json=json.dumps({
            "posts_per_sub": posts_per_sub,
            "time_window_hours": time_window,
            "comments_per_post": comments_per_post,
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

    # Save digest
    digest = Digest(
        run_id=run_id,
        model_mode=config.llm.mode,
        models_json=json.dumps([config.llm.model]),
        markdown=markdown_output,
        card_json=None,  # TODO: Generate card data
        created_at=time.time(),
    )
    db.insert_digest(digest)

    # Write output file
    if output is None:
        output_dir = Path(config.output.directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / f"{run_date}_{template_id}.md"

    output_path = Path(output)

    # Build full markdown with header
    cache_note = " (cached)" if cache_hit else ""
    token_note = f"*Tokens used: {tokens_used}*" if tokens_used > 0 else "*Retrieved from cache*"

    full_markdown = f"""# Reddit Radar Report

**Date:** {run_date}
**Template:** {goal_template.name}
**Subreddits:** {', '.join(f'r/{s}' for s in subreddit_list)}
**Posts analyzed:** {len(shortlist)}

---

{markdown_output}

---

*Generated by Reddit Radar using {config.llm.model}{cache_note}*
{token_note}
"""

    output_path.write_text(full_markdown)

    console.print(f"\n[green]Report saved to:[/green] {output_path}")
    if tokens_used > 0:
        console.print(f"[dim]Tokens used: {tokens_used}[/dim]")
    else:
        console.print(f"[dim]Retrieved from cache (0 tokens used)[/dim]")

    # Preview
    console.print("\n[bold]Preview:[/bold]")
    console.print(Panel(
        Markdown(markdown_output[:1000] + ("..." if len(markdown_output) > 1000 else "")),
        border_style="dim"
    ))


@cli.command()
@click.option("--days", "-d", default=7, help="Number of days to show")
def history(days: int):
    """Show run history."""
    db = get_database()
    db.initialize()

    runs = db.get_recent_runs(limit=days * 5)  # Assume max 5 runs per day

    if not runs:
        console.print("[yellow]No runs found.[/yellow]")
        return

    console.print(f"\n[bold]Recent Runs (last {days} days)[/bold]\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Date", style="cyan")
    table.add_column("Template")
    table.add_column("Subreddits")
    table.add_column("Run ID", style="dim")

    for run in runs:
        subreddits = json.loads(run.subreddits_json)
        sub_preview = ", ".join(f"r/{s}" for s in subreddits[:3])
        if len(subreddits) > 3:
            sub_preview += f" +{len(subreddits) - 3}"

        table.add_row(
            run.run_date,
            run.goal_template,
            sub_preview,
            run.run_id[:20] + "...",
        )

    console.print(table)


@cli.command()
def stats():
    """Show database statistics."""
    db = get_database()
    db.initialize()

    stats = db.get_stats()

    console.print("\n[bold]Database Statistics[/bold]\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Table", style="cyan")
    table.add_column("Records", justify="right")

    for table_name, count in stats.items():
        table.add_row(table_name, str(count))

    console.print(table)


@cli.command()
def check():
    """Check configuration and credentials."""
    console.print("\n[bold]Configuration Check[/bold]\n")

    config = get_config()

    # Reddit credentials
    if config.reddit.is_valid():
        console.print("[green]✓[/green] Reddit credentials: Configured")
        try:
            reddit = get_reddit_client()
            reddit.verify_connection()
            console.print("[green]✓[/green] Reddit connection: Working")
        except Exception as e:
            console.print(f"[red]✗[/red] Reddit connection: {e}")
    else:
        console.print("[red]✗[/red] Reddit credentials: Not configured")

    # LLM credentials
    provider = config.llm.provider
    if config.llm_credentials.has_key_for_provider(provider):
        console.print(f"[green]✓[/green] LLM credentials ({provider}): Configured")
    else:
        console.print(f"[red]✗[/red] LLM credentials ({provider}): Not configured")

    # Database
    try:
        db = get_database()
        db.initialize()
        console.print(f"[green]✓[/green] Database: {config.database.path}")
    except Exception as e:
        console.print(f"[red]✗[/red] Database: {e}")

    # Output directory
    output_dir = Path(config.output.directory)
    if output_dir.exists():
        console.print(f"[green]✓[/green] Output directory: {output_dir}")
    else:
        console.print(f"[yellow]![/yellow] Output directory: {output_dir} (will be created)")

    console.print(f"\n[bold]Current Settings:[/bold]")
    console.print(f"  Template: {config.goal.template}")
    console.print(f"  LLM: {config.llm.provider}/{config.llm.model}")
    console.print(f"  Posts per sub: {config.collection.posts_per_subreddit}")
    console.print(f"  Time window: {config.collection.time_window_hours}h")


@cli.command()
@click.option("--days", "-d", default=7, type=int, help="Number of days to backfill")
@click.option("--template", "-t", required=True, help="Goal template ID to use")
@click.option("--subreddits", "-s", default=None, help="Comma-separated subreddit filter (optional)")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def backfill(days: int, template: str, subreddits: str | None, verbose: bool):
    """Re-analyze historical posts from the database.

    This command uses existing posts stored in the database to generate
    a new analysis without fetching from Reddit again.
    """
    config = get_config()

    # Validate template
    goal_template = get_template(template)
    if not goal_template:
        console.print(f"[red]Error:[/red] Unknown template: {template}")
        console.print("Use [cyan]reddit-radar templates[/cyan] to see available templates.")
        return

    console.print(Panel(
        f"[bold]{goal_template.name}[/bold]\nBackfilling last {days} days",
        title="Backfill Analysis",
        border_style="cyan"
    ))

    db = get_database()
    db.initialize()

    # Calculate time range
    cutoff_time = time.time() - (days * 24 * 60 * 60)

    # Get posts from database
    all_posts = db.get_posts_since(cutoff_time)

    if not all_posts:
        console.print("[yellow]No posts found in database for the specified time range.[/yellow]")
        console.print("Run [cyan]reddit-radar run[/cyan] first to fetch posts.")
        return

    # Filter by subreddits if specified
    subreddit_list = None
    if subreddits:
        subreddit_list = [s.strip().lower() for s in subreddits.split(",")]
        all_posts = [p for p in all_posts if p.subreddit.lower() in subreddit_list]

    if not all_posts:
        console.print("[yellow]No posts match the subreddit filter.[/yellow]")
        return

    # Get unique subreddits from posts
    found_subreddits = list(set(p.subreddit for p in all_posts))

    console.print(f"\n[bold]Found {len(all_posts)} posts from {len(found_subreddits)} subreddits[/bold]")
    if verbose:
        console.print(f"  Subreddits: {', '.join(f'r/{s}' for s in found_subreddits[:5])}")
        if len(found_subreddits) > 5:
            console.print(f"  ... and {len(found_subreddits) - 5} more")

    try:
        llm = get_llm_client()
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        return

    # Rank and shortlist
    goal_prompt = goal_template.user_prompt_template.format(content="")
    shortlist = create_shortlist(all_posts, goal_prompt, config.ranking.shortlist_size)

    # Attach comments from database
    comments_per_post = config.collection.comments_per_post
    attach_comments_to_ranked(
        shortlist,
        lambda post_id: db.get_comments_for_post(post_id, comments_per_post)
    )

    console.print(f"[green]Shortlisted {len(shortlist)} posts for analysis[/green]")

    # Format for LLM
    posts_data = ranked_to_llm_format(shortlist)
    content = format_posts_for_llm(posts_data)

    # Build prompts
    system_prompt, user_prompt = build_full_prompt(
        goal_template,
        content,
        config.prompt_builder,
    )

    # Generate summary
    console.print("\n[bold]Generating analysis...[/bold]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Analyzing with LLM...", total=None)

        response = llm.generate(user_prompt, system_prompt)

        progress.update(task, description="[green]✓[/green] Analysis complete")

    markdown_output = response.content

    # Generate run ID and save
    run_id = generate_run_id()
    run_date = datetime.now().strftime("%Y-%m-%d")

    run_record = Run(
        run_id=run_id,
        run_date=run_date,
        goal_template=template,
        goal_prompt_hash=hash_prompt(goal_prompt),
        settings_json=json.dumps({
            "backfill_days": days,
            "type": "backfill",
        }),
        subreddits_json=json.dumps(found_subreddits),
        created_at=time.time(),
    )
    db.insert_run(run_record)

    # Save run items
    run_items = [
        RunItem(run_id=run_id, post_id=r.post.post_id, rank_score=r.total_score)
        for r in shortlist
    ]
    db.insert_run_items(run_items)

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
    output_path = output_dir / f"{run_date}_{template}_backfill.md"

    full_markdown = f"""# Reddit Radar Backfill Report

**Date:** {run_date}
**Template:** {goal_template.name}
**Backfill Period:** Last {days} days
**Subreddits:** {', '.join(f'r/{s}' for s in found_subreddits)}
**Posts analyzed:** {len(shortlist)}

---

{markdown_output}

---

*Generated by Reddit Radar backfill using {config.llm.model}*
*Tokens used: {response.total_tokens}*
"""

    output_path.write_text(full_markdown)

    console.print(f"\n[green]Report saved to:[/green] {output_path}")
    console.print(f"[dim]Tokens used: {response.total_tokens}[/dim]")

    # Preview
    console.print("\n[bold]Preview:[/bold]")
    console.print(Panel(
        Markdown(markdown_output[:1000] + ("..." if len(markdown_output) > 1000 else "")),
        border_style="dim"
    ))


@cli.command()
@click.option("--days", "-d", default=7, type=int, help="Number of days to analyze")
@click.option("--top", "-t", default=20, type=int, help="Number of top keywords to show")
@click.option("--subreddits", "-s", default=None, help="Comma-separated subreddit filter (optional)")
def trends(days: int, top: int, subreddits: str | None):
    """Show trending keywords and topics over time.

    Analyzes post titles and bodies to identify frequently mentioned
    topics and their trends.
    """
    from collections import Counter

    db = get_database()
    db.initialize()

    # Calculate time range
    cutoff_time = time.time() - (days * 24 * 60 * 60)

    # Get posts from database
    all_posts = db.get_posts_since(cutoff_time)

    if not all_posts:
        console.print("[yellow]No posts found in database for the specified time range.[/yellow]")
        console.print("Run [cyan]reddit-radar run[/cyan] first to fetch posts.")
        return

    # Filter by subreddits if specified
    if subreddits:
        subreddit_list = [s.strip().lower() for s in subreddits.split(",")]
        all_posts = [p for p in all_posts if p.subreddit.lower() in subreddit_list]

    if not all_posts:
        console.print("[yellow]No posts match the subreddit filter.[/yellow]")
        return

    console.print(f"\n[bold]Keyword Trends (last {days} days)[/bold]\n")
    console.print(f"Analyzing {len(all_posts)} posts...\n")

    # Common stop words to ignore
    stop_words = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'to', 'of',
        'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
        'during', 'before', 'after', 'above', 'below', 'between', 'under',
        'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where',
        'why', 'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some',
        'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
        'very', 'just', 'and', 'but', 'if', 'or', 'because', 'until', 'while',
        'this', 'that', 'these', 'those', 'what', 'which', 'who', 'whom', 'any',
        'both', 'each', 'i', 'me', 'my', 'we', 'our', 'you', 'your', 'it', 'its',
        'they', 'them', 'their', 'about', 'like', 'get', 'got', 'want', 'know',
        'think', 'make', 'going', 'see', 'way', 'come', 'now', 'time', 'good',
        'new', 'first', 'last', 'long', 'great', 'little', 'own', 'other', 'old',
        'right', 'big', 'high', 'different', 'small', 'large', 'next', 'early',
        'young', 'important', 'few', 'public', 'bad', 'same', 'able', 'use',
        'using', 'used', 'also', 'well', 'back', 'much', 'go', 'many', 'really',
        'even', 'still', 'every', 'day', 'look', 'looking', 'found', 'take',
        'people', 'thing', 'things', 'one', 'two', 'three', 'year', 'years',
        'reddit', 'sub', 'post', 'anyone', 'someone', 'something', 'nothing',
        'everything', 'anything', 'everyone', 'help', 'please', 'thanks', 'thank',
        'question', 'questions', 'amp', 'http', 'https', 'www', 'com', 'org',
    }

    # Count keywords
    keyword_counter: Counter = Counter()
    subreddit_keywords: dict[str, Counter] = {}

    for post in all_posts:
        # Combine title and body
        text = post.title
        if post.body:
            text += " " + post.body

        # Tokenize
        import re
        words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9]*\b', text.lower())
        words = [w for w in words if len(w) >= 3 and w not in stop_words]

        keyword_counter.update(words)

        # Track by subreddit
        if post.subreddit not in subreddit_keywords:
            subreddit_keywords[post.subreddit] = Counter()
        subreddit_keywords[post.subreddit].update(words)

    # Display top keywords
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("#", style="dim", width=4)
    table.add_column("Keyword", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Top Subreddits", style="dim")

    for i, (keyword, count) in enumerate(keyword_counter.most_common(top), 1):
        # Find top subreddits for this keyword
        sub_counts = []
        for sub, sub_counter in subreddit_keywords.items():
            if keyword in sub_counter:
                sub_counts.append((sub, sub_counter[keyword]))
        sub_counts.sort(key=lambda x: x[1], reverse=True)
        top_subs = ", ".join(f"r/{s}" for s, _ in sub_counts[:3])

        table.add_row(str(i), keyword, str(count), top_subs)

    console.print(table)

    # Show subreddit breakdown
    console.print(f"\n[bold]Posts by Subreddit[/bold]\n")

    sub_table = Table(show_header=True, header_style="bold magenta")
    sub_table.add_column("Subreddit", style="cyan")
    sub_table.add_column("Posts", justify="right")
    sub_table.add_column("Top Keywords", style="dim")

    sub_post_counts = Counter(p.subreddit for p in all_posts)
    for sub, count in sub_post_counts.most_common(10):
        top_kw = ", ".join(kw for kw, _ in subreddit_keywords[sub].most_common(5))
        sub_table.add_row(f"r/{sub}", str(count), top_kw)

    console.print(sub_table)


@cli.command()
@click.option("--days", "-d", default=30, type=int, help="Clear cache older than N days")
def clear_cache(days: int):
    """Clear old LLM response cache entries."""
    db = get_database()
    db.initialize()

    deleted = db.clear_old_cache(days)

    if deleted > 0:
        console.print(f"[green]Cleared {deleted} cache entries older than {days} days[/green]")
    else:
        console.print(f"[yellow]No cache entries older than {days} days found[/yellow]")


@cli.command()
@click.option("--host", "-h", default="127.0.0.1", help="Host to bind to")
@click.option("--port", "-p", default=8000, help="Port to listen on")
@click.option("--reload", "-r", is_flag=True, help="Enable auto-reload for development")
def serve(host: str, port: int, reload: bool):
    """Launch the web UI."""
    import uvicorn

    console.print(f"\n[bold]Starting Reddit Radar Web UI[/bold]\n")
    console.print(f"  URL: [cyan]http://{host}:{port}[/cyan]")
    console.print(f"  Reload: {'enabled' if reload else 'disabled'}")
    console.print("\n[dim]Press Ctrl+C to stop[/dim]\n")

    uvicorn.run(
        "src.ui.web:app",
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    cli()
