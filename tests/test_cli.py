"""Tests for CLI commands."""

import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from click.testing import CliRunner


def test_cli_imports():
    """Test that CLI imports correctly."""
    from main import cli
    assert cli is not None


def test_cli_help():
    """Test CLI --help works."""
    from main import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "Reddit Radar" in result.output
    assert "run" in result.output
    assert "backfill" in result.output
    assert "trends" in result.output


def test_templates_command():
    """Test templates command lists templates."""
    from main import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["templates"])

    assert result.exit_code == 0
    assert "content_ideas" in result.output.lower() or "Content" in result.output


def test_backfill_help():
    """Test backfill command help."""
    from main import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["backfill", "--help"])

    assert result.exit_code == 0
    assert "--days" in result.output
    assert "--template" in result.output
    assert "--subreddits" in result.output


def test_trends_help():
    """Test trends command help."""
    from main import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["trends", "--help"])

    assert result.exit_code == 0
    assert "--days" in result.output
    assert "--top" in result.output
    assert "--subreddits" in result.output


def test_clear_cache_help():
    """Test clear-cache command help."""
    from main import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["clear-cache", "--help"])

    assert result.exit_code == 0
    assert "--days" in result.output


def test_run_help():
    """Test run command help."""
    from main import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["run", "--help"])

    assert result.exit_code == 0
    assert "--template" in result.output
    assert "--no-cache" in result.output


def test_stats_command():
    """Test stats command with empty database."""
    from main import cli

    runner = CliRunner()
    with runner.isolated_filesystem():
        # Create an empty database
        result = runner.invoke(cli, ["init"])
        assert result.exit_code == 0

        result = runner.invoke(cli, ["stats"])
        assert result.exit_code == 0
        assert "posts" in result.output.lower()
        assert "comments" in result.output.lower()


def test_history_command_empty():
    """Test history command with no runs."""
    from main import cli

    runner = CliRunner()
    with runner.isolated_filesystem():
        # Create an empty database
        result = runner.invoke(cli, ["init"])
        assert result.exit_code == 0

        result = runner.invoke(cli, ["history"])
        assert result.exit_code == 0
        assert "No runs found" in result.output


def test_trends_command_empty():
    """Test trends command with no posts."""
    from main import cli

    runner = CliRunner()
    with runner.isolated_filesystem():
        # Create an empty database
        result = runner.invoke(cli, ["init"])
        assert result.exit_code == 0

        result = runner.invoke(cli, ["trends"])
        assert result.exit_code == 0
        assert "No posts found" in result.output


def test_clear_cache_command():
    """Test clear-cache command."""
    from main import cli

    runner = CliRunner()
    with runner.isolated_filesystem():
        # Create an empty database
        result = runner.invoke(cli, ["init"])
        assert result.exit_code == 0

        result = runner.invoke(cli, ["clear-cache", "--days", "30"])
        assert result.exit_code == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
