"""Analysis module for RedditRadar.

Provides clustering, scoring, and structured output generation
that works across all templates.
"""

from src.analysis.clustering import (
    CORE_CLUSTERS,
    ClusterResult,
    PostFeatures,
    extract_keywords,
    assign_cluster,
    compute_post_features,
    aggregate_cluster_metrics,
    format_cluster_summary,
)

from src.analysis.scoring import (
    TemplateWeights,
    get_template_weights,
    compute_base_score,
    rank_by_cluster,
    generate_scorecards,
    format_scorecards,
    Scorecard,
)

from src.analysis.radar import (
    RadarSection,
    compute_radar,
    format_radar_section,
)

from src.analysis.report import (
    StructuredReport,
    prepare_posts_for_analysis,
    analyze_posts,
    generate_structured_report,
    clusters_to_db_format,
    format_posts_for_llm_structured,
    apply_llm_value_scores,
)

from src.analysis.historical import (
    HistoricalRunData,
    HistoricalAnalysisResult,
    get_historical_runs,
    run_historical_analysis_sync,
    format_historical_report,
)

__all__ = [
    # Clustering
    "CORE_CLUSTERS",
    "ClusterResult",
    "PostFeatures",
    "extract_keywords",
    "assign_cluster",
    "compute_post_features",
    "aggregate_cluster_metrics",
    "format_cluster_summary",
    # Scoring
    "TemplateWeights",
    "get_template_weights",
    "compute_base_score",
    "rank_by_cluster",
    "generate_scorecards",
    "format_scorecards",
    "Scorecard",
    # Radar
    "RadarSection",
    "compute_radar",
    "format_radar_section",
    # Report
    "StructuredReport",
    "prepare_posts_for_analysis",
    "analyze_posts",
    "generate_structured_report",
    "clusters_to_db_format",
    "format_posts_for_llm_structured",
    "apply_llm_value_scores",
    # Historical
    "HistoricalRunData",
    "HistoricalAnalysisResult",
    "get_historical_runs",
    "run_historical_analysis_sync",
    "format_historical_report",
]
