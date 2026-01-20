"""Configuration management for Reddit Radar."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    yaml = None

try:
    from dotenv import load_dotenv
    # Load environment variables from .env file
    load_dotenv()
except ImportError:
    # dotenv not installed, environment variables must be set manually
    pass


@dataclass
class GoalConfig:
    """Goal configuration."""
    template: str = "content_ideas"
    custom_prompt: str = ""


@dataclass
class PromptBuilderConfig:
    """Prompt builder configuration."""
    output_format: str = "structured"  # bullets / brief / structured
    include_links: bool = True
    output_length: str = "medium"  # short / medium / long
    creativity: str = "medium"  # low / medium / high
    factuality: str = "balanced"  # strict / balanced / exploratory
    tone: str = "neutral"  # neutral / direct / playful
    extra_instructions: str = ""


@dataclass
class CollectionConfig:
    """Data collection configuration."""
    subreddit_count: int = 5  # 3 / 5 / 10
    posts_per_subreddit: int = 10  # 5 / 10 / 20
    comments_per_post: int = 5  # Always 5
    time_window_hours: int = 24  # 24 / 72 / 144
    feed_mode: str = "top"  # top / rising


@dataclass
class RankingConfig:
    """Ranking configuration."""
    shortlist_size: int = 25
    weight_score: float = 1.0
    weight_comments: float = 1.5
    weight_recency: float = 0.5
    weight_keywords: float = 2.0
    recency_half_life: int = 24


@dataclass
class LLMModelConfig:
    """Single LLM model configuration."""
    provider: str = "deepseek"
    model: str = "deepseek-chat"


@dataclass
class LLMConfig:
    """LLM configuration."""
    mode: str = "single"  # single / multi
    provider: str = "deepseek"
    model: str = "deepseek-chat"
    temperature: float = 0.7
    max_tokens: int = 4000
    multi_models: list[LLMModelConfig] = field(default_factory=list)


@dataclass
class OutputConfig:
    """Output configuration."""
    directory: str = "./output"
    date_format: str = "%Y-%m-%d"
    retention_days: int = 0


@dataclass
class DatabaseConfig:
    """Database configuration."""
    path: str = "./reddit_radar.db"


@dataclass
class UIConfig:
    """UI configuration."""
    host: str = "127.0.0.1"
    port: int = 8000
    debug: bool = False


@dataclass
class RedditCredentials:
    """Reddit API credentials from environment."""
    client_id: str = ""
    client_secret: str = ""
    user_agent: str = "RedditRadar/1.0"

    @classmethod
    def from_env(cls) -> "RedditCredentials":
        """Load credentials from environment variables."""
        return cls(
            client_id=os.getenv("REDDIT_CLIENT_ID", ""),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET", ""),
            user_agent=os.getenv("REDDIT_USER_AGENT", "RedditRadar/1.0"),
        )

    def is_valid(self) -> bool:
        """Check if credentials are configured."""
        return bool(self.client_id and self.client_secret)


@dataclass
class LLMCredentials:
    """LLM API credentials from environment."""
    deepseek_api_key: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    google_api_key: str = ""

    @classmethod
    def from_env(cls) -> "LLMCredentials":
        """Load credentials from environment variables."""
        return cls(
            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            google_api_key=os.getenv("GOOGLE_API_KEY", ""),
        )

    def get_key_for_provider(self, provider: str) -> str:
        """Get API key for a specific provider."""
        mapping = {
            "deepseek": self.deepseek_api_key,
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
            "google": self.google_api_key,
        }
        return mapping.get(provider, "")

    def has_key_for_provider(self, provider: str) -> bool:
        """Check if API key is configured for provider."""
        return bool(self.get_key_for_provider(provider))


@dataclass
class Config:
    """Main configuration container."""
    goal: GoalConfig = field(default_factory=GoalConfig)
    prompt_builder: PromptBuilderConfig = field(default_factory=PromptBuilderConfig)
    collection: CollectionConfig = field(default_factory=CollectionConfig)
    ranking: RankingConfig = field(default_factory=RankingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    ui: UIConfig = field(default_factory=UIConfig)

    # Credentials (loaded from environment)
    reddit: RedditCredentials = field(default_factory=RedditCredentials.from_env)
    llm_credentials: LLMCredentials = field(default_factory=LLMCredentials.from_env)


def _dict_to_dataclass(data: dict[str, Any], cls: type) -> Any:
    """Convert a dictionary to a dataclass instance."""
    if not data:
        return cls()

    # Get field names and types from the dataclass
    field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}

    kwargs = {}
    for key, value in data.items():
        if key in field_types:
            kwargs[key] = value

    return cls(**kwargs)


def load_config(config_path: str | Path | None = None) -> Config:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config file. If None, loads default config.

    Returns:
        Config object with all settings.
    """
    # If yaml is not available, return default config
    if not HAS_YAML:
        return Config()

    # Determine config file path
    if config_path is None:
        # Check for local config first, then default
        local_config = Path("config/local.yaml")
        default_config = Path("config/default.yaml")

        if local_config.exists():
            config_path = local_config
        elif default_config.exists():
            config_path = default_config
        else:
            # Return default config if no file exists
            return Config()

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load YAML
    with open(config_path, "r") as f:
        data = yaml.safe_load(f) or {}

    # Build config object
    config = Config(
        goal=_dict_to_dataclass(data.get("goal", {}), GoalConfig),
        prompt_builder=_dict_to_dataclass(
            data.get("prompt_builder", {}), PromptBuilderConfig
        ),
        collection=_dict_to_dataclass(data.get("collection", {}), CollectionConfig),
        ranking=_dict_to_dataclass(data.get("ranking", {}), RankingConfig),
        llm=_dict_to_dataclass(data.get("llm", {}), LLMConfig),
        output=_dict_to_dataclass(data.get("output", {}), OutputConfig),
        database=_dict_to_dataclass(data.get("database", {}), DatabaseConfig),
        ui=_dict_to_dataclass(data.get("ui", {}), UIConfig),
        reddit=RedditCredentials.from_env(),
        llm_credentials=LLMCredentials.from_env(),
    )

    # Handle multi_models specially (list of dicts -> list of dataclasses)
    if "llm" in data and "multi_models" in data["llm"]:
        config.llm.multi_models = [
            _dict_to_dataclass(m, LLMModelConfig)
            for m in data["llm"]["multi_models"]
        ]

    return config


def get_project_root() -> Path:
    """Get the project root directory."""
    # Start from this file's directory and go up until we find pyproject.toml
    current = Path(__file__).parent
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    # Fallback to current working directory
    return Path.cwd()


def ensure_directories(config: Config) -> None:
    """Ensure required directories exist."""
    output_dir = Path(config.output.directory)
    output_dir.mkdir(parents=True, exist_ok=True)


# Global config instance (lazy loaded)
_config: Config | None = None


def get_config() -> Config:
    """Get the global config instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reload_config(config_path: str | Path | None = None) -> Config:
    """Reload configuration from file."""
    global _config
    _config = load_config(config_path)
    return _config
