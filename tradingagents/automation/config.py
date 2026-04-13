"""Unified configuration for the automated trading platform."""

import os
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.research import is_dynamic_universe, resolve_universe


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def build_config(overrides: dict = None) -> dict:
    """Build a complete config merging defaults + automation + overrides."""
    config = DEFAULT_CONFIG.copy()
    overrides = overrides or {}
    trading_universe = overrides.get("trading_universe", os.getenv("TRADING_UNIVERSE", "growth"))
    try:
        default_watchlist = [] if is_dynamic_universe(trading_universe) else resolve_universe(trading_universe)
    except Exception:
        trading_universe = "growth"
        default_watchlist = resolve_universe(trading_universe)

    automation_defaults = {
        # Alpaca
        "alpaca_api_key": os.getenv("ALPACA_API_KEY", ""),
        "alpaca_secret_key": os.getenv("ALPACA_SECRET_KEY", ""),
        "paper_trading": True,

        # Trading mode
        "trading_mode": "swing",  # "swing" | "day" | "both"
        "trading_universe": trading_universe,
        "watchlist": default_watchlist,
        "broad_market_enabled": is_dynamic_universe(trading_universe),
        "broad_market_min_price": float(os.getenv("BROAD_MARKET_MIN_PRICE", "10.0")),
        "broad_market_max_price": (
            float(os.getenv("BROAD_MARKET_MAX_PRICE"))
            if os.getenv("BROAD_MARKET_MAX_PRICE")
            else None
        ),
        "broad_market_min_prev_volume": float(
            os.getenv("BROAD_MARKET_MIN_PREV_VOLUME", "200000")
        ),
        "broad_market_min_prev_dollar_volume": float(
            os.getenv("BROAD_MARKET_MIN_PREV_DOLLAR_VOLUME", "25000000")
        ),
        "broad_market_min_avg_dollar_volume": float(
            os.getenv("BROAD_MARKET_MIN_AVG_DOLLAR_VOLUME", "20000000")
        ),
        "broad_market_max_seed_symbols": int(
            os.getenv("BROAD_MARKET_MAX_SEED_SYMBOLS", "600")
        ),
        "broad_market_max_candidates": int(
            os.getenv("BROAD_MARKET_MAX_CANDIDATES", "160")
        ),
        "broad_market_snapshot_batch_size": int(
            os.getenv("BROAD_MARKET_SNAPSHOT_BATCH_SIZE", "200")
        ),
        "broad_market_history_batch_size": int(
            os.getenv("BROAD_MARKET_HISTORY_BATCH_SIZE", "100")
        ),
        "broad_market_history_period": os.getenv("BROAD_MARKET_HISTORY_PERIOD", "1y"),
        "broad_market_exclude_funds": _env_flag("BROAD_MARKET_EXCLUDE_FUNDS", True),
        "broad_market_max_below_52w_high": float(
            os.getenv("BROAD_MARKET_MAX_BELOW_52W_HIGH", "0.30")
        ),
        "broad_market_min_above_52w_low": float(
            os.getenv("BROAD_MARKET_MIN_ABOVE_52W_LOW", "0.25")
        ),

        # Schedule (Eastern Time)
        "market_open_snapshot_time": "09:25",
        "swing_analysis_time": "15:30",
        "intraday_interval_minutes": int(os.getenv("INTRADAY_INTERVAL_MINUTES", "10")),
        "reflection_time": "16:30",
        "daily_summary_time": os.getenv("DAILY_SUMMARY_TIME", "16:40"),
        "weekly_summary_time": os.getenv("WEEKLY_SUMMARY_TIME", "16:45"),

        # Minervini swing-trading gate
        "minervini_enabled": True,
        "minervini_benchmark": "SPY",
        "minervini_db_path": os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "research_data",
            "market_data.duckdb",
        ),
        "minervini_lookback_days": 730,
        "minervini_min_rs_percentile": 70.0,
        "minervini_min_revenue_growth": 0.15,
        "minervini_min_eps_growth": 0.15,
        "minervini_min_roe": 0.15,
        "minervini_require_acceleration": False,
        "minervini_min_days_to_earnings": 5,
        "minervini_require_fundamentals": True,
        "minervini_require_market_uptrend": False,
        "minervini_max_stage_number": 3,
        "minervini_pivot_buffer_pct": 0.0,
        "minervini_max_buy_zone_pct": 0.07,
        "minervini_use_close_range_filter": True,
        "minervini_min_close_range_pct": 0.55,
        "minervini_allow_new_entries_in_correction": False,
        "minervini_target_exposure_confirmed_uptrend": 0.72,
        "minervini_target_exposure_uptrend_under_pressure": 0.48,
        "minervini_target_exposure_market_correction": 0.0,
        "leader_continuation_enabled": _env_flag("LEADER_CONTINUATION_ENABLED", True),
        "leader_continuation_min_rs_percentile": float(
            os.getenv("LEADER_CONTINUATION_MIN_RS_PERCENTILE", "75")
        ),
        "leader_continuation_min_close_range_pct": float(
            os.getenv("LEADER_CONTINUATION_MIN_CLOSE_RANGE_PCT", "0.15")
        ),
        "leader_continuation_min_adx_14": float(
            os.getenv("LEADER_CONTINUATION_MIN_ADX_14", "12")
        ),
        "leader_continuation_min_roc_60": float(
            os.getenv("LEADER_CONTINUATION_MIN_ROC_60", "0.0")
        ),
        "leader_continuation_min_roc_120": float(
            os.getenv("LEADER_CONTINUATION_MIN_ROC_120", "0.0")
        ),
        "leader_continuation_max_extension_pct": float(
            os.getenv("LEADER_CONTINUATION_MAX_EXTENSION_PCT", "0.07")
        ),
        "leader_continuation_max_pullback_pct": float(
            os.getenv("LEADER_CONTINUATION_MAX_PULLBACK_PCT", "0.08")
        ),
        "leader_continuation_allow_in_correction": _env_flag(
            "LEADER_CONTINUATION_ALLOW_IN_CORRECTION",
            True,
        ),
        "leader_continuation_target_exposure_confirmed_uptrend": float(
            os.getenv("LEADER_CONTINUATION_TARGET_EXPOSURE_CONFIRMED_UPTREND", "0.72")
        ),
        "leader_continuation_target_exposure_uptrend_under_pressure": float(
            os.getenv("LEADER_CONTINUATION_TARGET_EXPOSURE_UPTREND_UNDER_PRESSURE", "0.36")
        ),
        "leader_continuation_target_exposure_market_correction": float(
            os.getenv("LEADER_CONTINUATION_TARGET_EXPOSURE_MARKET_CORRECTION", "0.12")
        ),
        "leader_continuation_stop_loss_pct": float(
            os.getenv("LEADER_CONTINUATION_STOP_LOSS_PCT", "0.06")
        ),

        # Portable-alpha overlay
        "overlay_enabled": _env_flag("OVERLAY_ENABLED", False),
        "overlay_symbol": os.getenv("OVERLAY_SYMBOL", "SMH").upper(),
        "overlay_fraction": float(os.getenv("OVERLAY_FRACTION", "1.0")),
        "overlay_trigger": os.getenv("OVERLAY_TRIGGER", "confirmed_uptrend"),
        "overlay_rebalance_threshold_pct": float(
            os.getenv("OVERLAY_REBALANCE_THRESHOLD_PCT", "0.03")
        ),
        "overlay_min_notional": float(os.getenv("OVERLAY_MIN_NOTIONAL", "500")),
        "overlay_max_total_exposure": float(os.getenv("OVERLAY_MAX_TOTAL_EXPOSURE", "1.0")),
        "overlay_context_symbols": [
            item.strip().upper()
            for item in os.getenv("OVERLAY_CONTEXT_SYMBOLS", "SPY,QQQ,IWM,SMH,^VIX").split(",")
            if item.strip()
        ],
        "ntfy_enabled": _env_flag("NTFY_ENABLED", False),
        "ntfy_server": os.getenv("NTFY_SERVER", "https://ntfy.sh"),
        "ntfy_topic": os.getenv("NTFY_TOPIC", ""),
        "ntfy_priority": os.getenv("NTFY_PRIORITY", "default"),
        "ntfy_tags": [
            item.strip()
            for item in os.getenv("NTFY_TAGS", "trading_chart").split(",")
            if item.strip()
        ],
        "ntfy_click_url": os.getenv("NTFY_CLICK_URL", ""),
        "ntfy_morning_scan_enabled": _env_flag("NTFY_MORNING_SCAN_ENABLED", True),
        "ntfy_daily_summary_enabled": _env_flag("NTFY_DAILY_SUMMARY_ENABLED", True),
        "ntfy_weekly_summary_enabled": _env_flag("NTFY_WEEKLY_SUMMARY_ENABLED", True),
        "ntfy_morning_scan_top_n": int(os.getenv("NTFY_MORNING_SCAN_TOP_N", "5")),
        "ntfy_miss_review_enabled": _env_flag("NTFY_MISS_REVIEW_ENABLED", True),
        "ntfy_miss_review_top_n": int(os.getenv("NTFY_MISS_REVIEW_TOP_N", "5")),
        "ntfy_miss_review_near_buy_threshold_pct": float(
            os.getenv("NTFY_MISS_REVIEW_NEAR_BUY_THRESHOLD_PCT", "0.12")
        ),
        "social_monitor_enabled": _env_flag("SOCIAL_MONITOR_ENABLED", False),
        "social_monitor_usernames": [
            item.strip().lstrip("@")
            for item in os.getenv("SOCIAL_MONITOR_USERNAMES", "markminervini").split(",")
            if item.strip()
        ],
        "social_feed_url_template": os.getenv(
            "SOCIAL_FEED_URL_TEMPLATE",
            "https://nitter.net/{username}/rss",
        ),
        "social_check_interval_minutes": int(os.getenv("SOCIAL_CHECK_INTERVAL_MINUTES", "30")),
        "social_translation_model": os.getenv("SOCIAL_TRANSLATION_MODEL", "gpt-4o-mini"),
        "social_ntfy_enabled": _env_flag("SOCIAL_NTFY_ENABLED", False),
        "social_ntfy_server": os.getenv(
            "SOCIAL_NTFY_SERVER",
            os.getenv("NTFY_SERVER", "https://ntfy.sh"),
        ),
        "social_ntfy_topic": os.getenv("SOCIAL_NTFY_TOPIC", ""),
        "social_ntfy_priority": os.getenv("SOCIAL_NTFY_PRIORITY", "default"),
        "social_ntfy_tags": [
            item.strip()
            for item in os.getenv("SOCIAL_NTFY_TAGS", "newspaper,speech_balloon").split(",")
            if item.strip()
        ],
        "social_ntfy_click_url": os.getenv("SOCIAL_NTFY_CLICK_URL", ""),

        # Position sizing
        "max_position_pct": 0.12,
        "max_total_exposure": 0.72,
        "risk_per_trade": 0.012,

        # Risk controls
        "max_daily_loss": 0.03,
        "max_drawdown": 0.10,
        "min_cash_reserve": 0.20,
        "max_open_positions": int(os.getenv("MAX_OPEN_POSITIONS", "6")),
        "default_stop_loss_pct": 0.08,
        "default_take_profit_pct": 0.15,

        # Storage
        "db_path": os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "trading.db"
        ),

        # LLM (use cheaper model for automation to control costs)
        "llm_provider": "openai",
        "deep_think_llm": "gpt-4o-mini",
        "quick_think_llm": "gpt-4o-mini",
        "max_debate_rounds": 1,
        "max_risk_discuss_rounds": 1,

        # Data
        "data_vendors": {
            "core_stock_apis": "yfinance",
            "technical_indicators": "yfinance",
            "fundamental_data": "yfinance",
            "news_data": "yfinance",
        },
    }

    config.update(automation_defaults)
    if overrides:
        if "watchlist" not in overrides and overrides.get("trading_universe") != trading_universe:
            resolved_universe = overrides.get("trading_universe", trading_universe)
            config["watchlist"] = (
                [] if is_dynamic_universe(resolved_universe) else resolve_universe(resolved_universe)
            )
            config["broad_market_enabled"] = is_dynamic_universe(resolved_universe)
        config.update(overrides)

    return config


class AutomationConfig:
    """Typed access to automation configuration."""

    def __init__(self, config: dict = None):
        self._config = config or build_config()

    def __getitem__(self, key):
        return self._config[key]

    def get(self, key, default=None):
        return self._config.get(key, default)

    def to_dict(self) -> dict:
        return self._config.copy()

    @property
    def watchlist(self):
        return self._config["watchlist"]

    @property
    def paper_trading(self):
        return self._config["paper_trading"]

    @property
    def trading_mode(self):
        return self._config["trading_mode"]

    @property
    def db_path(self):
        return self._config["db_path"]

    def update(self, overrides: dict):
        self._config.update(overrides)
