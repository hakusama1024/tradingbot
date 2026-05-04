"""Approximate CAN SLIM screening primitives for historical research."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .minervini import MinerviniConfig, MinerviniScreener


@dataclass
class CANSLIMConfig:
    breakout_lookback: int = 20
    volume_window: int = 50
    min_avg_volume: float = 200_000
    min_avg_dollar_volume: float = 15_000_000
    near_52w_high_pct: float = 0.15
    pivot_buffer_pct: float = 0.0
    max_buy_zone_pct: float = 0.08
    volume_surge_multiple: float = 1.2
    min_close_range_pct: float = 0.50
    min_current_eps_growth: float = 0.25
    min_current_revenue_growth: float = 0.20
    min_annual_eps_growth: float = 0.20
    min_annual_revenue_growth: float = 0.15
    require_positive_eps: bool = True
    require_fundamentals: bool = False
    require_market_uptrend: bool = False


class CANSLIMScreener:
    """Price/volume-led CAN SLIM approximation with optional quarterly overlays."""

    FUNDAMENTAL_COLUMNS = [
        "current_revenue_yoy_growth",
        "current_eps_yoy_growth",
        "annual_revenue_growth",
        "annual_eps_growth",
        "quarterly_revenue",
        "quarterly_eps",
        "ttm_revenue",
        "ttm_eps",
        "canslim_current_ok",
        "canslim_annual_ok",
        "canslim_positive_eps_ok",
        "canslim_fundamental_ok",
    ]

    def __init__(self, config: CANSLIMConfig | None = None):
        self.config = config or CANSLIMConfig()
        self._price_engine = MinerviniScreener(
            MinerviniConfig(
                breakout_lookback=self.config.breakout_lookback,
                volume_window=self.config.volume_window,
                min_avg_volume=self.config.min_avg_volume,
                min_avg_dollar_volume=self.config.min_avg_dollar_volume,
                pivot_buffer_pct=self.config.pivot_buffer_pct,
                max_buy_zone_pct=self.config.max_buy_zone_pct,
                volume_surge_multiple=self.config.volume_surge_multiple,
                require_fundamentals=False,
                require_market_uptrend=False,
            )
        )

    def prepare_features(
        self,
        df: pd.DataFrame,
        quarterly_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        features = self._price_engine.prepare_features(df)
        if features.empty:
            return features

        features["canslim_buy_point"] = features["pivot_price"] * (
            1.0 + self.config.pivot_buffer_pct
        )
        features["canslim_buy_limit_price"] = features["canslim_buy_point"] * (
            1.0 + self.config.max_buy_zone_pct
        )
        features["near_52w_high"] = (
            features["close"] >= features["52w_high"] * (1.0 - self.config.near_52w_high_pct)
        )
        features["new_high_signal"] = (
            features["close"] >= features["52w_high"] * 0.95
        ) | features["breakout_signal"].fillna(False).astype(bool)
        features["canslim_volume_signal"] = (
            features["breakout_volume_ratio"] >= self.config.volume_surge_multiple
        )
        features["canslim_leader_signal"] = (
            (features["close"] > features["sma_50"])
            & (features["sma_50"] > features["sma_200"])
            & (features["roc_60"] > 0)
            & (features["roc_120"] > 0)
        )

        if quarterly_df is not None and not quarterly_df.empty:
            features = self._attach_quarterly_features(features, quarterly_df)
        else:
            for column in self.FUNDAMENTAL_COLUMNS:
                features[column] = pd.NA

        features["canslim_current_ok"] = (
            (
                pd.to_numeric(features["current_eps_yoy_growth"], errors="coerce")
                >= self.config.min_current_eps_growth
            )
            | (
                pd.to_numeric(features["current_revenue_yoy_growth"], errors="coerce")
                >= self.config.min_current_revenue_growth
            )
        )
        features["canslim_annual_ok"] = (
            (
                pd.to_numeric(features["annual_eps_growth"], errors="coerce")
                >= self.config.min_annual_eps_growth
            )
            | (
                pd.to_numeric(features["annual_revenue_growth"], errors="coerce")
                >= self.config.min_annual_revenue_growth
            )
        )
        quarterly_eps = pd.to_numeric(features["quarterly_eps"], errors="coerce")
        features["canslim_positive_eps_ok"] = quarterly_eps > 0
        if not self.config.require_positive_eps:
            features["canslim_positive_eps_ok"] = True

        if self.config.require_fundamentals:
            features["canslim_fundamental_ok"] = (
                features["canslim_current_ok"].fillna(False).astype(bool)
                & features["canslim_annual_ok"].fillna(False).astype(bool)
                & features["canslim_positive_eps_ok"].fillna(False).astype(bool)
            )
        else:
            features["canslim_fundamental_ok"] = True

        features["canslim_score"] = (
            features["near_52w_high"].fillna(False).astype(int)
            + features["new_high_signal"].fillna(False).astype(int)
            + features["canslim_volume_signal"].fillna(False).astype(int)
            + features["canslim_leader_signal"].fillna(False).astype(int)
            + features["canslim_current_ok"].fillna(False).astype(int)
            + features["canslim_annual_ok"].fillna(False).astype(int)
            + (features["close_range_pct"] >= self.config.min_close_range_pct).fillna(False).astype(int)
        )
        return features

    def _attach_quarterly_features(
        self,
        features: pd.DataFrame,
        quarterly_df: pd.DataFrame,
    ) -> pd.DataFrame:
        quarterly = quarterly_df.copy()
        if quarterly.empty:
            return features

        quarterly["fiscal_date"] = pd.to_datetime(quarterly["fiscal_date"])
        quarterly = quarterly.sort_values("fiscal_date").reset_index(drop=True)
        quarterly["quarterly_revenue"] = pd.to_numeric(
            quarterly["total_revenue"], errors="coerce"
        )
        quarterly["quarterly_eps"] = pd.to_numeric(
            quarterly["diluted_eps"], errors="coerce"
        )
        quarterly["current_revenue_yoy_growth"] = pd.to_numeric(
            quarterly["revenue_yoy_growth"], errors="coerce"
        )
        quarterly["current_eps_yoy_growth"] = pd.to_numeric(
            quarterly["eps_yoy_growth"], errors="coerce"
        )
        quarterly["ttm_revenue"] = quarterly["quarterly_revenue"].rolling(4).sum()
        quarterly["ttm_eps"] = quarterly["quarterly_eps"].rolling(4).sum()
        quarterly["prior_ttm_revenue"] = quarterly["ttm_revenue"].shift(4)
        quarterly["prior_ttm_eps"] = quarterly["ttm_eps"].shift(4)
        quarterly["annual_revenue_growth"] = (
            quarterly["ttm_revenue"] / quarterly["prior_ttm_revenue"] - 1.0
        )
        quarterly["annual_eps_growth"] = (
            quarterly["ttm_eps"] / quarterly["prior_ttm_eps"] - 1.0
        )

        merge_cols = [
            "fiscal_date",
            "quarterly_revenue",
            "quarterly_eps",
            "current_revenue_yoy_growth",
            "current_eps_yoy_growth",
            "ttm_revenue",
            "ttm_eps",
            "annual_revenue_growth",
            "annual_eps_growth",
        ]
        merge_frame = quarterly[merge_cols].dropna(how="all", subset=merge_cols[1:])

        merged = (
            features.reset_index()
            .rename(columns={"trade_date": "trade_date"})
            .sort_values("trade_date")
        )
        merged["trade_date"] = pd.to_datetime(merged["trade_date"])
        merged = pd.merge_asof(
            merged,
            merge_frame.sort_values("fiscal_date"),
            left_on="trade_date",
            right_on="fiscal_date",
            direction="backward",
        )
        merged = merged.drop(columns=["fiscal_date"], errors="ignore")
        merged = merged.set_index("trade_date")
        merged.index.name = features.index.name
        return merged
