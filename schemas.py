"""
Inferno Core - Database Schemas (Pydantic)
Each class name maps to a MongoDB collection using its lowercase name.
"""
from __future__ import annotations
from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime

# Core user/settings/strategy/checklist
class Settings(BaseModel):
    user_id: str
    theme: str = "dark"
    max_trades_day: int = 5
    max_trades_week: int = 20
    deviation_threshold_sl_pct: float = 5.0
    deviation_threshold_tp_pct: float = 10.0
    default_timeframe: str = "1D"
    default_risk_per_trade: float = 1.0  # in R

class ChecklistItem(BaseModel):
    id: str
    label: str
    required: bool = True
    strategy_scoped: Optional[str] = None

class Checklist(BaseModel):
    user_id: str
    name: str
    items: List[ChecklistItem] = Field(default_factory=list)

class Strategy(BaseModel):
    user_id: str
    name: str
    description: Optional[str] = None
    presets: dict = Field(default_factory=dict)

# Trade and analytics
AssetClass = Literal["stock", "crypto", "forex"]
Side = Literal["long", "short"]
TradeStatus = Literal["open", "closed", "scratched"]

class ChecklistStateItem(BaseModel):
    item_id: str
    checked: bool
    timestamp: Optional[datetime] = None

class TradePlan(BaseModel):
    thesis: str
    exit_strategy: str

class Trade(BaseModel):
    user_id: str
    asset_class: AssetClass
    ticker_pair: str
    side: Side
    leverage: Optional[float] = None
    timeframe: str = "1D"
    status: TradeStatus = "open"
    opened_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None

    entry: float
    planned_stop: float
    planned_target: Optional[float] = None
    actual_stop: Optional[float] = None
    actual_target: Optional[float] = None

    units: Optional[float] = None
    position_value: Optional[float] = None
    risk_per_trade_R: Optional[float] = None
    max_stop_usd: Optional[float] = None
    fees: Optional[float] = 0.0
    slippage: Optional[float] = 0.0

    checklist_state: List[ChecklistStateItem] = Field(default_factory=list)
    plan: TradePlan

    pnl_usd: Optional[float] = None
    pnl_R: Optional[float] = None
    win_loss: Optional[Literal["win", "loss", "be"]] = None
    execution_rating: Optional[int] = Field(default=None, ge=1, le=5)
    discipline_rating: Optional[int] = Field(default=None, ge=1, le=5)
    learnings: Optional[str] = None

    MFE_usd: Optional[float] = None
    MFE_R: Optional[float] = None
    MAE_usd: Optional[float] = None
    MAE_R: Optional[float] = None

    strategy_tags: List[str] = Field(default_factory=list)
    attachments: List[str] = Field(default_factory=list)

class MarketSnapshot(BaseModel):
    trade_id: str
    symbol: str
    asset_class: AssetClass
    timeframe: str
    ohlc: List[dict] = Field(default_factory=list)

class AnalyticsSummaryRequest(BaseModel):
    user_id: str
    period_days: Optional[int] = 90

class SentinelAnalyzeRequest(BaseModel):
    user_id: str
    last_n: int = 10

class CreateTradeRequest(BaseModel):
    # Minimal subset for creating an open trade
    user_id: str
    asset_class: AssetClass
    ticker_pair: str
    side: Side
    entry: float
    planned_stop: float
    planned_target: Optional[float] = None
    timeframe: str = "1D"
    strategy_tags: List[str] = Field(default_factory=list)
    plan: TradePlan
    checklist_state: List[ChecklistStateItem]

class CloseTradeRequest(BaseModel):
    exit_price: float
    closed_at: Optional[datetime] = None
    execution_rating: int
    discipline_rating: int
    learnings: str
    actual_stop: Optional[float] = None
    actual_target: Optional[float] = None
    MFE_usd: Optional[float] = None
    MAE_usd: Optional[float] = None
