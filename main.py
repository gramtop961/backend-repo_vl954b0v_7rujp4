import os
from datetime import datetime, timedelta, timezone
from typing import List, Optional
from bson import ObjectId
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from database import db, create_document, get_documents
from schemas import (
    Settings, Checklist, ChecklistItem, Strategy, Trade, TradePlan, ChecklistStateItem,
    CreateTradeRequest, CloseTradeRequest, AnalyticsSummaryRequest, SentinelAnalyzeRequest
)

app = FastAPI(title="Inferno Core API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Utilities ----------

def _collection(name: str):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")
    return db[name]


def ensure_bootstrap(user_id: str):
    # Settings
    s = _collection("settings").find_one({"user_id": user_id})
    if not s:
        settings = Settings(user_id=user_id).model_dump()
        create_document("settings", settings)
    # Default checklist
    c = _collection("checklist").find_one({"user_id": user_id})
    if not c:
        items = [
            ChecklistItem(id="rr_gt_2", label="Is R:R >= 2?", required=True),
            ChecklistItem(id="rvol_gt_2", label="Daily RVOL >= 2?", required=True),
            ChecklistItem(id="news_checked", label="News checked?", required=True),
        ]
        checklist = Checklist(user_id=user_id, name="Core Discipline", items=items)
        create_document("checklist", checklist)
    # Strategy example
    st = _collection("strategy").find_one({"user_id": user_id})
    if not st:
        strategy = Strategy(user_id=user_id, name="Breakout from Dormancy", description="Primary momentum breakout")
        create_document("strategy", strategy)


# ---------- Health ----------
@app.get("/")
def read_root():
    return {"message": "Inferno Core Backend Running"}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set",
        "database_name": "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set",
        "connection_status": "Not Connected",
        "collections": []
    }
    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["connection_status"] = "Connected"
            response["collections"] = db.list_collection_names()
            response["database"] = "✅ Connected & Working"
    except Exception as e:
        response["database"] = f"⚠️  Connected but Error: {str(e)[:80]}"
    return response


# ---------- Settings & Checklist ----------
@app.get("/api/settings/{user_id}")
def get_settings(user_id: str):
    ensure_bootstrap(user_id)
    s = _collection("settings").find_one({"user_id": user_id})
    s["_id"] = str(s["_id"]) if s and "_id" in s else None
    return s


@app.get("/api/checklist/{user_id}")
def get_checklist(user_id: str):
    ensure_bootstrap(user_id)
    c = _collection("checklist").find_one({"user_id": user_id})
    if not c:
        raise HTTPException(status_code=404, detail="Checklist not found")
    c["_id"] = str(c["_id"]) if "_id" in c else None
    return c


# ---------- Trades ----------
@app.post("/api/trades")
def create_trade(payload: CreateTradeRequest):
    # Enforce checklist: all required items must be checked
    cl = _collection("checklist").find_one({"user_id": payload.user_id})
    if not cl:
        ensure_bootstrap(payload.user_id)
        cl = _collection("checklist").find_one({"user_id": payload.user_id})
    required_ids = {item["id"] for item in cl.get("items", []) if item.get("required", True)}
    checked_ids = {s.item_id for s in payload.checklist_state if s.checked}
    if not required_ids.issubset(checked_ids):
        missing = list(required_ids - checked_ids)
        raise HTTPException(status_code=400, detail={"message": "Checklist incomplete", "missing": missing})

    trade = Trade(
        user_id=payload.user_id,
        asset_class=payload.asset_class,
        ticker_pair=payload.ticker_pair,
        side=payload.side,
        entry=payload.entry,
        planned_stop=payload.planned_stop,
        planned_target=payload.planned_target,
        timeframe=payload.timeframe,
        strategy_tags=payload.strategy_tags,
        plan=payload.plan,
        checklist_state=payload.checklist_state,
        opened_at=datetime.now(timezone.utc),
    )
    trade_id = create_document("trade", trade)
    return {"id": trade_id, "status": "open"}


@app.get("/api/trades")
def list_trades(user_id: str, status: Optional[str] = Query(default=None)):
    filt = {"user_id": user_id}
    if status:
        filt["status"] = status
    trades = get_documents("trade", filt)
    for t in trades:
        t["_id"] = str(t["_id"])
    return trades


def _compute_r(entry: float, planned_stop: float, side: str) -> float:
    r = abs(entry - planned_stop)
    return r if r > 0 else 1e-9


def _pnl_r(entry: float, exit_price: float, planned_stop: float, side: str) -> float:
    r_unit = _compute_r(entry, planned_stop, side)
    if side == "long":
        pnl = (exit_price - entry) / r_unit
    else:
        pnl = (entry - exit_price) / r_unit
    return pnl


@app.post("/api/trades/{trade_id}/close")
def close_trade(trade_id: str, payload: CloseTradeRequest):
    t = _collection("trade").find_one({"_id": ObjectId(trade_id)})
    if not t:
        raise HTTPException(status_code=404, detail="Trade not found")

    pnl_R = _pnl_r(t["entry"], payload.exit_price, t["planned_stop"], t["side"])
    pnl_usd = None
    if t.get("units"):
        if t["side"] == "long":
            pnl_usd = (payload.exit_price - t["entry"]) * t["units"]
        else:
            pnl_usd = (t["entry"] - payload.exit_price) * t["units"]
        pnl_usd -= (t.get("fees", 0) or 0)

    win_loss = "be"
    if pnl_R > 0.05:
        win_loss = "win"
    elif pnl_R < -0.05:
        win_loss = "loss"

    update = {
        "status": "closed",
        "closed_at": payload.closed_at or datetime.now(timezone.utc),
        "actual_stop": payload.actual_stop,
        "actual_target": payload.actual_target,
        "pnl_R": pnl_R,
        "pnl_usd": pnl_usd,
        "win_loss": win_loss,
        "execution_rating": payload.execution_rating,
        "discipline_rating": payload.discipline_rating,
        "learnings": payload.learnings,
        "MFE_usd": payload.MFE_usd,
        "MAE_usd": payload.MAE_usd,
    }
    _collection("trade").update_one({"_id": ObjectId(trade_id)}, {"$set": update})
    return {"id": trade_id, "status": "closed", "pnl_R": pnl_R, "pnl_usd": pnl_usd}


# ---------- Analytics ----------
class AnalyticsResponse(BaseModel):
    win_rate: float
    avg_r_win: float
    avg_r_loss: float
    total_pnl_r: float
    total_pnl_usd: Optional[float] = None
    profit_factor: float
    expectancy_r: float
    max_drawdown_r: float
    current_drawdown_r: float
    discipline_score: float
    equity_curve: List[float]


def _compute_analytics(user_id: str, since: datetime) -> AnalyticsResponse:
    trades = list(_collection("trade").find({
        "user_id": user_id,
        "status": "closed",
        "closed_at": {"$gte": since}
    }).sort("closed_at", 1))

    pnl_r_list: List[float] = []
    pnl_dollar_list: List[float] = []
    wins: List[float] = []
    losses: List[float] = []
    eq: List[float] = []
    equity = 0.0

    checklist_penalties = 0.0
    deviation_penalties = 0.0
    overtrade_penalties = 0.0
    ratings = []

    # Overtrading: count per day
    per_day_counts = {}

    for t in trades:
        r = float(t.get("pnl_R") or 0.0)
        pnl_r_list.append(r)
        if t.get("pnl_usd") is not None:
            pnl_dollar_list.append(float(t.get("pnl_usd")))
        if r > 0:
            wins.append(r)
        elif r < 0:
            losses.append(abs(r))
        equity += r
        eq.append(equity)

        # ratings
        if t.get("execution_rating") and t.get("discipline_rating"):
            ratings.append((t["execution_rating"] + t["discipline_rating"]) / 2)

        # deviations: actual vs planned
        planned_stop = t.get("planned_stop")
        actual_stop = t.get("actual_stop")
        planned_target = t.get("planned_target")
        actual_target = t.get("actual_target")
        settings = _collection("settings").find_one({"user_id": user_id}) or {}
        sl_th = float(settings.get("deviation_threshold_sl_pct", 5.0))
        tp_th = float(settings.get("deviation_threshold_tp_pct", 10.0))
        if planned_stop and actual_stop:
            dev = abs((actual_stop - planned_stop) / planned_stop) * 100
            if dev > sl_th:
                deviation_penalties += 5
        if planned_target and actual_target:
            dev = abs((actual_target - planned_target) / planned_target) * 100
            if dev > tp_th:
                deviation_penalties += 5

        # checklist adherence (if any unchecked)
        state = t.get("checklist_state", [])
        if any((not s.get("checked")) for s in state):
            checklist_penalties += 15

        # overtrading
        day = (t.get("closed_at") or datetime.now(timezone.utc)).astimezone(timezone.utc).date().isoformat()
        per_day_counts[day] = per_day_counts.get(day, 0) + 1

    # overtrading penalties based on settings
    settings = _collection("settings").find_one({"user_id": user_id}) or {}
    max_day = int(settings.get("max_trades_day", 5))
    for day, count in per_day_counts.items():
        if count > max_day:
            overtrading_excess = count - max_day
            overtrading_penalties += 3 * overtrading_excess

    total_r = sum(pnl_r_list) if pnl_r_list else 0.0
    total_usd = sum(pnl_dollar_list) if pnl_dollar_list else None
    win_rate = (len(wins) / len(pnl_r_list) * 100.0) if pnl_r_list else 0.0
    avg_r_win = sum(wins) / len(wins) if wins else 0.0
    avg_r_loss = sum(losses) / len(losses) if losses else 0.0
    profit_factor = (sum(wins) / sum(losses)) if (wins and losses and sum(losses) > 0) else (sum(wins) / 0.00001 if wins else 0.0)
    expectancy = (win_rate / 100.0) * (avg_r_win) - (1 - (win_rate / 100.0)) * (avg_r_loss)

    # drawdowns
    max_dd = 0.0
    peak = 0.0
    current_dd = 0.0
    for v in eq:
        peak = max(peak, v)
        dd = peak - v
        max_dd = max(max_dd, dd)
        current_dd = dd

    # discipline score
    ratings_norm = 0.0
    if ratings:
        avg_rating = sum(ratings) / len(ratings)
        ratings_norm = (avg_rating - 1) / 4 * 100  # 1..5 -> 0..100
    score = 100.0
    score -= min(30, deviation_penalties)
    score -= min(15, overtrading_penalties)
    # checklist missing blocked by validation, but handle anyway
    score -= min(30, checklist_penalties)
    # blend ratings at 20%
    score = max(0.0, min(100.0, 0.8 * score + 0.2 * ratings_norm))

    return AnalyticsResponse(
        win_rate=round(win_rate, 2),
        avg_r_win=round(avg_r_win, 3),
        avg_r_loss=round(avg_r_loss, 3),
        total_pnl_r=round(total_r, 3),
        total_pnl_usd=round(total_usd, 2) if total_usd is not None else None,
        profit_factor=round(profit_factor, 3) if profit_factor else 0.0,
        expectancy_r=round(expectancy, 3),
        max_drawdown_r=round(max_dd, 3),
        current_drawdown_r=round(current_dd, 3),
        discipline_score=round(score, 1),
        equity_curve=[round(v, 3) for v in eq],
    )


@app.get("/api/analytics/summary")
def analytics_summary(user_id: str, period_days: int = 90):
    since = datetime.now(timezone.utc) - timedelta(days=period_days)
    return _compute_analytics(user_id, since)


# ---------- Sentinel Core ----------
class SentinelResponse(BaseModel):
    summary: str
    findings: List[str]
    recommendations: List[str]


@app.post("/api/sentinel/analyze")
def sentinel_analyze(payload: SentinelAnalyzeRequest):
    ensure_bootstrap(payload.user_id)
    # last N closed trades
    trades = list(_collection("trade").find({
        "user_id": payload.user_id,
        "status": "closed"
    }).sort("closed_at", -1).limit(payload.last_n))
    trades = list(reversed(trades))

    if not trades:
        return SentinelResponse(
            summary="No closed trades found.",
            findings=[],
            recommendations=["Complete at least one trade to enable analysis."]
        )

    # metrics
    settings = _collection("settings").find_one({"user_id": payload.user_id}) or {}
    sl_th = float(settings.get("deviation_threshold_sl_pct", 5.0))
    tp_th = float(settings.get("deviation_threshold_tp_pct", 10.0))

    sl_dev_count = 0
    tp_dev_count = 0
    mfe_gaps_r: List[float] = []

    for t in trades:
        # deviations
        ps = t.get("planned_stop")
        as_ = t.get("actual_stop")
        pt = t.get("planned_target")
        at = t.get("actual_target")
        if ps and as_:
            dev = abs((as_ - ps) / ps) * 100
            if dev > sl_th:
                sl_dev_count += 1
        if pt and at:
            dev = abs((at - pt) / pt) * 100
            if dev > tp_th:
                tp_dev_count += 1
        # MFE gap approximation using MFE_usd and per-trade R unit
        mfe_usd = t.get("MFE_usd")
        pnl_R = t.get("pnl_R") or 0.0
        if mfe_usd is not None and t.get("entry") and t.get("planned_stop"):
            r_unit = abs(t["entry"] - t["planned_stop"]) or 1e-9
            mfe_r = abs(mfe_usd) / r_unit
            gap = max(0.0, mfe_r - pnl_R)
            mfe_gaps_r.append(gap)

    n = len(trades)
    avg_gap = sum(mfe_gaps_r) / len(mfe_gaps_r) if mfe_gaps_r else 0.0

    summary = (
        f"Analyzed last {n} trades. Stop-loss deviations in {sl_dev_count}/{n}, take-profit deviations in {tp_dev_count}/{n}. "
        f"Average MFE minus realized result gap ≈ {avg_gap:.2f}R."
    )

    findings = [
        f"SL deviations > threshold: {sl_dev_count}/{n} (>{sl_th}%).",
        f"TP deviations > threshold: {tp_dev_count}/{n} (>{tp_th}%).",
        f"Avg MFE vs Final P&L gap: {avg_gap:.2f}R (potential money left on table).",
    ]

    recs = [
        "Adopt trailing stops when MFE exceeds 1R to capture additional run.",
        "Tighten adherence to planned stops; avoid widening stops post-entry.",
        "Define partial profit rules (e.g., take 1/3 at 1.5R; move stop to breakeven).",
        "Limit daily trades to your set maximum; skip suboptimal setups to avoid overtrading.",
    ]

    return SentinelResponse(summary=summary, findings=findings, recommendations=recs)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
