"""后端入口

FastAPI 应用：静态看板、数据接口、AI 洞察接口，统一 JSON 返回格式
{code, msg, data}，开启 CORS，集成 APScheduler 定时采集。
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

import uvicorn
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.ai_service import (
    generate_city_analysis,
    generate_daily_overview,
    get_scenic_spots,
)
from app.database import (
    get_city_detail,
    get_city_trend,
    get_daily_avg_trend,
    get_heat_by_date,
    get_latest_date,
    init_db,
)
from app.data_collector import run_daily_collect

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

scheduler = BackgroundScheduler(timezone="Asia/Shanghai")

# AI 每日综述缓存：同一天只调用一次大模型
_ai_daily_cache: dict = {}


def ok(data=None, msg: str = "success") -> dict:
    """统一成功响应格式。"""
    return {"code": 0, "msg": msg, "data": data}


def fail(msg: str, code: int = 1) -> dict:
    """统一失败响应格式。"""
    return {"code": code, "msg": msg, "data": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动建表并保证当日有数据，退出时关闭调度器。"""
    init_db()
    today = date.today().isoformat()
    if get_latest_date() != today:
        logger.info("当日暂无数据，启动时自动采集一次")
        run_daily_collect()  # 内部已含模拟数据兜底
    # 每周一凌晨 2 点自动执行数据采集
    scheduler.add_job(run_daily_collect, "cron", day_of_week="mon", hour=2, minute=0, id="weekly_collect")
    scheduler.start()
    yield
    scheduler.shutdown()


app = FastAPI(
    title="全国旅游热度AI洞察看板",
    description="全国城市旅游热度数据采集、可视化与 AI 洞察",
    version="0.2.0",
    lifespan=lifespan,
)

# 跨域支持：允许所有来源
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件服务
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index() -> FileResponse:
    """看板主页面"""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health() -> dict:
    """健康检查"""
    return ok({"status": "ok", "service": "tourism-heat-dashboard"})


# ---------------------------------------------------------------------------
# 数据接口
# ---------------------------------------------------------------------------
@app.get("/api/overview")
async def overview() -> dict:
    """当日全国总览：平均热度、整体环比、TOP10、更新时间。"""
    latest = get_latest_date()
    if not latest:
        return fail("暂无数据，请先执行数据采集")
    rows = get_heat_by_date(latest, top_n=1000)
    if not rows:
        return fail("暂无数据")

    avg_heat = round(sum(r["heat_index"] for r in rows) / len(rows), 2)
    avg_mom = round(sum(r["mom_change"] or 0 for r in rows) / len(rows), 2)
    return ok(
        {
            "date": latest,
            "avg_heat": avg_heat,
            "avg_mom": avg_mom,
            "top10": rows[:10],
            "total": len(rows),
            "update_time": latest,
        }
    )


@app.get("/api/city/{city_name}")
async def city_detail(city_name: str) -> dict:
    """单城市详情 + 近7天趋势。"""
    detail = get_city_detail(city_name)
    if not detail:
        return fail(f"未找到城市「{city_name}」的数据")
    trend = get_city_trend(city_name, 7)
    return ok(
        {
            "city": city_name,
            "province": detail.get("province"),
            "latest": detail,
            "trend": trend,
            "scenic": get_scenic_spots(city_name),
        }
    )


@app.get("/api/map-data")
async def map_data() -> dict:
    """全国所有城市热度数据（用于地图渲染）。"""
    latest = get_latest_date()
    if not latest:
        return fail("暂无数据")
    rows = get_heat_by_date(latest, top_n=1000)
    data = [
        {
            "name": r["city"],
            "city": r["city"],
            "province": r["province"],
            "value": round(r["heat_index"], 2),
            "search_heat": r["search_heat"],
            "migration_heat": r["migration_heat"],
            "scenic_heat": r["scenic_heat"],
            "mom_change": r["mom_change"],
        }
        for r in rows
    ]
    return ok({"date": latest, "data": data})


@app.get("/api/trend")
async def trend(days: int = Query(7, ge=1, le=90)) -> dict:
    """近 N 天全国平均热度趋势。"""
    rows = get_daily_avg_trend(days)
    return ok({"data": rows})


# ---------------------------------------------------------------------------
# AI 接口
# ---------------------------------------------------------------------------
@app.get("/api/ai/daily-overview")
async def ai_daily_overview() -> dict:
    """AI 每日热度综述（当日缓存，同一天只调用一次大模型）。"""
    latest = get_latest_date()
    if not latest:
        return fail("暂无数据")

    if latest in _ai_daily_cache:
        cached = _ai_daily_cache[latest]
        return ok({"analysis": cached["analysis"], "advice": cached["advice"], "cached": True})

    rows = get_heat_by_date(latest, top_n=10)
    avg_mom = round(sum(r["mom_change"] or 0 for r in rows) / len(rows), 2) if rows else 0
    # 是否存在历史环比基准：累计至少 2 个采集日期才具备周环比计算条件
    has_history = len(get_daily_avg_trend(2)) >= 2
    content = generate_daily_overview({
        "date": latest,
        "avg_mom": avg_mom,
        "top10": rows,
        "has_history": has_history,
    })
    _ai_daily_cache[latest] = content
    return ok({"analysis": content["analysis"], "advice": content["advice"], "cached": False})


@app.get("/api/ai/city/{city_name}")
async def ai_city(city_name: str) -> dict:
    """AI 单城市热度解读。"""
    detail = get_city_detail(city_name)
    if not detail:
        return fail(f"未找到城市「{city_name}」的数据")
    trend = get_city_trend(city_name, 7)
    content = generate_city_analysis(
        city_name,
        {"latest": detail, "trend": trend, "scenic": get_scenic_spots(city_name)},
    )
    return ok({"city": city_name, "content": content})


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
