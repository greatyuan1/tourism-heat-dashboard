#!/usr/bin/env python3
"""项目本地运行自检脚本

用法：python test.py
依次验证：数据库读写、数据采集、AI 调用是否正常。
"""
from __future__ import annotations

import logging

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

_PASS = 0


def run(name: str, fn) -> None:
    """执行单条检查并打印结果。"""
    global _PASS
    try:
        fn()
        _PASS += 1
        print(f"  ✅ {name}")
    except Exception as e:  # noqa: BLE001
        print(f"  ❌ {name}：{e}")


def main() -> None:
    print("=" * 52)
    print("  全国旅游热度AI洞察看板 —— 自检开始")
    print("=" * 52)

    # 1. 数据库建表与连接
    def check_db() -> None:
        from app.database import init_db, get_latest_date

        init_db()
        get_latest_date()  # 无异常即通过

    run("数据库建表与连接", check_db)

    # 2. 数据采集（30城入库 + 去重）
    def check_collect() -> None:
        from app.data_collector import run_daily_collect
        from app.database import get_heat_by_date, get_latest_date

        records = run_daily_collect()
        assert len(records) == 30, f"应采集30城，实际 {len(records)}"
        latest = get_latest_date()
        rows = get_heat_by_date(latest, top_n=1000)
        assert len(rows) >= 30, f"数据库应有30城数据，实际 {len(rows)}"

    run("数据采集（30城入库）", check_collect)

    # 3. AI 调用（每日综述，无 Key 时返回兜底文案）
    def check_ai() -> None:
        from app.ai_service import generate_daily_overview, is_configured

        sample = {
            "date": "2026-08-18",
            "avg_mom": 5.2,
            "top10": [
                {"city": "北京", "province": "北京市", "heat_index": 98.5, "mom_change": 2.1},
                {"city": "成都", "province": "四川省", "heat_index": 90.2, "mom_change": 6.5},
                {"city": "杭州", "province": "浙江省", "heat_index": 85.0, "mom_change": -1.2},
            ],
        }
        text = generate_daily_overview(sample)
        assert text and len(text) > 0, "AI 返回内容为空"
        if not is_configured():
            print("     ℹ️  未配置 DEEPSEEK_API_KEY，本次返回本地兜底文案")

    run("AI 每日综述调用", check_ai)

    print("=" * 52)
    print(f"  自检完成：{_PASS}/3 项通过")
    if _PASS == 3:
        print("  🎉 一切正常，可执行 uvicorn app.main:app --reload --port 8000 启动")
    else:
        print("  ⚠️ 存在失败项，请根据上方提示排查")
    print("=" * 52)


if __name__ == "__main__":
    main()
