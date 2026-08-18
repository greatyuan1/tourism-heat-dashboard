"""数据库操作模块

基于 SQLite 提供基础连接与「每日城市热度表」的读写操作。
表字段：日期、城市名称、省份、综合热度指数、搜索热度、迁徙热度、景区热度、环比变化。
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

# 数据库文件路径：项目根目录 / data / tourism.db
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "tourism.db"

# 每日城市热度表建表语句
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS city_heat (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    date            TEXT    NOT NULL,            -- 日期 (YYYY-MM-DD)
    city            TEXT    NOT NULL,            -- 城市名称
    province        TEXT    NOT NULL,            -- 省份
    heat_index      REAL    NOT NULL DEFAULT 0,  -- 综合热度指数
    search_heat     REAL    NOT NULL DEFAULT 0,  -- 搜索热度
    migration_heat  REAL    NOT NULL DEFAULT 0,  -- 迁徙热度
    scenic_heat     REAL    NOT NULL DEFAULT 0,  -- 景区热度
    mom_change      REAL,                        -- 环比变化 (%)
    created_at      TEXT    DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (date, city)
);
"""


def get_connection() -> sqlite3.Connection:
    """获取 SQLite 数据库连接（行以 dict 形式返回）。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """初始化数据库，创建每日城市热度表。"""
    conn = get_connection()
    try:
        conn.execute(CREATE_TABLE_SQL)
        conn.commit()
    finally:
        conn.close()


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    """将 sqlite3.Row 转为普通字典。"""
    return {key: row[key] for key in row.keys()}


def upsert_city_heat(records: List[Dict[str, Any]]) -> int:
    """批量插入/更新城市热度数据（按 date + city 去重覆盖）。

    返回实际写入的记录条数。
    """
    if not records:
        return 0
    conn = get_connection()
    try:
        conn.executemany(
            """
            INSERT INTO city_heat
                (date, city, province, heat_index, search_heat,
                 migration_heat, scenic_heat, mom_change)
            VALUES
                (:date, :city, :province, :heat_index, :search_heat,
                 :migration_heat, :scenic_heat, :mom_change)
            ON CONFLICT(date, city) DO UPDATE SET
                province       = excluded.province,
                heat_index     = excluded.heat_index,
                search_heat    = excluded.search_heat,
                migration_heat = excluded.migration_heat,
                scenic_heat    = excluded.scenic_heat,
                mom_change     = excluded.mom_change
            """,
            records,
        )
        conn.commit()
        return len(records)
    finally:
        conn.close()


def get_latest_date() -> Optional[str]:
    """获取数据表中最新日期。"""
    conn = get_connection()
    try:
        row = conn.execute("SELECT MAX(date) AS d FROM city_heat").fetchone()
        return row["d"] if row and row["d"] else None
    finally:
        conn.close()


def get_heat_by_date(
    target_date: str,
    top_n: int = 20,
    province: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """查询指定日期的城市热度榜单（按综合热度降序）。"""
    conn = get_connection()
    try:
        sql = "SELECT * FROM city_heat WHERE date = ?"
        params: List[Any] = [target_date]
        if province:
            sql += " AND province = ?"
            params.append(province)
        sql += " ORDER BY heat_index DESC LIMIT ?"
        params.append(top_n)
        rows = conn.execute(sql, params).fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        conn.close()


def get_city_trend(city: str, days: int = 7) -> List[Dict[str, Any]]:
    """查询单城市最近 N 天热度趋势（按日期升序返回）。"""
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT date, heat_index, search_heat, migration_heat,
                   scenic_heat, mom_change
            FROM city_heat
            WHERE city = ?
            ORDER BY date DESC
            LIMIT ?
            """,
            (city, days),
        ).fetchall()
        result = [_row_to_dict(r) for r in rows]
        result.reverse()  # 转成日期升序，便于折线图展示
        return result
    finally:
        conn.close()


def get_city_detail(city: str) -> Optional[Dict[str, Any]]:
    """查询单城市最新一条热度记录。"""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM city_heat WHERE city = ? ORDER BY date DESC LIMIT 1",
            (city,),
        ).fetchone()
        return _row_to_dict(row) if row else None
    finally:
        conn.close()


def get_daily_avg_trend(days: int = 7) -> List[Dict[str, Any]]:
    """查询近 N 天全国平均热度趋势（按日期升序返回）。"""
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT date,
                   ROUND(AVG(heat_index), 2) AS avg_heat,
                   ROUND(AVG(mom_change), 2) AS avg_mom
            FROM city_heat
            GROUP BY date
            ORDER BY date DESC
            LIMIT ?
            """,
            (days,),
        ).fetchall()
        result = [_row_to_dict(r) for r in rows]
        result.reverse()
        return result
    finally:
        conn.close()
