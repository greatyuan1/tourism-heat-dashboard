"""数据采集模块

采集全国 30 个热门旅游城市的每日热度数据，加权计算综合热度指数：

- 搜索热度（30%）：百度指数「城市名 + 旅游」关键词
- 迁徙热度（40%）：百度迁徙 城市每日迁入规模指数
- 景区热度（30%）：高德地图开放平台 城市热门景区平均热度

三个维度分别做 0-1 归一化后，按 搜索30% + 迁徙40% + 景区30% 权重合成综合热度指数，
再对比前一日计算环比变化，最后写入 SQLite（按「日期 + 城市」去重，重复执行不报错）。

**模拟数据模式**：当真实采集失败或无 API 密钥时，自动降级为合理的模拟数据，
保证看板始终可演示；单城市采集失败不影响整体运行（异常捕获 + 日志输出）。
"""
from __future__ import annotations

import json
import logging
import os
import random
import time
import zlib
from datetime import date, timedelta
from typing import Callable, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

from app.database import get_heat_by_date, init_db, upsert_city_heat

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
GAODE_API_KEY = os.getenv("GAODE_API_KEY", "")
BAIDU_COOKIE = os.getenv("BAIDU_COOKIE", "")  # 百度指数/迁徙所需的登录 Cookie（可选）
DATA_MODE = os.getenv("DATA_MODE", "auto").lower()  # auto / mock / real

# 综合热度权重（搜索 / 迁徙 / 景区）
WEIGHTS: Dict[str, float] = {"search": 0.30, "migration": 0.40, "scenic": 0.30}

# 接口地址
GAODE_PLACE_URL = "https://restapi.amap.com/v3/place/text"
BAIDU_MIGRATION_URL = "https://huiyan.baidu.com/migration/cityrank.jsonp"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)
REQUEST_DELAY = 0.2  # 相邻请求间隔（秒），避免触发限流

# 共享 Session（复用连接）
_http = requests.Session()
_http.headers.update({"User-Agent": USER_AGENT})

# 城市元信息：(城市名, 省份, 行政区划代码 adcode)
CITY_META: List[Tuple[str, str, str]] = [
    ("北京", "北京市", "110000"),
    ("上海", "上海市", "310000"),
    ("广州", "广东省", "440100"),
    ("深圳", "广东省", "440300"),
    ("成都", "四川省", "510100"),
    ("重庆", "重庆市", "500000"),
    ("杭州", "浙江省", "330100"),
    ("西安", "陕西省", "610100"),
    ("长沙", "湖南省", "430100"),
    ("南京", "江苏省", "320100"),
    ("武汉", "湖北省", "420100"),
    ("青岛", "山东省", "370200"),
    ("三亚", "海南省", "460200"),
    ("昆明", "云南省", "530100"),
    ("厦门", "福建省", "350200"),
    ("哈尔滨", "黑龙江省", "230100"),
    ("桂林", "广西壮族自治区", "450300"),
    ("丽江", "云南省", "530700"),
    ("苏州", "江苏省", "320500"),
    ("天津", "天津市", "120000"),
    ("郑州", "河南省", "410100"),
    ("大连", "辽宁省", "210200"),
    ("贵阳", "贵州省", "520100"),
    ("拉萨", "西藏自治区", "540100"),
    ("乌鲁木齐", "新疆维吾尔自治区", "650100"),
    ("黄山", "安徽省", "341000"),
    ("张家界", "湖南省", "430800"),
    ("呼和浩特", "内蒙古自治区", "150100"),
    ("银川", "宁夏回族自治区", "640100"),
    ("珠海", "广东省", "440400"),
]


class DataSourceUnavailable(Exception):
    """数据源不可用（缺少密钥或未接入），触发整维度降级为模拟数据。"""


# ---------------------------------------------------------------------------
# 模拟数据（兜底）
# ---------------------------------------------------------------------------
def _stable_seed(*parts) -> int:
    """基于 zlib.crc32 的稳定种子（跨进程一致，保证模拟数据可复现）。"""
    raw = "|".join(str(p) for p in parts).encode("utf-8")
    return zlib.crc32(raw)


# 各维度相对城市基础热度的系数区间（收窄以避免过度重塑排名）
_DIM_RANGE: Dict[str, Tuple[float, float]] = {
    "search": (0.82, 1.08),
    "migration": (0.78, 1.02),
    "scenic": (0.72, 0.98),
}

# 城市旅游热度权重（仅模拟数据用，反映真实热门程度，未列城市默认 1.0）
_POPULARITY: Dict[str, float] = {
    "北京": 1.25, "上海": 1.22, "成都": 1.18, "重庆": 1.16, "杭州": 1.15,
    "西安": 1.15, "长沙": 1.12, "广州": 1.12, "三亚": 1.10, "厦门": 1.08,
    "南京": 1.05, "武汉": 1.05, "深圳": 1.00, "青岛": 1.00, "苏州": 0.98,
    "昆明": 0.96, "丽江": 0.94, "桂林": 0.92, "珠海": 0.90, "哈尔滨": 0.90,
    "大连": 0.88, "天津": 0.85, "郑州": 0.84, "张家界": 0.82, "贵阳": 0.80,
    "黄山": 0.78, "拉萨": 0.75, "乌鲁木齐": 0.72, "呼和浩特": 0.68, "银川": 0.65,
}


def _mock_raw(city: str, dim_key: str, target_date: date) -> float:
    """生成单城市单维度的合理模拟原始值（城市间有差异、日间有波动）。

    基础热度由城市热度权重决定（稳定），叠加「市场统一日抖动 + 城市小噪声 +
    维度系数」，使热门城市稳定靠前、逐日环比波动维持在合理区间。
    """
    pop = _POPULARITY.get(city, 1.0)
    base = 55 + pop * 40  # 热度权重 1.25→105，0.65→81，稳定可控
    market = random.Random(_stable_seed("market", target_date.isoformat())).uniform(
        0.95, 1.05
    )  # 全国整体旅游热度日波动（所有城市共享）
    city_noise = random.Random(_stable_seed("noise", city, target_date.isoformat())).uniform(
        0.97, 1.03
    )
    lo, hi = _DIM_RANGE[dim_key]
    dim_factor = random.Random(_stable_seed("dim", city, dim_key)).uniform(lo, hi)
    return round(max(0.0, base * market * city_noise * dim_factor), 2)


def _fill_missing_with_mock(
    raw: Dict[str, Optional[float]], dim_key: str, target_date: date
) -> Dict[str, float]:
    """将采集失败的维度（值为 None）用模拟数据补齐。"""
    missing = [c for c, v in raw.items() if v is None]
    if missing:
        logger.info("[%s] 维度共 %d 城缺失，已用模拟数据补齐", dim_key, len(missing))
    return {c: (v if v is not None else _mock_raw(c, dim_key, target_date)) for c, v in raw.items()}


# ---------------------------------------------------------------------------
# 真实数据源采集（单城市，失败抛异常由上层兜底）
# ---------------------------------------------------------------------------
def _fetch_search_heat(city: str, province: str, adcode: str, target_date: date) -> float:
    """搜索热度：百度指数「城市名 + 旅游」。

    百度指数无官方公开 API，需登录 Cookie(BAIDU_COOKIE) 并解析加密参数(ptbk)。
    当前为占位实现，统一抛出 DataSourceUnavailable，由上层降级为模拟数据。
    """
    keyword = f"{city}旅游"
    if not BAIDU_COOKIE:
        raise DataSourceUnavailable("未配置 BAIDU_COOKIE，无法采集百度指数")
    # TODO: 携带 Cookie 请求 index.baidu.com，解析 ptbk 加密后的指数数值
    raise DataSourceUnavailable(f"百度指数「{keyword}」需加密参数解析，暂未接入")


def _fetch_migration_heat(city: str, province: str, adcode: str, target_date: date) -> float:
    """迁徙热度：百度迁徙 城市每日迁入规模指数。

    调用百度迁徙公开 JSONP 接口，以该城市迁入来源榜单的规模指数之和作为迁入热度代理。
    说明：接口可能校验 Referer/Cookie，且城市编码需使用百度城市码（此处以 adcode 近似）；
    失败时由上层降级为模拟数据。
    """
    params = {
        "dt": "city",
        "id": adcode,
        "type": "move_in",
        "date": target_date.strftime("%Y%m%d"),
    }
    headers = {"Referer": "https://qianxi.baidu.com/"}
    resp = _http.get(BAIDU_MIGRATION_URL, params=params, headers=headers, timeout=10)
    resp.raise_for_status()
    data = _parse_jsonp(resp.text)
    items = (data.get("data") or {}).get("list") or []
    if not items:
        raise ValueError("百度迁徙接口未返回迁入榜单")
    total = sum(float(it.get("value", 0)) for it in items)
    if total <= 0:
        raise ValueError("百度迁徙迁入规模指数为 0")
    return round(total, 2)


def _fetch_scenic_heat(city: str, province: str, adcode: str, target_date: date) -> float:
    """景区热度：高德地图开放平台 城市热门景区平均热度。

    通过高德 POI 检索接口获取城市景区类 POI，以「POI 数量 × (1 + 平均评分/5)」
    作为景区热度代理（高德开放接口不直接提供实时客流热度）。
    未配置 GAODE_API_KEY 时抛出 DataSourceUnavailable。
    """
    if not GAODE_API_KEY:
        raise DataSourceUnavailable("未配置 GAODE_API_KEY，无法调用高德接口")
    params = {
        "key": GAODE_API_KEY,
        "keywords": "景点|景区|公园",
        "city": adcode,
        "types": "110000",  # 高德 POI 分类：风景名胜大类
        "citylimit": "true",
        "offset": 25,
        "page": 1,
        "extensions": "all",
    }
    resp = _http.get(GAODE_PLACE_URL, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") != "1":
        raise RuntimeError(f"高德接口返回异常：{data.get('info', '')}")
    pois = data.get("pois") or []
    if not pois:
        raise ValueError("未检索到景区 POI")
    ratings = [float(p.get("biz_ext", {}).get("rating") or 0) for p in pois]
    avg_rating = sum(ratings) / len(ratings)
    return round(len(pois) * (1 + avg_rating / 5.0), 2)


def _parse_jsonp(text: str) -> dict:
    """解析 JSONP 响应（去除 callback(...) 包裹）。"""
    text = text.strip().strip(";")
    start, end = text.find("("), text.rfind(")")
    if start != -1 and end != -1 and end > start:
        text = text[start + 1 : end]
    return json.loads(text)


# ---------------------------------------------------------------------------
# 维度采集调度（逐城市、异常隔离）
# ---------------------------------------------------------------------------
def _collect_dimension(
    dim_key: str,
    fetcher: Callable[[str, str, str, date], float],
    target_date: date,
) -> Dict[str, Optional[float]]:
    """采集某维度全部城市数据。单城市失败记为 None，不影响整体。"""
    result: Dict[str, Optional[float]] = {}
    success = 0
    for city, province, adcode in CITY_META:
        try:
            result[city] = fetcher(city, province, adcode, target_date)
            success += 1
        except DataSourceUnavailable as e:
            # 缺密钥/未接入：整维度不再逐城尝试，直接降级
            logger.warning("[%s] 数据源不可用，整维度降级为模拟数据：%s", dim_key, e)
            result[city] = None
            for rest_city, _, _ in CITY_META[len(result) :]:
                result[rest_city] = None
            break
        except Exception as e:  # noqa: BLE001 —— 单城市失败不应拖垮整体
            logger.warning("[%s] 城市「%s」采集失败，将使用模拟数据：%s", dim_key, city, e)
            result[city] = None
        time.sleep(REQUEST_DELAY)
    logger.info("[%s] 维度采集完成：成功 %d / %d", dim_key, success, len(CITY_META))
    return result


# ---------------------------------------------------------------------------
# 归一化、加权合成、环比
# ---------------------------------------------------------------------------
def _minmax_normalize(values: Dict[str, float]) -> Dict[str, float]:
    """对维度原始值做 0-1 min-max 归一化（跨城市）。"""
    vals = list(values.values())
    if not vals:
        return {k: 0.0 for k in values}
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return {k: 0.5 for k in values}
    return {k: (v - lo) / (hi - lo) for k, v in values.items()}


def _synthesize_records(
    target_date: date,
    search: Dict[str, float],
    migration: Dict[str, float],
    scenic: Dict[str, float],
) -> List[dict]:
    """归一化 + 加权合成综合热度指数，构造入库记录。"""
    n_search = _minmax_normalize(search)
    n_migration = _minmax_normalize(migration)
    n_scenic = _minmax_normalize(scenic)

    records: List[dict] = []
    for city, province, _ in CITY_META:
        s, m, sc = n_search[city], n_migration[city], n_scenic[city]
        heat_index = (
            WEIGHTS["search"] * s + WEIGHTS["migration"] * m + WEIGHTS["scenic"] * sc
        ) * 100
        records.append(
            {
                "date": target_date.isoformat(),
                "city": city,
                "province": province,
                "heat_index": round(heat_index, 2),
                "search_heat": round(s * 100, 2),
                "migration_heat": round(m * 100, 2),
                "scenic_heat": round(sc * 100, 2),
                "mom_change": None,  # 后续填充
            }
        )
    records.sort(key=lambda r: r["heat_index"], reverse=True)
    return records


def _attach_mom_change(records: List[dict], target_date: date) -> None:
    """对比前一日综合热度，计算环比变化（%）。"""
    prev_date = (target_date - timedelta(days=1)).isoformat()
    prev_map = {r["city"]: r["heat_index"] for r in get_heat_by_date(prev_date, top_n=1000)}
    for r in records:
        prev = prev_map.get(r["city"])
        if prev:
            r["mom_change"] = round((r["heat_index"] - prev) / prev * 100, 2)
        else:
            r["mom_change"] = 0.0


# ---------------------------------------------------------------------------
# 主流程 / 入口
# ---------------------------------------------------------------------------
def _should_use_mock() -> bool:
    """判断是否整体进入模拟数据模式。"""
    if DATA_MODE == "mock":
        return True
    if DATA_MODE == "real":
        return False
    # auto：无任何数据源密钥时直接模拟，避免离线环境启动缓慢
    return not GAODE_API_KEY and not BAIDU_COOKIE


def _collect_mock(target_date: date) -> List[dict]:
    """纯模拟采集：三维度全部使用模拟数据。"""
    search = {c: _mock_raw(c, "search", target_date) for c, _, _ in CITY_META}
    migration = {c: _mock_raw(c, "migration", target_date) for c, _, _ in CITY_META}
    scenic = {c: _mock_raw(c, "scenic", target_date) for c, _, _ in CITY_META}
    records = _synthesize_records(target_date, search, migration, scenic)
    _attach_mom_change(records, target_date)
    upsert_city_heat(records)
    return records


def collect_city_heat(target_date: Optional[date] = None) -> List[dict]:
    """采集指定日期 30 城热度数据并写入数据库（按日期+城市去重）。"""
    target_date = target_date or date.today()
    init_db()  # 幂等建表，保证入口函数可直接调用
    logger.info("开始采集 %s 的全国城市旅游热度（%d 城）", target_date, len(CITY_META))

    if _should_use_mock():
        logger.info("进入「模拟数据模式」（DATA_MODE=%s）", DATA_MODE)
        records = _collect_mock(target_date)
    else:
        search_raw = _collect_dimension("搜索", _fetch_search_heat, target_date)
        migration_raw = _collect_dimension("迁徙", _fetch_migration_heat, target_date)
        scenic_raw = _collect_dimension("景区", _fetch_scenic_heat, target_date)

        search = _fill_missing_with_mock(search_raw, "搜索", target_date)
        migration = _fill_missing_with_mock(migration_raw, "迁徙", target_date)
        scenic = _fill_missing_with_mock(scenic_raw, "景区", target_date)

        records = _synthesize_records(target_date, search, migration, scenic)
        _attach_mom_change(records, target_date)
        upsert_city_heat(records)

    logger.info("采集完成：写入 %d 条记录", len(records))
    return records


def run_daily_collect(target_date: Optional[date] = None) -> List[dict]:
    """每日全量采集入口。

    供 APScheduler 定时任务或命令行直接调用，异常不会向外抛出。
    """
    target_date = target_date or date.today()
    start = time.time()
    logger.info("========== 开始每日全量采集 ==========")
    try:
        records = collect_city_heat(target_date)
        logger.info("每日采集完成：%d 条记录，耗时 %.2fs", len(records), time.time() - start)
        return records
    except Exception:  # noqa: BLE001 —— 采集任务兜底，保证调度器不中断
        logger.exception("每日采集任务发生未捕获异常")
        return []


def collect_historical(days: int = 7) -> List[dict]:
    """采集过去 N 天历史数据（用于趋势图）。"""
    all_records: List[dict] = []
    for offset in range(days, 0, -1):
        d = date.today() - timedelta(days=offset)
        all_records.extend(collect_city_heat(d))
    return all_records


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    run_daily_collect()
