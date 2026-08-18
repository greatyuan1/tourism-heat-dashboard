"""AI 大模型服务

封装 DeepSeek 大模型 API 调用（OpenAI 兼容格式），提供三个核心功能：
- generate_daily_overview：每日全国旅游热度综述（≤150字）
- generate_city_analysis：单城市热度解读 + 出行小贴士（约100字）
- generate_travel_recommend：目的地推荐（2-3个 + 简短理由）

所有接口在调用失败或未配置密钥时返回友好的本地兜底文案，保证程序不崩溃。
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"
TIMEOUT = 60

# 系统提示词：口语化、实用的旅游助手语气
SYSTEM_PROMPT = (
    "你是一个接地气的旅游小助手，说话自然实在、不端着、不打官腔，"
    "用普通游客听得懂的大白话分析旅游热度、给出行建议。"
)

# 城市代表性景区（用于充实城市解读与推荐，后续可接入高德 POI 替换）
SCENIC_SPOTS: Dict[str, List[str]] = {
    "北京": ["故宫", "八达岭长城"],
    "上海": ["外滩", "迪士尼乐园"],
    "广州": ["广州塔", "长隆度假区"],
    "深圳": ["世界之窗", "深圳湾公园"],
    "成都": ["宽窄巷子", "大熊猫繁育研究基地"],
    "重庆": ["洪崖洞", "磁器口古镇"],
    "杭州": ["西湖", "灵隐寺"],
    "西安": ["秦始皇兵马俑", "大雁塔"],
    "长沙": ["橘子洲头", "岳麓山"],
    "南京": ["中山陵", "夫子庙"],
    "武汉": ["黄鹤楼", "东湖"],
    "青岛": ["栈桥", "八大关"],
    "三亚": ["亚龙湾", "蜈支洲岛"],
    "昆明": ["石林", "滇池"],
    "厦门": ["鼓浪屿", "环岛路"],
    "哈尔滨": ["中央大街", "冰雪大世界"],
    "桂林": ["漓江", "象鼻山"],
    "丽江": ["丽江古城", "玉龙雪山"],
    "苏州": ["拙政园", "平江路"],
    "天津": ["五大道", "天津之眼"],
    "郑州": ["少林寺", "黄河风景名胜区"],
    "大连": ["星海广场", "老虎滩海洋公园"],
    "贵阳": ["黄果树瀑布", "黔灵山公园"],
    "拉萨": ["布达拉宫", "大昭寺"],
    "乌鲁木齐": ["天山天池", "国际大巴扎"],
    "黄山": ["黄山风景区", "宏村"],
    "张家界": ["张家界国家森林公园", "天门山"],
    "呼和浩特": ["希拉穆仁草原", "大召寺"],
    "银川": ["沙坡头", "西夏王陵"],
    "珠海": ["长隆海洋王国", "情侣路"],
}


def is_configured() -> bool:
    """判断 DeepSeek 是否已配置 API Key。"""
    return bool(DEEPSEEK_API_KEY)


def get_scenic_spots(city: str) -> List[str]:
    """获取城市代表性景区列表（无则返回空列表）。"""
    return SCENIC_SPOTS.get(city, [])


def _chat(messages: List[dict], max_tokens: int = 400, temperature: float = 0.7) -> str:
    """调用 DeepSeek 聊天接口（OpenAI 兼容），返回助手回复文本。"""
    if not is_configured():
        raise RuntimeError("未配置 DEEPSEEK_API_KEY")

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": DEFAULT_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    resp = requests.post(
        f"{DEEPSEEK_BASE_URL}/chat/completions",
        headers=headers,
        json=payload,
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def _ask(user_prompt: str, max_tokens: int) -> Optional[str]:
    """统一调用入口，失败返回 None（由上层兜底）。"""
    try:
        return _chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
        )
    except Exception as e:  # noqa: BLE001 —— 外部 API 失败不应让程序崩溃
        logger.warning("DeepSeek 调用失败：%s", e)
        return None


# ---------------------------------------------------------------------------
# 核心功能一：每日综述
# ---------------------------------------------------------------------------
def generate_daily_overview(hotspot_data: dict) -> str:
    """生成每日全国旅游热度综述（≤150字，通俗易懂）。

    hotspot_data 结构：{"date": str, "avg_mom": float, "top10": [城市热度字典]}
    """
    top10 = hotspot_data.get("top10") or []
    avg_mom = hotspot_data.get("avg_mom") or 0

    if not top10:
        return "今天还没有热度数据，先去采集一波吧～"

    city_line = "、".join(
        f"{c.get('city', '')}(热度{(c.get('heat_index') or 0):.0f})" for c in top10
    )
    rising = [c.get("city", "") for c in top10 if (c.get("mom_change") or 0) > 5][:3]

    user_prompt = (
        f"今天全国旅游热度TOP10：{city_line}。全国整体热度环比{float(avg_mom):+.1f}%。"
        + (f"热度上升明显的城市：{'、'.join(rising)}。" if rising else "")
        + "请用大白话写一段150字以内的每日旅游热度综述，"
          "告诉普通游客今天哪些地方最火、大致什么原因、适合去哪儿，语气轻松自然。"
    )

    text = _ask(user_prompt, max_tokens=300)
    if text:
        return text

    # 本地兜底文案
    top3 = "、".join(c.get("city", "") for c in top10[:3])
    trend_word = "热度整体在上升" if float(avg_mom) > 0 else "热度整体略有回落"
    return (
        f"今天全国最火的是{top3}，{trend_word}。"
        f"想感受热闹可以去{top10[0].get('city', '')}，想清静点不妨错峰出行。"
    )


# ---------------------------------------------------------------------------
# 核心功能二：单城市解读
# ---------------------------------------------------------------------------
def generate_city_analysis(city_name: str, city_data: dict) -> str:
    """生成单城市热度解读 + 出行小贴士（约100字）。

    city_data 结构：{"latest": 最新记录, "trend": 近7天趋势, "scenic": 景区列表}
    """
    trend = city_data.get("trend") or []
    scenic = city_data.get("scenic") or get_scenic_spots(city_name)
    latest = city_data.get("latest") or {}

    heat_now = latest.get("heat_index") or 0
    mom = latest.get("mom_change") or 0

    if trend:
        trend_desc = "→".join(
            f"{str(t.get('date', ''))[-5:]}:{(t.get('heat_index') or 0):.0f}" for t in trend
        )
    else:
        trend_desc = "暂无近7天数据"

    user_prompt = (
        f"城市「{city_name}」当前综合热度{(heat_now):.0f}分（环比{float(mom):+.1f}%）。"
        f"近7天热度变化：{trend_desc}。"
        f"热门景区：{'、'.join(scenic) if scenic else '暂无'}。"
        "请用100字左右写一段该城市的热度解读，再附一句出行小贴士，口语化、实用。"
    )

    text = _ask(user_prompt, max_tokens=250)
    if text:
        return text

    # 本地兜底文案
    trend_word = "热度在回升" if float(mom) > 0 else "热度比较平稳或略有回落"
    tip = f"建议早点订票、错峰游览{'、'.join(scenic)}" if scenic else "建议错峰出行"
    return f"{city_name}目前{trend_word}（热度{(heat_now):.0f}分）。{tip}。"


# ---------------------------------------------------------------------------
# 核心功能三：目的地推荐
# ---------------------------------------------------------------------------
def generate_travel_recommend(user_pref: dict, all_data: List[dict]) -> str:
    """生成2-3个目的地推荐 + 简短理由（≤150字）。

    user_pref 结构：{"origin": 出发地, "days": 天数, "budget": 预算, "preference": 喜好}
    all_data：当前全部城市热度数据列表。
    """
    origin = user_pref.get("origin") or "你所在城市"
    days = user_pref.get("days") or 3
    budget = user_pref.get("budget") or "不限"
    preference = user_pref.get("preference") or "不限"

    hot_cities = (all_data or [])[:10]
    city_line = "、".join(
        f"{c.get('city', '')}(热度{(c.get('heat_index') or 0):.0f})" for c in hot_cities
    )

    user_prompt = (
        f"游客从{origin}出发，打算玩{days}天，预算{budget}，喜欢{preference}。"
        f"当前热门城市：{city_line}。"
        "请结合热度数据推荐2-3个目的地，每个目的地附一句简短实在的理由，"
        "总字数150字以内，直接给结论别啰嗦。"
    )

    text = _ask(user_prompt, max_tokens=350)
    if text:
        return text

    # 本地兜底文案
    if not hot_cities:
        return "暂时没有可推荐的热度数据，稍后再试吧～"
    parts = []
    for c in hot_cities[:3]:
        city = c.get("city", "")
        spots = get_scenic_spots(city)
        spot_line = f"（{'、'.join(spots)}等值得一逛）" if spots else ""
        parts.append(f"{city}：热度{(c.get('heat_index') or 0):.0f}，人气正旺{spot_line}")
    return f"结合你的需求，推荐这几个：{'；'.join(parts)}。"
