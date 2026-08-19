"""AI 大模型服务

封装 DeepSeek 大模型 API 调用（OpenAI 兼容格式），提供两个核心功能：
- generate_daily_overview：上周全国旅游热度运营分析（结构化4段）
- generate_city_analysis：单城市热度解读 + 出行小贴士（约100字）

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

# 系统提示词：客观、专业的旅游行业数据分析师语气
SYSTEM_PROMPT = (
    "你是一名专业的旅游行业数据分析师，擅长从城市旅游热度数据中提炼客观、专业的洞察。"
    "表达准确、条理清晰，避免口语化与网络化表述。"
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
    """生成上周全国旅游热度运营分析（结构化4段，面向运营决策）。

    输出格式：4个段落，每段以加粗小标题 **【整体概览】** / **【梯队格局】** /
    **【区域特征】** / **【运营建议】** 开头，前端按 Markdown 渲染为加粗标题。
    【运营建议】分「营销投放 / 供给调度 / 用户运营」三个场景给出具体动作指引。
    单周无历史数据时（has_history=False）：不提供环比数值、禁止编造「0.0%」，
    并在【整体概览】末尾补充「当前为单周基准数据…」的严谨说明。
    hotspot_data 结构：{"date": str, "avg_mom": float, "top10": [城市热度字典], "has_history": bool}
    """
    top10 = hotspot_data.get("top10") or []
    avg_mom = hotspot_data.get("avg_mom") or 0
    has_history = bool(hotspot_data.get("has_history", True))

    if not top10:
        return "上周暂无热度数据，先去采集一波吧～"

    city_line = "、".join(
        f"{c.get('city', '')}(热度{(c.get('heat_index') or 0):.0f})" for c in top10
    )
    rising = [c.get("city", "") for c in top10 if (c.get("mom_change") or 0) > 0][:3]

    # 环比相关数据行：无历史数据时不提供环比数值，避免大模型把「无法计算」误写成「0.0%」
    if has_history:
        mom_line = f"全国整体热度环比：{float(avg_mom):+.1f}%。"
        rising_line = f"热度环比上升的城市：{'、'.join(rising) if rising else '无'}。"
        overview_extra = ""
        hard_rule = "客观专业、条理清晰、直接给结论，避免口语化与网络化表达。"
    else:
        mom_line = "全国整体热度环比：暂无历史基准数据（当前仅单周，尚无法计算环比）。"
        rising_line = "热度环比上升的城市：暂无（需累计两周数据后方可计算）。"
        overview_extra = "该段末尾必须补一句严谨说明：当前为单周基准数据，周环比变化指标将在累计两周数据后自动更新。"
        hard_rule = (
            "当数据标注为「暂无历史基准数据」时，严禁输出「环比0.0%」「环比持平」「环比无变化」等错误表述，"
            "也不得自行编造任何环比百分比；其余保持客观专业、条理清晰、直接给结论，避免口语化与网络化表达。"
        )

    user_prompt = (
        f"请基于以下数据撰写上周全国旅游热度运营分析，严格分为4个段落，"
        f"每段开头使用加粗小标题（格式：**小标题**）：\n"
        f"TOP10城市热度：{city_line}。\n"
        f"{mom_line}\n"
        f"{rising_line}\n"
        f"段落内容要求（面向文旅运营决策，从数据描述转向业务分析）：\n"
        f"1. **【整体概览】**：上周全国旅游热度整体水位与走势基调，并给出业务定性（如高位运行、温和增长、结构性分化）；{overview_extra}\n"
        f"2. **【梯队格局】**：头部城市排名情况与梯队分层特征，补充1句业务归因（如头部城市受暑期亲子游、文博游需求驱动）；\n"
        f"3. **【区域特征】**：区域分布差异与城市群表现，补充城市群运营价值判断（如长三角城市群协同运营价值高）；\n"
        f"4. **【运营建议】**：换行后完整输出三个场景，每个场景单独一行、以「【营销投放】」「【供给调度】」「【用户运营】」开头，各给出1条具体可执行的行动指引，不得省略或合并：\n"
        f"   【营销投放】建议重点投入的城市与理由；\n"
        f"   【供给调度】高热度城市的承载预警与优化方向；\n"
        f"   【用户运营】错峰引导建议与体验平衡策略。\n"
        f"硬性要求：{hard_rule}；语言精炼简洁，每段控制在1-2句话，全篇不超过250字；三个场景行前不得添加任何序号、圆点或符号前缀；【运营建议】三个场景必须全部输出、内容完整、不得截断。"
    )

    text = _ask(user_prompt, max_tokens=500)
    if text:
        return text

    # 本地兜底文案（同样结构化4段、带运营建议，与前端 Markdown 渲染兼容）
    top3 = "、".join(c.get("city", "") for c in top10[:3])
    if has_history:
        trend_word = "整体上升" if float(avg_mom) > 0 else "整体回落"
        overview_line = f"上周全国旅游热度{trend_word}，热度主要集中在{top3}等热门旅游城市。"
    else:
        overview_line = (
            f"上周全国旅游热度主要集中在{top3}等热门旅游城市。"
            "当前为单周基准数据，周环比变化指标将在累计两周数据后自动更新。"
        )
    return (
        f"**【整体概览】** {overview_line}\n"
        f"**【梯队格局】** 综合热度最高的是{top10[0].get('city', '')}，其余城市热度呈梯队分布，头部城市受暑期亲子游、文博游等需求驱动。\n"
        f"**【区域特征】** 热门城市覆盖多个区域，重点旅游城市群表现活跃，城市群协同运营价值较高。\n"
        f"**【运营建议】**\n【营销投放】建议重点投入热度上升城市，加大内容种草。\n【供给调度】对高热度城市提前预警景区承载量，优化住宿交通供给。\n【用户运营】引导游客错峰出行，平衡高峰体验与接待能力。"
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
        f"请针对城市「{city_name}」撰写一段100字以内的热度解读与出行建议：\n"
        f"当前综合热度{(heat_now):.0f}（环比{float(mom):+.1f}%）。\n"
        f"近7天热度变化：{trend_desc}。\n"
        f"代表性景区：{'、'.join(scenic) if scenic else '暂无'}。\n"
        f"要求：简洁实用，聚焦热度趋势与游玩建议，语气平实客观。"
    )

    text = _ask(user_prompt, max_tokens=250)
    if text:
        return text

    # 本地兜底文案
    trend_word = "热度呈上升趋势" if float(mom) > 0 else "热度较为平稳或有所回落"
    tip = f"建议提前预订、错峰游览{'、'.join(scenic)}" if scenic else "建议错峰出行"
    return f"{city_name}当前综合热度{(heat_now):.0f}，{trend_word}。{tip}。"
