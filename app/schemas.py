"""数据结构定义（Pydantic 模型）

统一前后端数据契约，用于 API 请求/响应的校验与序列化。
"""
from __future__ import annotations

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field


class CityHeat(BaseModel):
    """每日城市热度记录"""

    date: date
    city: str = Field(..., description="城市名称")
    province: str = Field(..., description="省份")
    heat_index: float = Field(..., description="综合热度指数")
    search_heat: float = Field(..., description="搜索热度")
    migration_heat: float = Field(..., description="迁徙热度")
    scenic_heat: float = Field(..., description="景区热度")
    mom_change: Optional[float] = Field(None, description="环比变化（%）")

    model_config = {"from_attributes": True}


class RecommendRequest(BaseModel):
    """AI 目的地推荐请求参数"""

    origin: str = Field("", description="出发地")
    days: int = Field(3, ge=1, le=30, description="游玩天数")
    budget: str = Field("", description="预算（如：经济/中等/充裕）")
    preference: str = Field("", description="喜好（如：自然风光/美食/历史文化/亲子）")
