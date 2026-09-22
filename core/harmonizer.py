#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
🌾 GACED30 参考真值语义与地类编码对齐归一化器 (Semantic Cross-Walking Harmonizer)
================================================================================
遵循联合国《农业统计遥感手册》标准，将第三方各类公信力地表覆被或地面样点语义，
动态归一化对齐为二值体系：1 (耕地) 与 0 (非耕地)。
"""

import numpy as np
import pandas as pd

PRESET_SCHEMES = {
    "fromglc": {
        "name": "FROM-GLC 10m (清华大学全球地表覆被)",
        "crop_values": [10],
        "desc": "10=耕地(Cropland), 20=林地, 30=草地, 40=灌木, 50=湿地, 60=水体, 70=苔原, 80=不透水面, 90=裸地, 100=雪/冰"
    },
    "esa_worldcover": {
        "name": "ESA WorldCover 10m",
        "crop_values": [40],
        "desc": "40=Cropland, 10=Trees, 20=Shrub, 30=Grass, 50=Built, 60=Bare, 80=Water"
    },
    "esri_10m": {
        "name": "ESRI 10m Annual Land Cover",
        "crop_values": [5],
        "desc": "5=Crops, 1=Water, 2=Trees, 7=Built, 8=Bare, 11=Rangeland"
    },
    "globeland30": {
        "name": "GlobeLand30",
        "crop_values": [10],
        "desc": "10=耕地, 20=林地, 30=草地, 50=湿地, 60=水体, 80=人造地表"
    },
    "clcd": {
        "name": "CLCD (中国30米土地覆被)",
        "crop_values": [1],
        "desc": "1=Cropland, 2=Forest, 3=Shrub, 4=Grass, 5=Water, 7=Barren, 8=Impervious"
    },
    "cnlucc": {
        "name": "CNLUCC (中科院土地利用现状分类)",
        "crop_values": [11, 12],
        "desc": "11=水田, 12=旱地, 2x=林地, 3x=草地, 5x=建设用地"
    },
    "worldcereal": {
        "name": "WorldCereal 10m",
        "crop_values": [100, 101, 102],
        "desc": "100=Temporary crops, 101=Summer crops, 102=Winter crops"
    },
    "binary": {
        "name": "标准二值/GACED30",
        "crop_values": [1, 10],
        "desc": "1 或 10=耕地, 0=非耕地"
    }
}

CROP_KEYWORDS = [
    "耕地", "水田", "旱地", "水浇地", "农田", "大棚", "温室", "休耕",
    "小麦", "水稻", "玉米", "大豆", "油菜", "花生", "棉花", "蔬菜",
    "crop", "agriculture", "cultivated", "arable", "paddy", "wheat", "rice",
    "corn", "maize", "soy", "fallow", "greenhouse"
]

NONCROP_KEYWORDS = [
    "林地", "森林", "灌木", "草地", "水体", "河流", "湖泊", "水库", "海洋",
    "湿地", "建设用地", "城镇", "农村居民点", "工业", "道路", "裸地", "荒漠",
    "沙漠", "冰川", "雪", "非耕地", "forest", "tree", "grass", "shrub",
    "water", "urban", "built", "bare", "wetland", "snow", "ice", "non-crop", "noncrop"
]


def harmonize_reference_labels(series, preset="auto", custom_crop_values=None):
    """
    联合国手册规范：参考真值语义与分类编码对齐归一化器 (Semantic Cross-Walking Harmonizer)
    将任意第三方公信力真值 (ESA WorldCover, ESRI, GlobeLand30, CLCD, CNLUCC, 地面调查文字)
    归一化转换为联合国标准的 1 (耕地) 与 0 (非耕地)
    """
    if custom_crop_values:
        crop_vals = set(int(x) if str(x).isdigit() else x for x in custom_crop_values)
        info = f"自定义用户规则 (指定耕地编码: {sorted(list(crop_vals))})"
        mapped = np.array([1 if x in crop_vals else 0 for x in series], dtype=np.int64)
        return mapped, info

    # 判断是否为中英文本描述 (如 '水田', '林地', 'cropland')
    is_text = False
    for v in series[:min(20, len(series))]:
        if isinstance(v, str) and not v.strip().replace('.', '').isdigit():
            is_text = True
            break

    if is_text:
        res = []
        for val in series:
            s = str(val).lower().strip()
            is_non = any(nk in s for nk in NONCROP_KEYWORDS)
            is_crp = any(ck in s for ck in CROP_KEYWORDS)
            if is_crp and not is_non:
                res.append(1)
            elif is_non:
                res.append(0)
            elif is_crp:
                res.append(1)
            else:
                res.append(0)
        return np.array(res, dtype=np.int64), "智能中/英文文本语义解析 (水田/旱地/玉米/大棚->1, 林地/草地/水体/建设用地->0)"

    # 数值类别编码映射
    try:
        numeric_vals = [int(float(x)) for x in series if pd.notnull(x)]
        unique_vals = set(numeric_vals)
    except Exception:
        unique_vals = set()

    if preset != "auto" and preset in PRESET_SCHEMES:
        sch = PRESET_SCHEMES[preset]
        crop_vals = set(sch["crop_values"])
        mapped = np.array([1 if int(float(x)) in crop_vals else 0 for x in series], dtype=np.int64)
        return mapped, f"按指定预设体系 [{sch['name']}] 映射: {crop_vals}->耕地(1), 其余->非耕地(0)"

    # 自动识别 FROM-GLC 10m 或 GlobeLand30 (以 10 为耕地，20/30/80/90 等为非耕地)
    if (10 in unique_vals and (90 in unique_vals or 80 in unique_vals or 30 in unique_vals)) and (40 not in unique_vals or 90 in unique_vals):
        sch = PRESET_SCHEMES["fromglc"]
        crop_vals = set(sch["crop_values"])
        mapped = np.array([1 if int(float(x)) in crop_vals else 0 for x in series], dtype=np.int64)
        return mapped, f"自动识别为【{sch['name']}】: 像元编码 10->耕地(1), 其余(20/30/80/90...)->非耕地(0)"

    # 自动识别 ESA WorldCover 10m (以 40 为耕地，其它为 10, 20, 30, 50, 60, 80... 不含90裸地)
    if 40 in unique_vals and 90 not in unique_vals and (10 in unique_vals or 20 in unique_vals or 30 in unique_vals or 50 in unique_vals or 80 in unique_vals):
        sch = PRESET_SCHEMES["esa_worldcover"]
        crop_vals = set(sch["crop_values"])
        mapped = np.array([1 if int(float(x)) in crop_vals else 0 for x in series], dtype=np.int64)
        return mapped, f"自动识别为【{sch['name']}】: 像元编码 40->耕地(1), 其余(10/20/30/50...)->非耕地(0)"

    # 自动识别 CNLUCC (中科院土地利用分类，水田11，旱地12，林草水建为2x, 3x, 5x...)
    if (11 in unique_vals or 12 in unique_vals) and any(v >= 20 for v in unique_vals):
        sch = PRESET_SCHEMES["cnlucc"]
        crop_vals = set(sch["crop_values"])
        mapped = np.array([1 if int(float(x)) in crop_vals else 0 for x in series], dtype=np.int64)
        return mapped, f"自动识别为【{sch['name']}】: 编码 11/12(水田/旱地)->耕地(1), 其余->非耕地(0)"

    # 自动识别 ESRI 10m (5 为作物，其它为 1, 2, 7, 8, 11...)
    if 5 in unique_vals and (1 in unique_vals or 2 in unique_vals or 7 in unique_vals or 8 in unique_vals or 11 in unique_vals):
        sch = PRESET_SCHEMES["esri_10m"]
        crop_vals = set(sch["crop_values"])
        mapped = np.array([1 if int(float(x)) in crop_vals else 0 for x in series], dtype=np.int64)
        return mapped, f"自动识别为【{sch['name']}】: 编码 5->耕地(1), 其余->非耕地(0)"

    # 默认标准二值 / GACED30 体系 (1 或 10 为耕地, 0 为非耕地)
    mapped = np.array([1 if int(float(x)) in [1, 10] else 0 for x in series], dtype=np.int64)
    return mapped, "标准二值体系: 1 或 10->耕地(1), 0->非耕地(0)"
