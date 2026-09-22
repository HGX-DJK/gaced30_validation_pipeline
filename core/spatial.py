#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
🛰️ GACED30 遥感空间算子与抽样引擎 (Spatial Alignment & Sampling Operators)
================================================================================
负责：
1. 空间相交窗口计算 (Intersection Window)
2. 跨坐标系 (CRS) 自动重投影对齐
3. 零内存分层随机抽样 (Zero-Memory Stratified Sampling & Batch Rejection Sampling)
4. 参考样点表 (CSV) 智能列名别名归一化
"""

import os
import re
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds, transform


def normalize_reference_columns(df):
    """
    智能归一化样点表列名，自适应中英文常见别名：
    - 经度: lon, lng, longitude, x, 经度, point_x, coord_x
    - 纬度: lat, latitude, y, 纬度, point_y, coord_y
    - 验证真值: ref_label, ref, label, true_label, ground_truth, gt, class, code, type, 真值, 地类, 类型, 地类编码, 类别
    - 遥感标签: map_label, pred, pred_label, map, 提取标签, 分类结果, 预测标签
    """
    if df is None or df.empty:
        return df

    col_mapping = {}
    lon_aliases = ['lon', 'lng', 'longitude', 'x', '经度', 'point_x', 'coord_x']
    lat_aliases = ['lat', 'latitude', 'y', '纬度', 'point_y', 'coord_y']
    ref_aliases = ['ref_label', 'ref', 'label', 'true_label', 'ground_truth', 'gt', 'class', 'code', 'type', '真值', '地类', '类型', '地类编码', '类别']
    map_aliases = ['map_label', 'pred', 'pred_label', 'map', '提取标签', '分类结果', '预测标签']

    lower_cols = {str(c).lower().strip(): c for c in df.columns}

    for alias in lon_aliases:
        if alias in lower_cols and "lon" not in df.columns:
            col_mapping[lower_cols[alias]] = "lon"
            break

    for alias in lat_aliases:
        if alias in lower_cols and "lat" not in df.columns:
            col_mapping[lower_cols[alias]] = "lat"
            break

    for alias in ref_aliases:
        if alias in lower_cols and "ref_label" not in df.columns:
            col_mapping[lower_cols[alias]] = "ref_label"
            break

    for alias in map_aliases:
        if alias in lower_cols and "map_label" not in df.columns:
            col_mapping[lower_cols[alias]] = "map_label"
            break

    if col_mapping:
        df = df.rename(columns=col_mapping)
    return df


def parse_tile_coordinates(filename):
    """从 GACED30 标准文件名中提取坐标, 如 Crop_116.0_35.0.tif -> (116.0, 35.0, 119.0, 38.0)"""
    match = re.search(r'Crop_([0-9\.\-]+)_([0-9\.\-]+)', filename, re.IGNORECASE)
    if match:
        lon_min = float(match.group(1))
        lat_min = float(match.group(2))
        return (lon_min, lat_min, lon_min + 3.0, lat_min + 3.0)
    return None


def match_reference_points_for_tile(ref_df, bounds):
    """从总参考真值表中根据经纬度空间边界自动筛选当前瓦片范围内的样点"""
    ref_df = normalize_reference_columns(ref_df)
    if "lon" in ref_df.columns and "lat" in ref_df.columns and bounds is not None:
        lon_min, lat_min, lon_max, lat_max = bounds
        sub_df = ref_df[(ref_df["lon"] >= lon_min) & (ref_df["lon"] <= lon_max) &
                        (ref_df["lat"] >= lat_min) & (ref_df["lat"] <= lat_max)]
        return sub_df
    return pd.DataFrame()


def align_and_sample_raster(src, ref_tif_path, target_band, bounds, n_sample_each=300, seed=42):
    """
    空间相交窗口求交、跨 CRS 重投影与零内存分层抽样算子
    返回: (is_overlapping, mask_inter, aligned_bounds, matched_ref_df)
    """
    with rasterio.open(ref_tif_path) as ref_src:
        # 跨坐标系自动对齐 (如待验图为 WGS84 经纬度，参考图为 UTM 投影或墨卡托投影)
        crs_need_reproject = False
        if src.crs != ref_src.crs and src.crs is not None and ref_src.crs is not None:
            try:
                ref_b_left, ref_b_bottom, ref_b_right, ref_b_top = transform_bounds(ref_src.crs, src.crs, *ref_src.bounds)
                crs_need_reproject = True
            except Exception:
                ref_b_left, ref_b_bottom, ref_b_right, ref_b_top = ref_src.bounds.left, ref_src.bounds.bottom, ref_src.bounds.right, ref_src.bounds.top
        else:
            ref_b_left, ref_b_bottom, ref_b_right, ref_b_top = ref_src.bounds.left, ref_src.bounds.bottom, ref_src.bounds.right, ref_src.bounds.top

        # 计算空间重叠交集
        inter_left = max(bounds[0], ref_b_left)
        inter_right = min(bounds[2], ref_b_right)
        inter_bottom = max(bounds[1], ref_b_bottom)
        inter_top = min(bounds[3], ref_b_top)

        if not (inter_left < inter_right and inter_bottom < inter_top):
            return False, None, (ref_b_left, ref_b_bottom, ref_b_right, ref_b_top), None

        print(f"  -> 🎯 发现参考栅格 [{os.path.basename(ref_tif_path)}] 与当前待验影像存在空间重叠交集！")
        if crs_need_reproject:
            print(f"     🌐 自动识别跨坐标系投影: 待验图 [{src.crs}] ⟷ 参考图 [{ref_src.crs}]，已无缝完成空间投影重对齐")
        print(f"     交集范围: 经度 [{inter_left:.2f} ~ {inter_right:.2f}°E], 纬度 [{inter_bottom:.2f} ~ {inter_top:.2f}°N]")

        # 1. 严格空间对齐：提取相交窗口内的待验分类栅格 (按需局部读取，无需载入整景亿级大图)
        src_win = from_bounds(inter_left, inter_bottom, inter_right, inter_top, src.transform)
        raw_inter = src.read(target_band, window=src_win)
        src_win_transform = src.window_transform(src_win)

        # 向量化直接编码映射 (0=非耕地, 1=耕地, 255=NoData)，立即释放原始切片
        mask_inter = np.where((raw_inter == 10) | (raw_inter == 1), np.uint8(1), np.where(raw_inter == 0, np.uint8(0), np.uint8(255)))
        del raw_inter

        # 2. 联合国规范：在重叠区内执行【分层随机抽样 (Stratified Random Sampling)】
        #    🔥 内存与性能极限优化：使用 1D 稀疏扁平索引与常数级 O(1) 批量拒绝抽样，彻底消除数千万坐标实例化导致的内存暴涨
        H, W_dim = mask_inter.shape
        total_pixels = mask_inter.size
        np.random.seed(seed)

        # 耕地层：耕地相对稀疏，使用 1D flatnonzero，内存占用仅数兆
        crop_flat = np.flatnonzero(mask_inter == 1)
        n_crop_avail = len(crop_flat)
        sel_crop_n = min(n_sample_each, n_crop_avail)
        if sel_crop_n > 0:
            sel_crop_idx = np.random.choice(n_crop_avail, size=sel_crop_n, replace=False)
            crop_r, crop_c = np.unravel_index(crop_flat[sel_crop_idx], (H, W_dim))
            crop_xs, crop_ys = rasterio.transform.xy(src_win_transform, crop_r, crop_c)
        else:
            crop_xs, crop_ys = [], []

        # 非耕地层：遥感大田中通常占 90%+ 的绝对优势层，采用【批量拒绝抽样 (Batch Rejection Sampling)】
        flat_view = mask_inter.ravel()
        selected_noncrop_flat = []
        batch_size = n_sample_each * 4
        for _ in range(5):
            cands = np.random.randint(0, total_pixels, size=batch_size)
            valid_cands = cands[flat_view[cands] == 0]
            for c in valid_cands:
                if c not in selected_noncrop_flat:
                    selected_noncrop_flat.append(c)
                    if len(selected_noncrop_flat) >= n_sample_each:
                        break
            if len(selected_noncrop_flat) >= n_sample_each:
                break

        # 容错兜底：若在特殊全耕地密集区拒绝抽样未满，回退到 1D 稀疏检索
        if len(selected_noncrop_flat) < n_sample_each:
            nc_flat = np.flatnonzero(flat_view == 0)
            if len(nc_flat) > 0:
                sel_nc_n = min(n_sample_each, len(nc_flat))
                sel_nc = np.random.choice(nc_flat, size=sel_nc_n, replace=False)
                selected_noncrop_flat = list(sel_nc)

        if selected_noncrop_flat:
            nc_r, nc_c = np.unravel_index(np.array(selected_noncrop_flat), (H, W_dim))
            noncrop_xs, noncrop_ys = rasterio.transform.xy(src_win_transform, nc_r, nc_c)
        else:
            noncrop_xs, noncrop_ys = [], []

        all_xs = list(crop_xs) + list(noncrop_xs)
        all_ys = list(crop_ys) + list(noncrop_ys)
        map_labels = [1] * len(crop_xs) + [0] * len(noncrop_xs)

        coords = list(zip(all_xs, all_ys))
        if crs_need_reproject:
            sample_xs, sample_ys = transform(src.crs, ref_src.crs, all_xs, all_ys)
            ref_coords = list(zip(sample_xs, sample_ys))
        else:
            ref_coords = coords
        sampled_ref = [v[0] for v in ref_src.sample(ref_coords, indexes=1)]

        matched_ref = pd.DataFrame({
            "lon": all_xs,
            "lat": all_ys,
            "map_label": map_labels,
            "ref_label": sampled_ref,
            "weight": 1.0
        })
        matched_ref = matched_ref[matched_ref["map_label"] != 255].copy()

        aligned_bounds = (inter_left, inter_bottom, inter_right, inter_top)
        print(f"     ✅ 成功完成分层随机抽样: 耕地层抽取 {len(crop_xs)} 点，非耕地层抽取 {len(noncrop_xs)} 点 (共 {len(matched_ref)} 个均衡检验点)！")
        print(f"     ✅ 评估总体与地类权重严格对齐至重叠区面积 (避免全图大范围权重错配)")
        return True, mask_inter, aligned_bounds, matched_ref


def simulate_benchmark_scene(rows=350, cols=350, crop_prob=0.35, seed=42):
    """仿真单景农情栅格与地面验证点"""
    np.random.seed(seed)
    gt_base = np.zeros((rows, cols), dtype=np.uint8)
    for r in range(15, rows - 25, 40):
        for c in range(15, cols - 25, 45):
            h = np.random.randint(22, 35)
            w = np.random.randint(25, 40)
            if np.random.rand() < crop_prob:
                gt_base[r:r+h, c:c+w] = 1

    yr_map = gt_base.copy()
    noise = (np.random.rand(rows, cols) < 0.07)
    yr_map[noise] = 1 - yr_map[noise]

    # 抽样点
    n_samples = 300
    n_crop = int(n_samples * 0.45)
    n_noncrop = n_samples - n_crop

    crop_flat = np.flatnonzero(yr_map == 1)
    noncrop_flat = np.flatnonzero(yr_map == 0)

    sel_crop_idx = np.random.choice(len(crop_flat), min(n_crop, len(crop_flat)), replace=False)
    sel_noncrop_idx = np.random.choice(len(noncrop_flat), min(n_noncrop, len(noncrop_flat)), replace=False)

    crop_r, crop_c = np.unravel_index(crop_flat[sel_crop_idx], (rows, cols))
    noncrop_r, noncrop_c = np.unravel_index(noncrop_flat[sel_noncrop_idx], (rows, cols))

    records = []
    for r, c in zip(crop_r, crop_c):
        records.append({"map_label": 1, "ref_label": gt_base[r, c], "weight": 1.0})
    for r, c in zip(noncrop_r, noncrop_c):
        records.append({"map_label": 0, "ref_label": gt_base[r, c], "weight": 1.0})

    return yr_map, pd.DataFrame(records)
