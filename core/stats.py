#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
🌾 GACED30 联合国标准量化评判与无偏统计引擎 (Statistical & Accuracy Assessment Core)
================================================================================
理论标准对齐：
    严格遵循联合国粮农组织 (FAO) 与联合国统计司 (UNSD) 联合专著：
    《联合国农业统计遥感手册》（UN Handbook on Remote Sensing for Agricultural Statistics）
    - 第 8 章：Map Validation and Use of Maps for Area Estimation (Olofsson et al., 2014)
    - 第 22-24 章：Weighted Area Estimator 与多区域/多图层分层联合推断
    - Cochran (1977) 分层随机抽样理论最小样本量测算
"""

import os
import numpy as np
import pandas as pd

from .harmonizer import harmonize_reference_labels
from .report import generate_comparison_chart, generate_html_dashboard


class GACED30Validator:
    """GACED30 联合国标准量化评判与无偏统计引擎 (支持多景多区域)"""

    def __init__(self, pixel_res_m=30.0, confidence_level=0.95, n_bootstrap=2000, ref_preset="auto", custom_crop_values=None):
        self.pixel_res = float(pixel_res_m)
        self.pixel_area_m2 = self.pixel_res * self.pixel_res
        self.confidence_level = float(confidence_level)
        self.z_score = 1.96 if abs(confidence_level - 0.95) < 1e-4 else 2.576
        self.n_bootstrap = int(n_bootstrap)
        self.ref_preset = ref_preset
        self.custom_crop_values = custom_crop_values

    def calculate_cochran_sample_size(self, w_cropland, expected_ua_crop=0.85, expected_ua_noncrop=0.90, target_se=0.01):
        """Cochran (1977) 分层随机抽样最小检验样本量公式"""
        w_noncrop = 1.0 - w_cropland
        s_crop = np.sqrt(expected_ua_crop * (1.0 - expected_ua_crop))
        s_noncrop = np.sqrt(expected_ua_noncrop * (1.0 - expected_ua_noncrop))
        numerator = (w_cropland * s_crop) + (w_noncrop * s_noncrop)
        return int(np.ceil((numerator / target_se) ** 2))

    def evaluate_single_scene(self, scene_id, map_binary_mask, ref_binary_samples, output_scene_dir, year=2024):
        """对单景耕地二值图进行联合国标准量化评估"""
        os.makedirs(output_scene_dir, exist_ok=True)

        # GACED30 官方语义规范: 10=耕地, 0=非耕地, 255=NoData; 兼容处理 1=耕地
        # 使用 np.count_nonzero 直接在原始掩膜上快速统计，彻底避免创建多份千万级元素 boolean 临时数组
        crop_pixels = int(np.count_nonzero((map_binary_mask == 1) | (map_binary_mask == 10)))
        noncrop_pixels = int(np.count_nonzero(map_binary_mask == 0))
        total_valid_pixels = crop_pixels + noncrop_pixels

        if total_valid_pixels == 0:
            raise ValueError(f"景 [{scene_id}] 有效像元总数为 0，请检查输入栅格。")

        W_crop = float(crop_pixels) / float(total_valid_pixels)
        W_noncrop = float(noncrop_pixels) / float(total_valid_pixels)
        W = np.array([W_noncrop, W_crop], dtype=np.float64)

        total_area_m2 = total_valid_pixels * self.pixel_area_m2
        total_area_km2 = total_area_m2 / 1e6
        total_area_mu = total_area_m2 * 0.0015

        naive_crop_km2 = (crop_pixels * self.pixel_area_m2) / 1e6
        naive_crop_mu = (crop_pixels * self.pixel_area_m2) * 0.0015

        # 标准化标签编码: 10 或 1 映射为 1 (耕地), 0 映射为 0 (非耕地)
        raw_map = ref_binary_samples["map_label"].values
        y_map = np.where((raw_map == 1) | (raw_map == 10), 1, 0)

        # 智能参考真值语义与编码对齐转换 (支持 ESA 40, ESRI 5, CNLUCC 11/12, 文本描述等)
        raw_ref = ref_binary_samples["ref_label"].values
        y_true, harmonizer_desc = harmonize_reference_labels(raw_ref, preset=self.ref_preset, custom_crop_values=self.custom_crop_values)
        weights = ref_binary_samples["weight"].values if "weight" in ref_binary_samples.columns else np.ones(len(y_map), dtype=np.float64)

        n_matrix = np.zeros((2, 2), dtype=np.float64)
        for i in [0, 1]:
            for j in [0, 1]:
                mask_ij = (y_map == i) & (y_true == j)
                n_matrix[i, j] = np.sum(weights[mask_ij])

        n_i_dot = np.sum(n_matrix, axis=1)
        n_samples_total = np.sum(n_matrix)

        cond_matrix = np.zeros((2, 2), dtype=np.float64)
        for i in [0, 1]:
            if n_i_dot[i] > 0:
                cond_matrix[i, :] = n_matrix[i, :] / n_i_dot[i]
            else:
                cond_matrix[i, i] = 1.0

        p_matrix = np.zeros((2, 2), dtype=np.float64)
        for i in [0, 1]:
            p_matrix[i, :] = W[i] * cond_matrix[i, :]

        OA = float(np.sum(np.diag(p_matrix)))
        var_oa = np.sum([ (W[i]**2) * (cond_matrix[i, i] * (1.0 - cond_matrix[i, i])) / (n_i_dot[i] - 1.0) if n_i_dot[i] > 1 else 0.0 for i in [0, 1] ])
        SE_OA = float(np.sqrt(max(0.0, var_oa)))

        UA_crop = cond_matrix[1, 1]
        UA_noncrop = cond_matrix[0, 0]
        SE_UA_crop = np.sqrt(UA_crop * (1.0 - UA_crop) / (n_i_dot[1] - 1.0)) if n_i_dot[1] > 1 else 0.0
        Commission_crop = 1.0 - UA_crop

        p_dot_crop = np.sum(p_matrix[:, 1])
        p_dot_noncrop = np.sum(p_matrix[:, 0])

        PA_crop = p_matrix[1, 1] / p_dot_crop if p_dot_crop > 0 else 0.0
        PA_noncrop = p_matrix[0, 0] / p_dot_noncrop if p_dot_crop > 0 else 0.0
        Omission_crop = 1.0 - PA_crop

        var_pa_crop = 0.0
        if n_i_dot[1] > 1 and p_dot_crop > 0:
            term1 = (W[1]**2) * ((1.0 - PA_crop)**2) * (UA_crop * (1.0 - UA_crop)) / (n_i_dot[1] - 1.0)
            term2 = (W[0]**2) * (cond_matrix[0, 1] * (1.0 - cond_matrix[0, 1])) / (n_i_dot[0] - 1.0) if n_i_dot[0] > 1 else 0.0
            var_pa_crop = (term1 + (PA_crop**2) * term2) / (p_dot_crop**2)
        SE_PA_crop = float(np.sqrt(max(0.0, var_pa_crop)))

        F1_crop = (2.0 * UA_crop * PA_crop) / (UA_crop + PA_crop) if (UA_crop + PA_crop) > 0 else 0.0
        IoU_crop = p_matrix[1, 1] / (p_matrix[1, 1] + p_matrix[1, 0] + p_matrix[0, 1]) if (p_matrix[1, 1] + p_matrix[1, 0] + p_matrix[0, 1]) > 0 else 0.0

        calibrated_crop_km2 = p_dot_crop * total_area_km2
        calibrated_crop_mu = p_dot_crop * total_area_mu

        var_p_crop = 0.0
        for i in [0, 1]:
            if n_i_dot[i] > 1:
                pi_term = cond_matrix[i, 1] * (1.0 - cond_matrix[i, 1])
                var_p_crop += (W[i]**2) * (pi_term / (n_i_dot[i] - 1.0))

        SE_p_crop = float(np.sqrt(max(0.0, var_p_crop)))
        SE_area_km2 = SE_p_crop * total_area_km2
        SE_area_mu = SE_p_crop * total_area_mu
        var_area_km2 = (SE_area_km2) ** 2

        CV_percent = (SE_area_km2 / calibrated_crop_km2) * 100.0 if calibrated_crop_km2 > 0 else 0.0
        bias_km2 = naive_crop_km2 - calibrated_crop_km2
        bias_percent = (bias_km2 / calibrated_crop_km2) * 100.0 if calibrated_crop_km2 > 0 else 0.0

        ci_lower_km2 = max(0.0, calibrated_crop_km2 - self.z_score * SE_area_km2)
        ci_upper_km2 = calibrated_crop_km2 + self.z_score * SE_area_km2

        if CV_percent < 5.0:
            defensibility_badge = "⭐⭐⭐ 官方发布标准 (A级-极优)"
        elif CV_percent <= 10.0:
            defensibility_badge = "⭐⭐ 省级监测可用 (B级-良好)"
        else:
            defensibility_badge = "⚠️ 抽样变异较大 (C级-需补样)"

        # 导出单景 CSV (严格以耕地自身专项指标为核心，彻底剔除易产生假象的全图 OA)
        df_acc = pd.DataFrame([
            {"指标类别": "核心耕地", "指标名称": "耕地综合总体精度 (F1-Score)", "量化数值": f"{F1_crop*100:.2f}%", "标准差 (SE)": "-", "合格门槛与规范说明": "推荐 ≥ 85.0% (纯耕地综合质检核心)"},
            {"指标类别": "核心耕地", "指标名称": "耕地用户精度 (UA / 查准率)", "量化数值": f"{UA_crop*100:.2f}%", "标准差 (SE)": f"±{SE_UA_crop*100:.2f}%", "合格门槛与规范说明": f"虚报/错报率 = {Commission_crop*100:.2f}% (重点核心指标)"},
            {"指标类别": "核心耕地", "指标名称": "耕地生产者精度 (PA / 查全率)", "量化数值": f"{PA_crop*100:.2f}%", "标准差 (SE)": f"±{SE_PA_crop*100:.2f}%", "合格门槛与规范说明": f"漏报/欠报率 = {Omission_crop*100:.2f}% (重点核心指标)"},
            {"指标类别": "几何重合", "指标名称": "耕地空间交并比 (IoU)", "量化数值": f"{IoU_crop*100:.2f}%", "标准差 (SE)": "-", "合格门槛与规范说明": "小农破碎带优良线 75%~85%"},
            {"指标类别": "面积校准", "指标名称": "直接数像元偏差率 (Bias%)", "量化数值": f"{bias_percent:+.2f}%", "标准差 (SE)": "-", "合格门槛与规范说明": "直接数像素面积与联合国无偏面积的偏离度"}
        ])
        df_acc.to_csv(os.path.join(output_scene_dir, "accuracy_metrics.csv"), index=False, encoding="utf-8-sig")

        df_conf = pd.DataFrame(
            data=[
                [f"{p_matrix[0, 0]:.4f} ({int(n_matrix[0, 0])})", f"{p_matrix[0, 1]:.4f} ({int(n_matrix[0, 1])})", f"{W_noncrop:.4f}", f"{UA_noncrop*100:.2f}%"],
                [f"{p_matrix[1, 0]:.4f} ({int(n_matrix[1, 0])})", f"{p_matrix[1, 1]:.4f} ({int(n_matrix[1, 1])})", f"{W_crop:.4f}", f"{UA_crop*100:.2f}%"],
                [f"{p_dot_noncrop:.4f}", f"{p_dot_crop:.4f}", "1.0000", f"OA={OA*100:.2f}%"],
                [f"{PA_noncrop*100:.2f}%", f"{PA_crop*100:.2f}%", "-", f"F1={F1_crop:.4f}"]
            ],
            index=["地图: 非耕地 (0)", "地图: 耕地 (1)", "无偏面积比例", "生产者精度 (PA)"],
            columns=["真实: 非耕地 (0)", "真实: 耕地 (1)", "面积权重 (Wi)", "用户精度 (UA)"]
        )
        df_conf.to_csv(os.path.join(output_scene_dir, "area_weighted_confusion_matrix.csv"), encoding="utf-8-sig")

        return {
            "scene_id": scene_id,
            "total_area_km2": total_area_km2,
            "naive_crop_km2": naive_crop_km2,
            "calibrated_crop_km2": calibrated_crop_km2,
            "bias_km2": bias_km2,
            "bias_percent": bias_percent,
            "SE_area_km2": SE_area_km2,
            "var_area_km2": var_area_km2,
            "CV_percent": CV_percent,
            "ci_km2": (ci_lower_km2, ci_upper_km2),
            "OA": OA, "SE_OA": SE_OA,
            "UA_crop": UA_crop, "SE_UA_crop": SE_UA_crop,
            "PA_crop": PA_crop, "SE_PA_crop": SE_PA_crop,
            "F1_crop": F1_crop, "IoU_crop": IoU_crop,
            "n_samples": n_samples_total,
            "defensibility_badge": defensibility_badge,
            "p_matrix": p_matrix, "n_matrix": n_matrix,
            "year": year
        }

    def evaluate_multi_scenes(self, scene_list, output_base_dir="output", year=2024):
        """
        对多景不同区域瓦片执行批量验证，并按照联合国手册进行全区联合无偏推断
        """
        os.makedirs(output_base_dir, exist_ok=True)
        scenes_dir = os.path.join(output_base_dir, "scenes")
        os.makedirs(scenes_dir, exist_ok=True)

        n_scenes = len(scene_list)
        print(f"\n{'='*80}")
        print(f"🌾 【联合国标准】启动多景多区域遥感耕地批量验证 (共检测到 {n_scenes} 景待验数据)")
        print(f"{'='*80}")

        scene_results = []
        for idx, sc in enumerate(scene_list):
            sid = sc["scene_id"]
            mask = sc["mask"]
            ref_df = sc["ref_samples"]
            bounds = sc.get("bounds", "N/A")
            out_s_dir = os.path.join(scenes_dir, sid)

            print(f"\n[{idx+1}/{n_scenes}] 正在处理区域/瓦片: {sid} (范围: {bounds})...")
            res = self.evaluate_single_scene(sid, mask, ref_df, out_s_dir, year=year)
            sc["mask"] = None  # 及时释放单景大栅格内存，支持百景大批量处理防爆内存
            res["bounds"] = bounds
            res["multi_cube"] = sc.get("multi_cube", None)
            scene_results.append(res)
            print(f"   -> 面积: 待验数像元={res['naive_crop_km2']:,.1f} km² | 联合国无偏={res['calibrated_crop_km2']:,.1f} km² (SE: ±{res['SE_area_km2']:.1f} km², CV: {res['CV_percent']:.2f}%)")
            print(f"   -> 耕地精度: F1={res['F1_crop']*100:.2f}%, UA(查准)={res['UA_crop']*100:.2f}%, PA(查全)={res['PA_crop']*100:.2f}%, IoU={res['IoU_crop']*100:.2f}%")

        # -----------------------------------------------------------------------------------------
        # 联合国手册第 22-24 章：跨区域/多景分层联合无偏推断 (Stratified Cross-Domain Aggregation)
        # -----------------------------------------------------------------------------------------
        grand_total_area_km2 = sum(r["total_area_km2"] for r in scene_results)
        grand_naive_crop_km2 = sum(r["naive_crop_km2"] for r in scene_results)
        grand_calibrated_crop_km2 = sum(r["calibrated_crop_km2"] for r in scene_results)
        
        # 跨景独立分层解析方差与标准误: Var_tot = sum(Var_d), SE_tot = sqrt(Var_tot)
        grand_var_area_km2 = sum(r["var_area_km2"] for r in scene_results)
        grand_se_area_km2 = np.sqrt(grand_var_area_km2)
        grand_cv_percent = (grand_se_area_km2 / grand_calibrated_crop_km2) * 100.0 if grand_calibrated_crop_km2 > 0 else 0.0

        grand_bias_km2 = grand_naive_crop_km2 - grand_calibrated_crop_km2
        grand_bias_percent = (grand_bias_km2 / grand_calibrated_crop_km2) * 100.0 if grand_calibrated_crop_km2 > 0 else 0.0

        # 面积加权全域综合制图精度
        grand_oa = sum(r["OA"] * (r["total_area_km2"] / grand_total_area_km2) for r in scene_results)
        grand_ua = sum(r["UA_crop"] * (r["naive_crop_km2"] / grand_naive_crop_km2) for r in scene_results) if grand_naive_crop_km2 > 0 else 0.0
        grand_pa = sum(r["PA_crop"] * (r["calibrated_crop_km2"] / grand_calibrated_crop_km2) for r in scene_results) if grand_calibrated_crop_km2 > 0 else 0.0
        grand_f1 = (2.0 * grand_ua * grand_pa) / (grand_ua + grand_pa) if (grand_ua + grand_pa) > 0 else 0.0

        grand_ci_lower_km2 = max(0.0, grand_calibrated_crop_km2 - self.z_score * grand_se_area_km2)
        grand_ci_upper_km2 = grand_calibrated_crop_km2 + self.z_score * grand_se_area_km2

        if grand_cv_percent < 5.0:
            grand_badge = "⭐⭐⭐ 联合官方统计发布标准 (A级-极优)"
        elif grand_cv_percent <= 10.0:
            grand_badge = "⭐⭐ 跨区联合监测可用 (B级-良好)"
        else:
            grand_badge = "⚠️ 抽样变异较大 (C级-需增加样方)"

        # 构造多景汇总对比台账 (CSV) - 严格以耕地专项精度为绝对核心
        summary_rows = []
        for r in scene_results:
            summary_rows.append({
                "区域/瓦片标识 (Scene ID)": r["scene_id"],
                "地理空间范围 (Bounds)": str(r["bounds"]),
                "有效校验幅员 (km²)": round(r["total_area_km2"], 2),
                "待验数像元面积 (km²)": round(r["naive_crop_km2"], 2),
                "联合国无偏面积 (km²)": round(r["calibrated_crop_km2"], 2),
                "无偏面积(万亩)": round(r["calibrated_crop_km2"] * 0.15, 2),
                "直接数像元偏差率": f"{r['bias_percent']:+.2f}%",
                "解析标准误 (SE km²)": f"±{r['SE_area_km2']:.2f}",
                "变异系数 (CV%)": f"{r['CV_percent']:.2f}%",
                "95% 置信区间 (km²)": f"[{r['ci_km2'][0]:.1f} ~ {r['ci_km2'][1]:.1f}]",
                "耕地综合总体精度 (F1)": f"{r['F1_crop']*100:.2f}%",
                "耕地查准率 (UA)": f"{r['UA_crop']*100:.2f}%",
                "耕地查全率 (PA)": f"{r['PA_crop']*100:.2f}%",
                "耕地交并比 (IoU)": f"{r['IoU_crop']:.4f}",
                "检验样点数": int(r["n_samples"]),
                "官方采信评级": r["defensibility_badge"]
            })

        # 添加全区联合汇总行
        summary_rows.append({
            "区域/瓦片标识 (Scene ID)": "【跨区全景联合汇总 Grand Total】",
            "地理空间范围 (Bounds)": f"涵盖 {n_scenes} 个独立农区瓦片",
            "有效校验幅员 (km²)": round(grand_total_area_km2, 2),
            "待验数像元面积 (km²)": round(grand_naive_crop_km2, 2),
            "联合国无偏面积 (km²)": round(grand_calibrated_crop_km2, 2),
            "无偏面积(万亩)": round(grand_calibrated_crop_km2 * 0.15, 2),
            "直接数像元偏差率": f"{grand_bias_percent:+.2f}%",
            "解析标准误 (SE km²)": f"±{grand_se_area_km2:.2f}",
            "变异系数 (CV%)": f"{grand_cv_percent:.2f}%",
            "95% 置信区间 (km²)": f"[{grand_ci_lower_km2:.1f} ~ {grand_ci_upper_km2:.1f}]",
            "耕地综合总体精度 (F1)": f"{grand_f1*100:.2f}%",
            "耕地查准率 (UA)": f"{grand_ua*100:.2f}%",
            "耕地查全率 (PA)": f"{grand_pa*100:.2f}%",
            "耕地交并比 (IoU)": "-",
            "检验样点数": sum(int(r["n_samples"]) for r in scene_results),
            "官方采信评级": grand_badge
        })

        df_summary = pd.DataFrame(summary_rows)
        summary_csv_path = os.path.join(output_base_dir, "gaced30_multi_scene_summary.csv")
        df_summary.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")

        # 生成多景对比图表与高管专报
        plot_path = os.path.join(output_base_dir, "gaced30_multi_scene_comparison.png")
        html_path = os.path.join(output_base_dir, "gaced30_multi_scene_dashboard.html")
        generate_comparison_chart(scene_results, grand_summary={
            "grand_naive": grand_naive_crop_km2, "grand_calibrated": grand_calibrated_crop_km2,
            "grand_se": grand_se_area_km2, "grand_cv": grand_cv_percent, "grand_badge": grand_badge,
            "grand_oa": grand_oa, "grand_ua": grand_ua, "grand_pa": grand_pa, "grand_f1": grand_f1
        }, output_plot=plot_path, z_score=self.z_score)

        generate_html_dashboard(df_summary, grand_summary={
            "grand_total_area": grand_total_area_km2, "grand_naive": grand_naive_crop_km2,
            "grand_calibrated": grand_calibrated_crop_km2, "grand_bias_pct": grand_bias_percent,
            "grand_se": grand_se_area_km2, "grand_cv": grand_cv_percent, "grand_badge": grand_badge,
            "grand_oa": grand_oa, "grand_ua": grand_ua, "grand_pa": grand_pa, "grand_f1": grand_f1,
            "n_scenes": n_scenes
        }, output_html=html_path)

        # 打印控制台总结
        print(f"\n{'='*80}")
        print(f"📊 【联合国手册】多景联合无偏面积与精度汇总报告")
        print(f"{'='*80}")
        print(f"全景联合评级: {grand_badge}")
        print(f"   • 跨区有效总幅员       : {grand_total_area_km2:,.2f} km²")
        print(f"   • 待验地图像元计数面积 : {grand_naive_crop_km2:,.2f} km² ({grand_naive_crop_km2*0.15:,.2f} 万亩)")
        print(f"   • 联合国联合无偏面积   : {grand_calibrated_crop_km2:,.2f} km² ({grand_calibrated_crop_km2*0.15:,.2f} 万亩)")
        print(f"   • 跨区直接数像元偏差率 : {grand_bias_percent:+.2f}%")
        print(f"   • 联合解析标准误 (SE)  : ±{grand_se_area_km2:.2f} km²")
        print(f"   • 联合变异系数 (CV%)   : {grand_cv_percent:.2f}% (方差通过跨区分层联合收敛)")
        print(f"   • 耕地综合总体精度 (F1): {grand_f1*100:.2f}% (核心指标: 纯耕地综合质量)")
        print(f"   • 耕地制图查准率 (UA)  : {grand_ua*100:.2f}% (错报/虚警率: {(1.0-grand_ua)*100:.2f}%)")
        print(f"   • 耕地地面查全率 (PA)  : {grand_pa*100:.2f}% (漏报/欠报率: {(1.0-grand_pa)*100:.2f}%)")
        print(f"{'='*80}")
        print(f"🎉 全部交付物生成完毕:")
        print(f"   • 📊 多景对比总台账:   {summary_csv_path}")
        print(f"   • 🖼️ 多景对比分析图:   {plot_path}")
        print(f"   • 📑 跨区官方决策驾驶舱: {html_path}")
        print(f"   • 📂 单景明细子目录:   {scenes_dir}\n")

        return df_summary
