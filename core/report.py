#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
📊 GACED30 可视化图表与官方决策驾驶舱报告引擎 (Report & Dashboard Generator)
================================================================================
负责生成高清对比分析图 (Matplotlib) 及富交互 HTML 跨区域决策驾驶舱。
"""

import datetime
import numpy as np
import matplotlib.pyplot as plt

# 设置 Matplotlib 字体以支持中文
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


def generate_comparison_chart(scene_results, grand_summary, output_plot, z_score=1.96):
    """生成多景多区域对比分析图表 (严格以耕地专项精度为呈现基准)"""
    n_scenes = len(scene_results)
    sids = [r["scene_id"] for r in scene_results]
    naive_areas = [r["naive_crop_km2"] for r in scene_results]
    calibrated_areas = [r["calibrated_crop_km2"] for r in scene_results]
    ses = [r["SE_area_km2"] * z_score for r in scene_results]

    f1s = [r["F1_crop"] * 100 for r in scene_results]
    uas = [r["UA_crop"] * 100 for r in scene_results]
    pas = [r["PA_crop"] * 100 for r in scene_results]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(12, n_scenes * 2.2), 10), dpi=180)

    x = np.arange(n_scenes)
    width = 0.35

    # 1. 各区域数像元面积 vs 联合国无偏面积对比
    ax1.bar(x - width/2, naive_areas, width, label="待验直接数像元 (Naive Map)", color="#e74c3c", alpha=0.85)
    ax1.bar(x + width/2, calibrated_areas, width, yerr=ses, capsize=5, label="联合国无偏校准 (Unbiased ±95%CI)", color="#27ae60", alpha=0.85)
    ax1.set_ylabel("耕地面积 (km²)", fontsize=11, fontweight="bold")
    ax1.set_title(f"图 A. 多景各区域耕地面积对比与 95% 置信区间 (全景联合 CV = {grand_summary['grand_cv']:.2f}%)", fontsize=12, pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(sids, rotation=15 if n_scenes > 4 else 0, fontsize=10)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    ax1.legend(loc='upper right')

    # 2. 各区域耕地专项制图精度对比 (UA, PA, F1)
    w3 = 0.25
    ax2.bar(x - w3, uas, w3, label="耕地查准率 (UA %)", color="#2980b9", alpha=0.85)
    ax2.bar(x, pas, w3, label="耕地查全率 (PA %)", color="#27ae60", alpha=0.85)
    ax2.bar(x + w3, f1s, w3, label="耕地综合总体精度 (F1 %)", color="#e67e22", alpha=0.85)
    ax2.set_ylabel("耕地精度百分比 (%)", fontsize=11, fontweight="bold")
    ax2.set_title("图 B. 各景【耕地专项制图精度】(查准率 UA, 查全率 PA, 综合 F1) 对比一览 (纯耕地评估)", fontsize=12, pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(sids, rotation=15 if n_scenes > 4 else 0, fontsize=10)
    ax2.set_ylim(0, 105)
    ax2.axhline(85, color="#e74c3c", linestyle=":", label="联合国优良参考红线 (85%)")
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    ax2.legend(loc='lower right')

    fig.subplots_adjust(top=0.94, bottom=0.08, left=0.07, right=0.96, hspace=0.35)
    plt.savefig(output_plot, dpi=180, bbox_inches='tight')
    plt.close()


def generate_html_dashboard(df_summary, grand_summary, output_html):
    """生成多景多区域联合国官方决策分析 HTML 驾驶舱"""
    now_str = datetime.datetime.now().strftime('%Y年%m月%d日 %H:%M')

    table_rows = ""
    for idx, r in df_summary.iterrows():
        is_total = "联合汇总" in str(r["区域/瓦片标识 (Scene ID)"])
        style = "font-weight:bold; background-color:#eaf2f8;" if is_total else ""
        table_rows += f"""<tr style="{style}">
            <td>{r['区域/瓦片标识 (Scene ID)']}</td>
            <td style="color:#666; font-size:12px;">{r['地理空间范围 (Bounds)']}</td>
            <td>{r['有效校验幅员 (km²)']}</td>
            <td style="color:#c0392b;">{r['待验数像元面积 (km²)']}</td>
            <td style="color:#27ae60;">{r['联合国无偏面积 (km²)']}</td>
            <td style="color:#d35400;">{r['无偏面积(万亩)']}</td>
            <td>{r['直接数像元偏差率']}</td>
            <td>{r['解析标准误 (SE km²)']}</td>
            <td><b>{r['变异系数 (CV%)']}</b></td>
            <td style="color:#d35400; font-weight:bold;">{r['耕地综合总体精度 (F1)']}</td>
            <td style="color:#2980b9; font-weight:bold;">{r['耕地查准率 (UA)']}</td>
            <td style="color:#27ae60; font-weight:bold;">{r['耕地查全率 (PA)']}</td>
            <td>{r['耕地交并比 (IoU)']}</td>
            <td><span style="font-size:12px;">{r['官方采信评级']}</span></td>
        </tr>"""

    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GACED30 多景多区域跨区联合验证与无偏面积决策驾驶舱</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; background-color: #f4f7f6; color: #2c3e50; margin: 0; padding: 25px; line-height: 1.6; }}
        .container {{ max-width: 1400px; margin: 0 auto; background: #fff; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.08); padding: 35px; }}
        .header {{ border-bottom: 2px solid #2980b9; padding-bottom: 15px; margin-bottom: 25px; }}
        .header h1 {{ margin: 0 0 8px 0; color: #1a5276; font-size: 26px; }}
        .badge {{ display: inline-block; padding: 6px 14px; background-color: #e8f8f5; color: #117864; border-radius: 20px; font-weight: bold; font-size: 14px; border: 1px solid #a3e4d7; }}
        .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 15px; margin: 25px 0; }}
        .kpi-item {{ background: #f8f9fa; border-left: 4px solid #3498db; border-radius: 6px; padding: 18px; box-shadow: 0 2px 6px rgba(0,0,0,0.03); }}
        .kpi-title {{ font-size: 13px; color: #7f8c8d; text-transform: uppercase; }}
        .kpi-val {{ font-size: 24px; font-weight: bold; color: #2c3e50; margin: 5px 0; }}
        .card {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 22px; margin-bottom: 25px; }}
        .card h3 {{ margin-top: 0; color: #2c3e50; border-left: 4px solid #27ae60; padding-left: 10px; font-size: 18px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; font-size: 13px; }}
        th, td {{ border: 1px solid #e2e8f0; padding: 10px 12px; text-align: left; }}
        th {{ background-color: #f1f5f9; color: #334155; font-weight: 600; }}
        tr:nth-child(even) {{ background-color: #f8fafc; }}
        .chart-container {{ text-align: center; margin: 20px 0; }}
        .chart-container img {{ max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.08); }}
        .footer {{ text-align: center; color: #94a3b8; font-size: 13px; margin-top: 30px; border-top: 1px solid #e2e8f0; padding-top: 15px; }}
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>🌾 GACED30 多景多区域跨区联合精度验证与无偏面积决策驾驶舱</h1>
        <div style="color: #7f8c8d; font-size: 14px;">
            数据源：鹏城实验室星云 iEarth DataHub (ID:67, SDC30 30米) | 理论标准：FAO/UNSD 联合国农业统计遥感手册 | 生成时间：{now_str}
        </div>
        <div class="badge" style="margin-top:10px;">{grand_summary['grand_badge']} (共校验 {grand_summary['n_scenes']} 个独立农区瓦片)</div>
    </div>

    <!-- 纯耕地考核说明 -->
    <div style="background:#fef9e7; border:1px solid #f9e79f; border-left:5px solid #f39c12; border-radius:6px; padding:12px 18px; margin:20px 0; font-size:13px; color:#7d6608;">
        <b>💡 纯耕地专属精度考核说明</b>：遵循《联合国农业统计遥感手册》规范，本系统已彻底剔除容易产生高精度假象的“全图背景包含度 (OA)”，全套评估指标与 KPI 驾驶舱<strong>100% 仅针对耕地自身的制图查准率 (UA)、查全率 (PA)、综合质量 (F1-Score) 与真实无偏面积</strong>进行严密评定。
    </div>

    <!-- 跨区联合总览 KPI -->
    <div class="kpi-grid">
        <div class="kpi-item" style="border-left-color: #27ae60;">
            <div class="kpi-title">联合国联合无偏面积</div>
            <div class="kpi-val">{grand_summary['grand_calibrated']:,.1f} km²</div>
            <div class="kpi-desc">{grand_summary['grand_calibrated']*0.15:,.1f} 万亩 (无偏校准基准)</div>
        </div>
        <div class="kpi-item" style="border-left-color: #2980b9;">
            <div class="kpi-title">耕地用户精度 (查准率 UA)</div>
            <div class="kpi-val">{grand_summary['grand_ua']*100:.2f}%</div>
            <div class="kpi-desc">错报/虚警率 {(1.0-grand_summary['grand_ua'])*100:.2f}% (重点核心)</div>
        </div>
        <div class="kpi-item" style="border-left-color: #8e44ad;">
            <div class="kpi-title">耕地生产者精度 (查全率 PA)</div>
            <div class="kpi-val">{grand_summary['grand_pa']*100:.2f}%</div>
            <div class="kpi-desc">漏报/欠报率 {(1.0-grand_summary['grand_pa'])*100:.2f}% (重点核心)</div>
        </div>
        <div class="kpi-item" style="border-left-color: #d35400;">
            <div class="kpi-title">综合总体精度 (F1-Score)</div>
            <div class="kpi-val">{grand_summary['grand_f1']*100:.2f}%</div>
            <div class="kpi-desc">推荐门槛 ≥ 85.0% (纯耕地综合质检)</div>
        </div>
        <div class="kpi-item" style="border-left-color: #c0392b;">
            <div class="kpi-title">直接数像元系统偏差 (Bias)</div>
            <div class="kpi-val">{grand_summary['grand_bias_pct']:+.2f}%</div>
            <div class="kpi-desc">数像元面积 {grand_summary['grand_naive']:,.1f} km² (虚报高估)</div>
        </div>
    </div>

    <!-- 多景对比汇总表 -->
    <div class="card">
        <h3>📋 一、 多景各区域独立指标与全区联合汇总台账 (Multi-Scene Breakdown)</h3>
        <div style="overflow-x: auto;">
            <table>
                <thead>
                    <tr>
                        <th>区域/瓦片标识</th>
                        <th>空间范围</th>
                        <th>有效校验幅员(km²)</th>
                        <th>数像素面积(km²)</th>
                        <th>无偏面积(km²)</th>
                        <th>无偏面积(万亩)</th>
                        <th>偏差率</th>
                        <th>标准误(SE)</th>
                        <th>变异系数(CV)</th>
                        <th style="color:#d35400;">耕地综合总体精度(F1)</th>
                        <th style="color:#2980b9;">耕地查准率(UA)</th>
                        <th style="color:#27ae60;">耕地查全率(PA)</th>
                        <th>耕地交并比(IoU)</th>
                        <th>官方采信评级</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
        </div>
    </div>

    <!-- 对比图表展示 -->
    <div class="card">
        <h3>📊 二、 多区域面积与精度对比可视化一览</h3>
        <div class="chart-container">
            <img src="gaced30_multi_scene_comparison.png" alt="Multi Scene Comparison">
        </div>
    </div>

    <div class="footer">
        本报告由《联合国农业统计遥感手册》（FAO/UNSD）官方多区域分层联合推断引擎自动生成 | 遵循 Olofsson et al. (2014) 国际统计规范
    </div>
</div>
</body>
</html>
"""
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_content)
