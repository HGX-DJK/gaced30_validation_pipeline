#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
🌾 GACED30 精度验证与无偏面积推断核心系统 (Core Engine Package)
================================================================================
遵循联合国粮农组织 (FAO) 与统计司 (UNSD)《农业统计遥感手册》规范设计。
"""

from .stats import GACED30Validator
from .harmonizer import harmonize_reference_labels, PRESET_SCHEMES
from .spatial import (
    normalize_reference_columns,
    parse_tile_coordinates,
    match_reference_points_for_tile,
    simulate_benchmark_scene,
    align_and_sample_raster
)
from .report import generate_comparison_chart, generate_html_dashboard

__all__ = [
    "GACED30Validator",
    "harmonize_reference_labels",
    "PRESET_SCHEMES",
    "normalize_reference_columns",
    "parse_tile_coordinates",
    "match_reference_points_for_tile",
    "simulate_benchmark_scene",
    "align_and_sample_raster",
    "generate_comparison_chart",
    "generate_html_dashboard"
]
