# 🌾 GACED30 (全球 30 米耕地动态数据集 2000–2024) 官方精度验证与无偏面积推断系统

> **对齐国际官方标准**：严格遵循联合国粮农组织 (**FAO**) 与联合国统计司 (**UNSD**) 联合编纂的权威专著《农业统计遥感手册》（*UN Handbook on Remote Sensing for Agricultural Statistics*，第 8、22、23、24、26 章）。  
> **服务数据源**：专为**鹏城实验室“星云”平台（iEarth DataHub）**发布的 **Global 30-m annual cropland extent dynamics (2000–2024)**（[数据集链接](https://data-starcloud.pcl.ac.cn/iearthdata/map?id=67&r_id=77)）打造的端到端自动化量化验证与无偏推断工业级工具。

---

## 📖 一、 核心理论渊源与算法架构

在卫星遥感大尺度耕地制图中，直接“数像素面积”（Pixel Counting）会因混合像元、物候混淆与边缘效应产生 15%~35% 的系统性统计偏差。本系统基于联合国手册建立双层验证闭环：

```
┌─────────────────────────┐     ┌─────────────────────────┐     ┌─────────────────────────┐
│  GACED30 30米多时相瓦片 │ ──► │  Cochran 分层随机抽样   │ ──► │   独立高分/地面真值标定   │
│ (2000-2024年 25波段COG) │     │ (按耕地/非耕地计算样点) │     │  (10m S2 / 高清影像盲审)│
└─────────────────────────┘     └─────────────────────────┘     └───────────┬─────────────┘
                                                                            │
                                ┌───────────────────────────────────────────┴─────────────┐
                                ▼                                                         ▼
                 ┌─────────────────────────────┐                           ┌─────────────────────────────┐
                 │ 1. 像元分类空间制图精度台账 │                             │  2. 联合国面积加权无偏统计    │
                 │   (OA, UA, PA, F1, IoU)     │                           │  (无偏面积, SE, CV%, 95%CI)  │
                 └──────────────┬──────────────┘                           └──────────────┬──────────────┘
                                │                                                         │
                                └─────────────────────────────┬───────────────────────────┘
                                                              ▼
                                               ┌─────────────────────────────┐
                                               │ 📑 出版级官方 HTML 分析专报 │
                                               │ 📊 全套统计 CSV 台账与高清图 │
                                               └─────────────────────────────┘
```

1. **Cochran (1977) 分层随机抽样设计**：
   依据期望标准误 $S(\hat{O}) \le 0.01$ 计算最小理论检验样点数 $n = \left( \frac{\sum W_i S_i}{S(\hat{O})} \right)^2$。
2. **Olofsson et al. (2014) 面积加权混淆矩阵（Table 2 规范）**：
   按各图层面积权重 $W_i$ 换算无偏面积比例 $\hat{p}_{ij} = W_i \frac{n_{ij}}{n_{i\cdot}}$，消除大背景带来的虚高精度。
3. **闭式解析标准误（SE）与变异系数（CV%）**：
   计算真实无偏估计面积 $\hat{A}$，输出变异系数 $CV\% = \frac{SE}{\hat{A}} \times 100\%$ 并自动给出官方数据质量评级：
   - $CV < 5\%$：⭐⭐⭐ 国家级官方发布标准（极高可靠度）
   - $5\% \le CV \le 10\%$：⭐⭐ 省部级统计可用（良好）
   - $CV > 10\%$：⚠️ 抽样变异较大（建议增加样方）
4. **2000–2024 年 25 年长时序时序稳定性评估**：
   自动检测单年伪跳变闪烁率（Single-year Flickering Rate）与多年常年耕种基质稳定度。

---

## 📁 二、 目录结构

```text
gaced30_validation_pipeline/
├── README.md                                # 📖 本说明文档
├── requirements.txt                         # 📋 运行依赖 (rasterio, numpy, pandas, matplotlib, scipy)
├── main.py                                  # 🚀 主运行入口 (自动扫描 source_data 与 reference_data，支持多景)
├── validate_gaced30.py                      # 别名执行脚本 (等价于 main.py)
├── data/                                    # 📂 核心数据目录 (源数据与参考真值分开放置)
│   ├── source_data/                         # 🛰️ 【待验源数据文件夹】：放入 1 景或多景 GACED30 .tif 瓦片
│   │   ├── README.md                        # 源数据放置指南与星云平台下载说明
│   │   ├── SDC30_N30E117_2000_2024.tif      # (示例) 区域 A 瓦片
│   │   └── SDC30_N33E117_2000_2024.tif      # (示例) 区域 B 瓦片
│   └── reference_data/                      # 📋 【权威参考数据文件夹】：支持单张全国大表或分区分景表
│       ├── README.md                        # 参考数据规范说明
│       ├── sample_ground_truth_reference.csv# 示例真值模板 (含经纬度、分类标签与真实地类)
│       └── china_cropland_samples_2024.csv  # (支持) 跨区域或全国独立参考真值大总表 (自动空间外包切分)
└── output/                                  # 📊 成果交付目录 (自动生成)
    ├── gaced30_multi_scene_dashboard.html   # 📑 跨区官方决策驾驶舱 (单文件直读，汇总全区与分景明细)
    ├── gaced30_multi_scene_comparison.png   # 🖼️ 多景对比分析高清图 (分景面积误差棒对比 + 精度指标柱状图)
    ├── gaced30_multi_scene_summary.csv      # 📊 多景对比总台账 (含各景明细与 Grand Total 联合无偏统计)
    ├── scenes/                              # 📂 各景独立深度明细子目录
    │   ├── Crop_Region_A/                   # 单景 A 精度台账、面积加权矩阵
    │   └── Crop_Region_B/                   # 单景 B 精度台账、面积加权矩阵
    └── gaced30_validation_report.html       # 📑 单景评估报告 (若仅单景运行)
```

---

## 🚀 三、 快速开始

### 1. 安装依赖
```bash
cd gaced30_validation_pipeline
pip install -r requirements.txt
```

### 2. 方式一：多景多区域快速仿真演示（无需准备数据）
```bash
python main.py --demo
```
> 系统将在内存中自动构建模拟华北平原 3 景相邻农区瓦片（河南、鲁西南、冀中南），完成多景分层抽样、跨区联合误差估计，生成多景驾驶舱与综合对比图。

### 3. 方式二：双文件夹批量运行（支持 1 景至任意多景）
1. 将下载的 1 景或多景 GACED30 瓦片（如 `Crop_114.0_35.0.tif`, `Crop_116.0_35.0.tif`）直接拷贝放入 `data/source_data/`；
2. 在 `data/reference_data/` 中放置真值数据（**支持以下两种组织方式之一**）：
   - **方式 A（全局一张表，最省心）**：放入一张包含多区域地面样点的总 CSV 表（包含 `lon`, `lat`, `ref_label`）。系统会**自动按每景 TIF 的经纬度外包矩形（Bounding Box）进行空间空间过滤截取**；
   - **方式 B（分景一一对应）**：放入命名匹配各瓦片的独立真值文件（如 `Crop_114.0_35.0_ref.csv`）；
   - **方式 C（无外部真值兜底）**：若某景未匹配到参考样点，系统会自动依据 Cochran (1977) 规程按理论最优样本量布设分层随机抽样。
3. 直接在终端运行：
```bash
python main.py
```
> 程序全自动完成：多景识别 ➔ 空间经纬度真值对齐 ➔ 智能语义映射归一化 ➔ 单景无偏推断 ➔ 跨区联合总方差合成 ➔ 输出全套报表！

### 4. 方式三：包含特殊分类体系与自定义编码的运行
若参考数据采用了 ESA WorldCover、ESRI 10m、中科院 CNLUCC 等第三方体系或自建调查编码：
```bash
# 自动识别模式（默认：自动探测 ESA 40、CNLUCC 11/12、ESRI 5 或中英文地类文字）
python main.py

# 显式指定权威预设体系
python main.py --ref-preset esa_worldcover   # 或 esri_10m, clcd, cnlucc, globeland30

# 显式指定自定义类别编码为耕地 (如指定类别 40 和 41 为耕地)
python main.py --ref-crop-values "40,41"
```


---

## 📐 四、 多景多区域跨区联合统计理论（联合国手册第 22-24 章）

当源数据包含覆盖不同地理区域的多景瓦片 $d \in \{1, \dots, D\}$ 时，联合国农业统计手册规范明确指出，**各景构成独立的空间抽样层/域（Independent Strata / Domains）**，其联合统计遵循分层全域估计公理：

1. **跨区联合无偏耕地面积（Grand Total Unbiased Area）**：
   $$\hat{A}_{grand\_unbiased} = \sum_{d=1}^D \hat{A}_{crop, d}$$
2. **联合解析标准误（Joint Standard Error）**：
   由于各景空间独立抽样，联合方差等于各景方差之和：
   $$Var(\hat{A}_{grand\_unbiased}) = \sum_{d=1}^D Var(\hat{A}_{crop, d})$$
   $$SE(\hat{A}_{grand\_unbiased}) = \sqrt{\sum_{d=1}^D Var(\hat{A}_{crop, d})}$$
3. **联合变异系数（Joint CV%）的大数定律收敛效应**：
   $$CV_{grand}\% = \frac{SE(\hat{A}_{grand\_unbiased})}{\hat{A}_{grand\_unbiased}} \times 100\%$$
   *注：随着覆盖多景瓦片联合估计，抽样方差在全域尺度上具有“分散抵消与规模收敛效应”，联合 $CV\%$ 通常显著低于单个瓦片的 $CV\%$，从而达到更高层级（国家级/联合国级）的发布标准！*
4. **全域面积加权综合精度（Area-Weighted Grand Accuracy）**：
   $$OA_{grand} = \sum_{d=1}^D \left( \frac{A_{tot, d}}{A_{grand\_tot}} \right) OA_d$$

---

## 📊 五、 成果产出解读

| 输出成果 | 核心指标与内容 | 决策与统计用途 |
| :--- | :--- | :--- |
| **`gaced30_multi_scene_dashboard.html`** | 汇总跨区联合 KPI 看板、多景横向对比台账、内嵌可视化图表 | 面向高管与统计决策人员的单文件跨区交互式综合决策驾驶舱。 |
| **`gaced30_multi_scene_summary.csv`** | 各景面积、无偏面积、偏差率、SE、CV%、OA/UA/PA、**跨区全景联合汇总（Grand Total）** | 全局与分区分景的官方统计发布总台账。 |
| **`gaced30_multi_scene_comparison.png`** | 多景面积对比柱状图（带 95% CI 误差棒）、多景制图精度（OA/UA/PA/F1）横向对比图 | 用于科技专报插图、跨区差异分析与学术出版。 |
| **`scenes/<scene_id>/`** | 单景的 `accuracy_metrics.csv` 与 `area_weighted_confusion_matrix.csv` | 针对特定网格或区县级行政单元的逐景穿透式核查明细。 |
