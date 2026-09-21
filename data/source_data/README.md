# 🛰️ 待验源数据目录 (Source Data)

本目录专门用于存放从**鹏城星云 iEarth DataHub**（[https://data-starcloud.pcl.ac.cn/iearthdata/map?id=67&r_id=77](https://data-starcloud.pcl.ac.cn/iearthdata/map?id=67&r_id=77)）下载的待评估 30 米耕地动态遥感影像。

### 格式要求：
- 文件类型：`.tif` / `.tiff` (Cloud-Optimized GeoTIFF)
- 命名规则示例：`Crop_116.0_35.0.tif`（按 3°×3° 瓦片组织）
- 波段结构：支持 1~25 个波段（对应 2000–2024 年历年耕地）
- 像元值编码：
  - `0`: 非耕地 (Non-Cropland)
  - `1`: 耕地 (Cropland，含一年生、多年生、休耕、大棚)
  - `255`: 无效/掩膜像元 (NoData)

### 多景与多区域支持：
- 支持放入 **1 景至任意多景** GeoTIFF 瓦片（覆盖不同网格/省份/流域）。
- 程序启动时会自动批量扫描全部 `.tif` 瓦片，逐一进行精度计算与无偏推断，并自动进行跨区域联合无偏面积估计（Grand Total）与联合误差收敛分析。

> 💡 **提示**：只需将下载的 1 景或多景 `.tif` 放入本目录，程序启动时即可全自动批量识别并加载，无需手动输入完整路径！

