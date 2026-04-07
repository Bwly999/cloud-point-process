# 光谱共焦 PNG 点云图处理工具

这个项目用于处理光谱共焦设备输出的 `16-bit` 单通道 PNG 高度图，输出以下结果：

- 接缝补偿后的高度图
- 重采样后的高度图
- `X/Y` 方向梯度图
- `X/Y` 方向二阶导图
- `X/Y` 方向剖面曲率图
- `PNG / CSV / TIFF` 三类结果文件
- 额外导出 `X,Y,Z` 三列点表 CSV
- 一张用于快速检查效果的总览图 `overview.png`

## 输入假设

- 输入文件为 `16-bit` 单通道 PNG
- 像素值与高度换算关系：`z_mm = gray * 0.0001`
- 默认扫描方式为 `3` 道拼接，每道高度 `2048` 像素
- 默认物理分辨率：
  - `dx = 0.08 mm/pixel`
  - `dy = 0.005615 mm/pixel`

## 处理流程

1. 读取 PNG，并可选裁切左右不需要参与计算的区域
2. 逐列估计每条接缝的整带偏移 `Δ(x)`
3. 按扫描带做整带基线补偿
4. 在原始高分辨率图上，对接缝窄带做局部重建，压掉 seam-local 锯齿
5. 可选地对高度图做离群噪点滤除，压掉孤立尖刺和小团簇异常
6. 在原始物理网格上做轻平滑并计算梯度、二阶导和剖面曲率
7. 沿 `Y` 方向做分步重采样，用于导出高度图和导数结果图
8. 导出 `PNG / CSV / TIFF`
9. 生成 `overview.png`

## 常用命令

### 1. 生成假图并跑完整流程

```bash
python process_heightmap.py --generate-synthetic --output-dir demo_outputs
```

### 2. 处理真实图

```bash
python process_heightmap.py ^
  --input your.png ^
  --output-dir out
```

### 3. 处理真实图并裁切左右边缘

```bash
python process_heightmap.py ^
  --input your.png ^
  --output-dir out ^
  --crop-left-px 120 ^
  --crop-right-px 80
```

### 4. 处理真实图并启用离群噪点滤除

```bash
python process_heightmap.py ^
  --input your.png ^
  --output-dir out ^
  --outlier-filter-mode medium ^
  --outlier-window-size 5 ^
  --outlier-threshold-mm 0.02 ^
  --outlier-max-cluster-size 4 ^
  --outlier-debug
```

### 5. 生成带尖刺/小团簇噪点的假图

```bash
python process_heightmap.py ^
  --generate-synthetic ^
  --output-dir noisy_demo ^
  --synthetic-outlier-spike-count 12 ^
  --synthetic-outlier-cluster-count 2 ^
  --synthetic-outlier-cluster-size 2 ^
  --synthetic-outlier-amplitude-mm 0.08
```

### 4. 只打包代码和文档为 zip

```bash
python package_zip.py
```

如需指定输出路径：

```bash
python package_zip.py --output my-release.zip
```

## 关键参数

### 基础分辨率

- `--dx-mm`
  - `X` 方向物理分辨率
- `--dy-mm`
  - `Y` 方向物理分辨率
- `--dz-mm`
  - 灰度到高度的换算分辨率

### 裁切参数

- `--crop-left-px`
  - 从左侧裁掉多少列像素
- `--crop-right-px`
  - 从右侧裁掉多少列像素

建议先用这两个参数切除原图中明确不需要分析的边缘区域，再进行接缝处理和求导。

### 接缝整带补偿

- `--seam-window`
  - 估计整带偏移时，接缝上下各取多少行
- `--smooth-sigma-x`
  - 对逐列偏移 `Δ(x)` 做 `X` 向轻平滑

如果接缝处不同 `X` 的偏移差异很快，`--smooth-sigma-x` 不要设太大。

### 接缝窄带重建

- `--seam-flatten-half-window`
  - 接缝核心带半宽
- `--seam-flatten-blend-width`
  - 核心带外侧混合宽度
- `--seam-flatten-sigma-x`
  - 对上下锚线做 `X` 向轻平滑
- `--seam-flatten-method`
  - `linear` 为旧版简单插值，默认使用
  - `quadratic` 为固定二阶导风格过渡
  - `cubic` 为三次插值

推荐调参顺序：

1. 先把 `--seam-flatten-half-window` 设小，例如 `2`
2. 如果边缘还有明显过渡，再增加 `--seam-flatten-blend-width`
3. 默认优先使用 `--seam-flatten-method linear`
4. 如果希望比 linear 更柔和、但又不想像 cubic 那样过度弯折，可尝试 `quadratic`
5. 最后再考虑增大 `--seam-flatten-sigma-x` 或尝试 `cubic`

### 求导前轻平滑

- `--gaussian-sigma`
  - 对求导前的高度图做轻量高斯平滑，单位 `mm`
- `--pre-smooth-x-sigma`
  - 仅对 `X` 方向做轻量高斯平滑，单位 `mm`

这两个参数现在都按物理长度 `mm` 定义，而不是像素单位。
这样在不同 `downsample` 下，平滑尺度会保持一致，不会因为采样率变化而把曲率额外压小。

### 离群噪点滤除

- `--outlier-filter-mode`
  - `off` 关闭，默认值
  - `conservative` 主要处理单像素尖刺
  - `medium` 在单点基础上，额外处理小团簇离群点
- `--outlier-window-size`
  - 局部中值窗口尺寸，必须是奇数
  - 设为 `0` 时，按模式使用默认值
- `--outlier-threshold-mm`
  - 像素相对局部中值的离群阈值，单位 `mm`
  - 设为 `0` 时，按模式使用默认值
- `--outlier-max-cluster-size`
  - 允许被替换的小连通域最大面积，单位像素
  - 设为 `0` 时，按模式使用默认值
- `--outlier-replace-mode`
  - 当前仅支持 `median`
- `--outlier-debug`
  - 导出噪点滤除诊断图和组件 CSV
  - 用来判断某块区域为什么没被替换

默认模式参数：

- `conservative`
  - `window_size = 3`
  - `threshold_mm = 0.02`
  - `max_cluster_size = 1`
- `medium`
  - `window_size = 5`
  - `threshold_mm = 0.01`
  - `max_cluster_size = 4`

建议：

- 如果真实表面上存在少量突兀飞点，先试 `conservative`
- 如果异常表现为 `2x2`、`3x3` 一类的小块噪点，再试 `medium`
- 阈值越小、连通域面积越大，滤除越激进，也越容易抹掉真实尖峰细节

开启 `--outlier-debug` 后，会额外导出：

- `outlier_local_median_preview.png`
- `outlier_residual.png`
- `outlier_candidate_mask.png`
- `outlier_core_mask.png`
- `outlier_replace_mask.png`
- `outlier_components.csv`

常见判读方法：

- `outlier_residual` 明显高，但 `candidate_mask` 没亮
  - `--outlier-threshold-mm` 设得太高
- `candidate_mask` 亮了，但 `replace_mask` 没亮
  - 通常是 `core_area` 超过 `--outlier-max-cluster-size`
- `local_median` 在该区域明显被周围结构带偏
  - 可以尝试增大 `--outlier-window-size`

## 输出文件

### 高度图

- `height_corrected_preview.png`
- `height_resampled_preview.png`
- `height_corrected_mm.tiff`
- `height_resampled_mm.tiff`
- `height_resampled_xyz.csv`

### 梯度图

- `grad_x.png` / `grad_x.csv` / `grad_x.tiff`
- `grad_y.png` / `grad_y.csv` / `grad_y.tiff`
- `grad_x_xyz.csv` / `grad_y_xyz.csv`

### 二阶导图

- `curv2_x.png` / `curv2_x.csv` / `curv2_x.tiff`
- `curv2_y.png` / `curv2_y.csv` / `curv2_y.tiff`
- `curv2_x_xyz.csv` / `curv2_y_xyz.csv`

### 剖面曲率图

- `curve_x.png` / `curve_x.csv` / `curve_x.tiff`
- `curve_y.png` / `curve_y.csv` / `curve_y.tiff`
- `curve_x_xyz.csv` / `curve_y_xyz.csv`

其中新增的 `*_xyz.csv` 统一为三列：

- `X`：物理 `X` 坐标，单位 `mm`
- `Y`：物理 `Y` 坐标，单位 `mm`
- `Z`：该点对应的高度、梯度或曲率数值

### 总览图

- `overview.png`

默认会把以下 6 张图拼成一张总览图：

- 原始高度预览
- 接缝补偿后高度预览
- 重采样后高度预览
- `grad_y`
- `curv2_y`
- `curve_y`

这个总览图主要用于快速判断：

- 接缝是否被压平
- `Y` 向梯度和曲率是否仍被接缝主导

## 查看命令行说明

```bash
python process_heightmap.py --help
```

## 代码打包说明

`package_zip.py` 会使用白名单打包，只包含代码和文档：

- `.gitignore`
- `README.md`
- `process_heightmap.py`
- `package_zip.py`
- `cloud_point_process/`
- `tests/`
- `docs/`

不会包含：

- `demo_outputs*`
- `sample_outputs*`
- `debug_outputs*`
- `tmp_sigma_scan/`
- `__pycache__/`
- 现有 zip 包
