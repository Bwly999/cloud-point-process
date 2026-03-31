# 光谱共焦 PNG 点云图处理工具

这个项目用于处理光谱共焦设备输出的 `16-bit` 单通道 PNG 高度图，输出以下结果：

- 接缝补偿后的高度图
- 重采样后的高度图
- `X/Y` 方向梯度图
- `X/Y` 方向二阶导图
- `X/Y` 方向剖面曲率图
- `PNG / CSV / TIFF` 三类结果文件
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
5. 沿 `Y` 方向做分步重采样
6. 计算梯度、二阶导和剖面曲率
7. 导出 `PNG / CSV / TIFF`
8. 生成 `overview.png`

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
  - `cubic` 为三次插值

推荐调参顺序：

1. 先把 `--seam-flatten-half-window` 设小，例如 `2`
2. 如果边缘还有明显过渡，再增加 `--seam-flatten-blend-width`
3. 默认优先使用 `--seam-flatten-method linear`
4. 最后再考虑增大 `--seam-flatten-sigma-x` 或尝试 `cubic`

### 求导前轻平滑

- `--gaussian-sigma`
  - 对重采样后的高度图做轻量平滑，抑制噪声放大

## 输出文件

### 高度图

- `height_corrected_preview.png`
- `height_resampled_preview.png`
- `height_corrected_mm.tiff`
- `height_resampled_mm.tiff`

### 梯度图

- `grad_x.png` / `grad_x.csv` / `grad_x.tiff`
- `grad_y.png` / `grad_y.csv` / `grad_y.tiff`

### 二阶导图

- `curv2_x.png` / `curv2_x.csv` / `curv2_x.tiff`
- `curv2_y.png` / `curv2_y.csv` / `curv2_y.tiff`

### 剖面曲率图

- `curve_x.png` / `curve_x.csv` / `curve_x.tiff`
- `curve_y.png` / `curve_y.csv` / `curve_y.tiff`

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
