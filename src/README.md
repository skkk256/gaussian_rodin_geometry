# 生成流水线运行指南

主流程自动执行：Gaussian PLY → SAM3D 朝向估计 → 六视图渲染 → FAL 清理与高斯过滤 → 体积 FPS4096 → Rodin Shaded / extreme-low → bbox → 条件 ICP → DINOv3/SIFT 与轮廓联合配准 → 评分与重试。

安装、输入属性、权重布局和八文件输出见[项目 README](../README.md)；指标公式与算法约束见[配准与重试说明](../ALIGNMENT_RETRY.md)。推理模型在使用前冻结参数，CUDA 设备必须带编号。原生扩展缓存与构建放在项目目录外；模型权重、凭据、结果和实验文件不提交到 Git。

## 单物体运行

在项目根目录使用已配置的 CUDA Python 环境：

```bash
python -B -u src/run_pipeline.py \
  --input /absolute/path/object.ply \
  --scene-id scene_001 --object-id object_001 \
  --output ./result_aligned --work-dir ./.work_aligned \
  --credentials .env --device cuda:1 --seed 22101 --max-retry 1
```

`--seed` 范围为 0..65535，默认 22101。`--max-retry` 为初次生成之外的最大重试数，默认 1。bbox、ICP、DINO 均由主流程调度，无需额外配准参数。

生成与前景过滤使用前/右/左 30°、背面 0°、顶底 ±90°。DINO 和评分单独从高斯渲染六视图：四侧均 30°、顶底 ±90°；两者使用同一套相机，均不使用 FAL 图作参考。DINO 输入为完整原高斯 RGB，以 mask 筛选特征点，并用过滤高斯的局部轮廓约束优化。

## 批量运行

JSON manifest 的每项包含以下字段；相对输入路径按 manifest 所在目录解析，每个 `(scene_id, object_id)` 必须唯一：

```json
[
  {"scene_id": "scene_001", "object_id": "object_001", "input": "data/chair.ply", "seed": 22101},
  {"scene_id": "scene_001", "object_id": "object_002", "input": "data/table.ply", "seed": 22102}
]
```

串行批量：

```bash
python -B -u src/run_batch.py \
  --manifest /absolute/path/cases.json \
  --output ./result_aligned --work-dir ./.work_aligned \
  --credentials .env --device cuda:1 --max-retry 1
```

并发批量：

```bash
python -B -u src/run_batch_concurrent.py \
  --manifest /absolute/path/cases.json \
  --output ./result_aligned --work-dir ./.work_aligned \
  --credentials .env --device cuda:1 --max-retry 1 \
  --fal-workers 2 --rodin-workers 2 --max-inflight 8
```

| 阶段 | 同时执行数 | 内容 |
|---|---|---|
| `preprocess_sam3d` | 1 | 属性预过滤、SAM3D 朝向估计、原高斯六视图 |
| `fal` | `--fal-workers` | FAL 请求、前四视图分割过滤、过滤高斯渲染、FPS 与归一化 |
| `rodin` | `--rodin-workers` | Rodin 请求、模型下载与校验 |
| `postprocess` | 1 | bbox、条件 ICP、联合 DINO、评分选优、预览与八文件导出 |

四段处理不同物体时可重叠。GPU 和前景分割资源分别使用 FIFO 锁，每个 API 工作线程独享 HTTP session。`--max-inflight` 限制已进入流水线但尚未完成的物体数。配准未达标时，任务回到 Rodin 或 FAL 阶段，使用与串行运行相同的重试策略。单项失败保存状态并继续其他项，批次最终以非零状态退出。并发模式收到 Ctrl+C 后不再接收新物体，已进入的物体完成后退出。

## 评分、重试与恢复

总分为 `0.4 × 分块高斯→mesh + 0.5 × 六视图 mask IoU + 0.1 × 原始 mesh→高斯`。bbox <70 时运行 ICP，取高分候选启动 DINO。DINO 得分相对起点下降不超过 5 分时采用 DINO，以实际选中结果的得分判断是否 ≥70。

仍不足 70 时，重试 1 只跑 Rodin；重试 2 从 FAL 开始；之后两次一组交替。重试耗尽后取各轮所选结果中的最高分，并明确记录 `below_threshold`。仅图像生成有可采样过滤高斯时仍配准评分；无 opacity>0.2 的过滤高斯时记录 `unscorable`、分数 null，进入重试。

工作目录 `.work_aligned/<scene>/<object>/` 包含：

| 路径 | 内容 |
|---|---|
| `config.json`、`canonical.json` | 输入与设置校验、原始坐标系 |
| `refinements/000/` | 初次 FAL、过滤高斯、点云条件与六视图高斯配准参考 |
| `refinements/001/` | 第二次重试重新执行 FAL 的结果 |
| `attempts/000/`、`001/` 等 | 每次 Rodin 请求、原始 GLB、配准分数、矩阵与 DINO 诊断 |
| `alignment.json` | 跨轮选优与最终状态 |
| `transform.json` | 选中模型到原始世界坐标的最终矩阵和评分汇总 |
| `timing.json`、`state.json` | 处理耗时与执行状态 |

有点云条件时，完整逆变换为 `canonicalToWorld × inverse(normalization)`，随后叠加配准修正。最终 `rodinToWorld.matrix` 按 column-major 保存；NumPy 使用 `reshape(4,4,order='F')` 读取，对原始 GLB 应用一次。行向量顶点计算为 `vertices @ M[:3,:3].T + M[:3,3]`。原始 GLB 不烘焙变换，保留 Shaded / extreme-low 材质；`rodin.png` 为灰蓝色几何预览。

用相同命令续跑会校验配置和 hash、复用已完成项并继续已提交的请求。修改输入、prompt、相机、材质、seed、配准设置或重试上限时使用新的 work/result 目录。工作目录锁防止同时处理同一物体。付费 POST 提交状态不确定时明确报错，需核查已有请求；不自动创建第二次任务。

## 耗时记录

`timing.json` 的 `attempts` 表示同一工作目录的执行记录；Rodin 自动重试轮次记录在 `alignment.json`，两者含义不同。物体 `totalSeconds` 为累计处理时间，扣除本地排队；每次执行保存 `wallSeconds`、`queueSeconds`、`stages` 与 `overheadSeconds`。模型初始化、CUDA warmup、API 服务端等待和下载均计入实际发生的阶段。

并发阶段保存 `seconds / processingSeconds / queueSeconds / overheadSeconds / details`；`details` 为子步骤明细。批次 `.work_aligned/batch-timing.json` 另存完成、失败、跳过数量与墙钟时间。各物体处理时间之和可能大于批次墙钟时间。

## Fal 完整参数

默认使用 **GPT Image 2.6 Sunburst edit**，精确模型 ID 为 **`openai/gpt-image-2.6/sunburst/edit`**。实际发送 `src/fal_prompt.txt` 全文：前、右、左俯视 30°，背面平视 0°，顶底为垂直视图。`prompts/` 中的版本文件仅供参考，不决定运行时 prompt。

| 字段 | 值 |
|---|---|
| 提交地址 | `POST https://queue.fal.run/openai/gpt-image-2.6/sunburst/edit` |
| `prompt` | `src/fal_prompt.txt` 全文，原样发送 |
| `image_urls` | 一张 `gaussian.png` 的 PNG data URL |
| `num_images` | `1` |
| `image_size` | `{"width":1024,"height":1536}` |
| `quality` | `high` |
| `reasoning_effort` | `high` |
| `output_format` | `png` |
| `background` | `opaque` |
| seed | 不发送；该接口没有沿用 Rodin seed |

提交 HTTP 超时 180 秒，状态 GET 超时 60 秒，每 5 秒轮询，单次等待最多 60 分钟；可续跑已有任务。

仅当 GPT 高斯保留率不足 30% 时，使用 `POST https://queue.fal.run/google/nano-banana-2.1/edit` 最多尝试两次：`num_images=1`、`aspect_ratio=2:3`、`resolution=1K`、`output_format=png`、`thinking_level=high`、当前 FAL 轮次的基础 seed 为 `(物体 seed + 尝试序号 // 2) % 65536`，两次 Nano 分别使用此 seed 与其加一取模值；prompt 和输入图片与 GPT 完全相同。每次请求各有独立日志。Nano 只提供分辨率档位，原生单格不保证达到 512；下载后校验接近 2:3 的画幅（允许服务端像素取整），统一缩放到 1024×1536 PNG 后切为六格 512×512，原始返回尺寸记入请求日志的 `downloadedShape`。任一次通过保留率检查即使用对应图片和点云；两次 Nano 均不达标时，只使用第一次 GPT Image 的图片，不发送点云 condition。

## Rodin 完整参数

使用公网 **`https://api.hyper3d.com/api/v2`**，multipart `POST /rodin`；保存返回 UUID 和 subscription key，再 `POST /status`、`POST /download` 获取原始 GLB。

| 字段 | 值 |
|---|---|
| `images` | 五个独立 512×512 PNG：front/right/back/left/bottom；不上传 top |
| `image_label` | `["F","L","B","R","D"]` |
| `tier` | `Gen-2.5-Medium` |
| `quality` | `medium` |
| `mesh_mode` | `Raw` |
| `geometry_file_format` | `glb` |
| `material` | `Shaded` |
| `texture_mode` | `extreme-low` |
| `use_original_alpha` | 字符串 `true` |
| `is_symmetric` | `unknown` |
| `pcd_condition_cfg` | 有点云时发送字符串 `true`；仅图像时省略 |
| `pcd_condition_weight` | 有点云时发送 `0.3`；仅图像时省略 |
| `pcd_condition` | 有点云时发送实际归一化后的 `4096×3` XYZ JSON 列表；仅图像时省略 |
| `seed` | `(物体 seed + 尝试序号) % 65536`，初次尝试序号为 0 |

`image_label` 按 Rodin 物体局部方向定义；图中右上是相机位于 +X 的一侧，对应标签 L，第二行右侧相机位于 −X，对应 R；第三行右侧为底视图，对应 D。不交换图片，也不将标签改成 F/R/B/L。不发送内部服务的 `normalize`、`point_count`、`pcd_tokens`、`cfg`、`steps` 或 `cutoff` 参数。

生成 POST HTTP 超时 600 秒；状态 / 下载地址查询超时 180 秒，文件下载超时 300 秒。每 5 秒轮询，单次等待最多 90 分钟；超时后用同一命令继续已有任务。
