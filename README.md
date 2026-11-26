# Gaussian Rodin Geometry

输入一个物体的 Gaussian PLY，输出带低分辨率材质的 Rodin 模型、配准到原始世界坐标的变换矩阵，以及各阶段六视图和过滤后的高斯。

当前主流程已接入 `align/`，单物体、串行批量和并发批量都会在每次 Rodin 生成后自动执行 **bbox → 条件 ICP → DINOv3/SIFT 联合配准 → 评分选优**。无需单独运行配准脚本；旧的 `--bbox` 参数已移除。评分不足 70 时按 `--max-retry` 重生成，默认最多额外重试一次。详细指标、算法和诊断文件见 [配准与重试说明](ALIGNMENT_RETRY.md)。

## 流程

```text
Gaussian PLY
  → 属性预过滤
  → SAM3D 五视角朝向估计
  → 高斯六视图渲染
  → Fal 清理与补全六视图
  → 前景分割与高斯过滤
  → 高斯体积采样、FPS 4096 与独立点云归一化
  → Rodin 几何与 extreme-low 材质生成
  → 完整逆变换与 bbox 配准、评分
  → bbox <70 时运行 ICP，选分数较高的候选
  → 六视图高斯参考 DINO/SIFT + 轮廓联合配准，分数下降不超过 5 分则采用 DINO
  → 最佳分数 <70 且有剩余重试次数时，重生成并重复配准
  → 导出所有尝试中分数最高的模型、最终矩阵与评分
```

Fal 默认使用 `openai/gpt-image-2.6/sunburst/edit`，实际发送 [src/fal_prompt.txt](src/fal_prompt.txt) 全文，其中明确前、右、左为 30°，背面为 0°，顶底为垂直视图。`prompts/` 保存版本参考，当前用途见 [prompt 说明](prompts/README.md)。

以属性预过滤后的高斯数量为基准，如果 GPT 输出导致高斯删除超过 70%，自动改用 `google/nano-banana-2.1/edit` 最多尝试两次。两次均使用相同原始输入图、prompt 和相机，seed 分别为该物体的 seed 与 `(seed + 1) % 65536`；任一次保留至少 30% 即使用对应点云继续。两次 Nano 均保留不足 30%（包括全部删空）时，从第一次 GPT Image 的六视图中选择前、右、后、左、底五张继续生成 Rodin，仅上传图片，省略 `pcd_condition`、`pcd_condition_cfg`、`pcd_condition_weight`，不生成点云条件。有可采样过滤高斯时，仍独立采样用于配准评分；没有 opacity>0.2 的过滤高斯时，该轮标记 `unscorable`，分数为 null，并进入重试。选择结果与原因保存在工作目录日志中。

### 自动配准、评分与重试

- 总分为 0–100：`40% × 分块高斯→mesh + 50% × 六视图 mask IoU + 10% × 原始 mesh→高斯`，参考为本轮过滤高斯。
- 先运行 bbox；bbox 分数 <70 才运行 ICP，并选两者中分数较高的变换作为 DINO 起点。
- DINO 和评分使用同一套六视图高斯/模型相机：正、右、后、左均为 30°，顶、底为 ±90°。生成流程的背面 0° 不用于配准或评分。
- 配准沿用旧版 FAL／高斯参考路由：前／右／左逐张检测漂移，三张全部通过才一起使用 FAL；顶／底分别判断。背面固定使用 30° 原高斯，不检测也不使用 0° FAL 背面。完整 RGB 输入 DINO 512／768 双尺度，再用 mask 筛选匹配点；FAL 使用完整分割轮廓，高斯使用过滤高斯局部轮廓。轮廓不足 64 像素时屏蔽该视角的特征和轮廓约束。
- DINO 比 bbox/ICP 最佳结果下降不超过 5 分时采用 DINO，超过 5 分才保留原候选。实际选中结果 ≥70 即结束；仍 <70 时，第一次重试只生成 Rodin，第二次从 FAL 开始，之后交替执行。所有尝试中保留选中结果的最高分，重试耗尽仍不达标时标记 `below_threshold`。
- `--max-retry 1` 表示初次生成之外允许一次重试；设为 `0` 则只生成一次，设为 `2` 才允许第二次从 FAL 开始的重试。此重试计数独立于高斯保留率不足时的 Nano 尝试。

## 环境与配置

使用带明确 CUDA 设备编号的 Linux / CUDA 环境。当前已验证的 Python 环境为 `sam3d-objects`，关键依赖见 [requirements.txt](requirements.txt)。PyTorch、gsplat、PyTorch3D、nvdiffrast 及 SAM3D 所需 attention 扩展需要与 CUDA 环境匹配；不能仅依靠通用的 pip 命令完成所有原生依赖安装。第三方运行源码已放在 `third_party/`。

在已经配置好的环境中进入仓库：

```bash
conda activate sam3d-objects
cd /path/to/gaussian_rodin_geometry
```

权重不随仓库提交。在项目根目录创建 `ckpt/`，放置以下文件或指向已有权重的符号链接：

| 路径 | 内容 |
|---|---|
| `ckpt/sam3d/` | `pipeline_ssonly.yaml`、`ss_generator.yaml`、`ss_generator.ckpt`、`ss_decoder.yaml`、`ss_decoder.ckpt` 及其配置引用的配套文件；保留原有目录结构 |
| `ckpt/dinov2_vitl14_reg4_pretrain.pth` | DINOv2 权重，供 SAM3D 朝向估计使用 |
| `ckpt/dinov3/` | 可由 Transformers 本地加载的 DINOv3 模型配置与权重，供自动配准使用 |
| `ckpt/u2net.onnx` | 前景分割权重 |
| `ckpt/isnet-general-use.onnx` | 细节前景分割权重 |

API 凭据通过环境变量 `FAL_API_KEY`、`RODIN_API_KEY`，或 `--credentials` 指定的私密 dotenv 文件加载。例如：

```bash
cp .env.template .env
# 编辑 .env，填写自己的 FAL_API_KEY 和 RODIN_API_KEY。
```

`.env`、本机的 `.env.example`、权重、生成结果及工作缓存均不提交到 Git。运行会调用 Fal 与 Rodin 的付费生成接口。

## 单个高斯输入

在项目根目录运行，将 `--input` 替换为一个物体的高斯文件路径：

```bash
python -B -u src/run_pipeline.py \
  --input /absolute/path/object.ply \
  --scene-id scene_001 --object-id object_001 \
  --output ./result_aligned --work-dir ./.work_aligned \
  --credentials .env --device cuda:1 --seed 22101 --max-retry 1
```

此命令执行完整的生成、自动配准与评分重试。`--device` 必须包含设备编号，例如 `cuda:0` 或 `cuda:1`。`scene-id` 和 `object-id` 决定输出子目录，seed 范围为 `0..65535`。首次使用新版流程必须选择新的工作目录与结果目录；旧的纯几何缓存不能按新配置直接续跑。

输入为 binary little-endian Gaussian PLY，属性使用 float32：`x/y/z`、`opacity`、`scale_0..2`、`rot_0..3`（wxyz）、`f_dc_0..2`，支持 SH degree 0 或 3。`opacity` 存 logit，`scale_*` 存 log scale；源坐标约定为 −Y 向上、+X 向前、+Z 向右。

## 输出目录

上述命令的结果位于 **`result_aligned/scene_001/object_001/`**：

```text
result_aligned/
└── scene_001/
    └── object_001/
        ├── rodin.glb
        ├── transform.json
        ├── gaussian.png
        ├── fal_refine.png
        ├── gaussian_filter.ply
        ├── gaussian_filter.png
        ├── rodin.png
        └── timing.json
```

| 文件 | 内容 |
|---|---|
| `rodin.glb` | 选中的 Rodin 原始模型，含 Shaded / extreme-low 材质，保持生成坐标，尚未烘焙世界坐标变换 |
| `transform.json` | `rodinToWorld` 最终变换；`alignmentScore`、`selectedAttempt`、`finalStage`、`alignmentReport` 记录分数、选中轮次及各阶段结果；仅图像生成另标记 `imageOnly=true` |
| `gaussian.png` | 前景过滤前的高斯六视图，也是 Fal 的原始输入 |
| `fal_refine.png` | 最终采用的 GPT 或 Nano Banana 补全六视图 |
| `gaussian_filter.ply` | 实际过滤后的高斯，保留原世界坐标和源属性；仅图像模式保存第二次 Nano 的过滤结果，可能为 0 点 |
| `gaussian_filter.png` | 过滤高斯六视图，与原图共用相机；0 点时为空白白底图 |
| `rodin.png` | 原始模型应用最终配准矩阵后的六视图，采用灰蓝色几何明暗预览；GLB 中保留材质 |
| `timing.json` | 各阶段处理耗时、排队及缓存记录 |

### 六视图布局与分辨率

`gaussian.png`、`fal_refine.png`、`gaussian_filter.png`、`rodin.png` 均采用以下规格。所有尺寸按**宽×高**表示：

| 项目 | 规格 |
|---|---|
| 视图数量与布局 | 6 张，3 行 × 2 列 |
| 单个 view | **512×512** |
| 输出 PNG 拼图 | **1024×1536** |
| 前四张相机 | 前、右、左为 30° 俯视，背面为 0° |
| 顶、底相机 | 正上方垂直向下、正下方垂直向上 |
| 高斯前景过滤 | **只用前四张**，顶、底不参与 |
| Rodin 图片输入 | **前四张 + 底视图**，共 5 张，每张 512×512 |
| DINO 配准输入 | **前、右、后、左、顶、底**六视图；按漂移检查选择 FAL／高斯，背面固定 30° 高斯；四侧 30°、顶底 ±90° |
| mask IoU 评分 | 过滤高斯与 mesh 的全部 6 张视图，与 DINO 共用相机，背面 30° |

拼图按从左到右、从上到下排列，索引从 0 开始：

```text
┌──────────────────┬──────────────────┐
│ 0 front（前）    │ 1 right（右）    │
├──────────────────┼──────────────────┤
│ 2 back（后）     │ 3 left（左）     │
├──────────────────┼──────────────────┤
│ 4 top（正上方）  │ 5 bottom（正下方）│
└──────────────────┴──────────────────┘
```

GPT Image 和 Nano Banana 都处理完整六视图拼图，prompt 明确标注最后一行是正上方和正下方直拍。GPT 设置整图 1024×1536；Nano Banana 设置 `resolution=1K`、`aspect_ratio=2:3`，原生返回尺寸可能取整，下载后统一到 1024×1536，再切成六张 512×512。这里的 512 是保存与上传的单格尺寸，不代表 Nano 原生生成的单格分辨率；原始返回尺寸记录在请求日志的 `downloadedShape`。

Rodin 取索引 `[0,1,2,3,5]`，**D / down 来自右下角**，不上传左下角 top。上传标签按物体局部方向定义为 `[F,L,B,R,D]`：拼图右上相机位于 +X，对应 L；第二行右侧相机位于 −X，对应 R。SAM3D 朝向估计独立使用五张 512×512 图。

`transform.json` 中矩阵按 column-major 存储，加载 `rodin.glb` 后应用一次即可；不要重复归一化或重复应用逆变换。用 `alignment` 和 `alignmentScore` 判断是否达标；不可评分的仅图像结果保留粗放置矩阵，分数为 null。

工作缓存保存在 **`.work_aligned/scene_001/object_001/`**。`refinements/000/` 保存初次 FAL、过滤高斯和点云条件，`refinements/001/` 保存第二次重试重跑 FAL 的产物；`attempts/000/`、`001/` 等分别保存每次 Rodin 与配准诊断。根部 `alignment.json` 汇总跨轮选优，每轮 `alignment.json` 保存 bbox／ICP／DINO 分数、矩阵和参考路由。仅图像模式另存 `image-only.json`，不生成 `condition.npz`。相同命令可继续已有请求；更改输入、prompt、相机、材质、seed 或 `--max-retry` 时应使用新的工作目录与结果目录。

## 批量运行

串行批量与并发批量使用相同的自动配准和重试策略，均支持 `--max-retry`。例如并发批量：

```bash
python -B -u src/run_batch_concurrent.py \
  --manifest /absolute/path/cases.json \
  --output ./result_aligned --work-dir ./.work_aligned \
  --credentials .env --device cuda:1 --max-retry 1 \
  --fal-workers 2 --rodin-workers 2 --postprocess-devices cuda:2 cuda:3 --max-inflight 8
```

串行批量改用 `src/run_batch.py`，省略三个并发参数。manifest 格式见 [详细运行指南](src/README.md#批量运行)。

## 代码目录

| 目录 | 用途 |
|---|---|
| `src/` | 生成、自动配准与评分重试的主流程，单物体与批量入口、结果 viewer |
| `prompts/` | Fal prompt 的中英文版本及修订说明 |
| `manifests/` | 本机批量示例；示例中的输入路径需按实际数据修改 |
| `third_party/` | 几何流程使用的 SAM3D、DINOv2 源码；不包含 viewer 浏览器依赖 |
| `align/` | 已接入主流程的 bbox、ICP、DINOv3/SIFT 联合配准与评分；另保留独立配准工具 |

并发批量可用 `--postprocess-devices cuda:2 cuda:3` 在两张 GPU 上分别启动独立配准进程；每个进程独立持有 DINO、分割器和渲染状态。省略时配准仍为单 worker。API 并发与配准并发分别配置，已完成结果和已生成模型可继续复用。

调用链为 `src/run_pipeline.py → Pipeline.run → generate → postprocess → align.generation.align_generation`；并发批量的 `postprocess` 阶段调用同一个方法，并按评分将任务送回 Rodin 或 FAL 阶段重试。

当前生成参数、评分公式和重试诊断以本页及 [ALIGNMENT_RETRY.md](ALIGNMENT_RETRY.md) 为准。[src/README.md](src/README.md) 提供批量调度、恢复与 API 参数说明，[align/README.md](align/README.md) 描述配准模块及独立工具的使用范围。viewer 为可选工具，其 Three.js / Spark 浏览器依赖不随仓库提交，使用时需本地另行提供；主流程不需要这些浏览器依赖。
