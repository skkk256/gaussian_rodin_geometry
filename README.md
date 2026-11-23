# Gaussian Rodin Geometry

输入一个物体的 Gaussian PLY，输出 Rodin 几何模型、恢复到原始世界坐标的变换矩阵，以及各阶段四视图和过滤后的高斯。

当前为**第一步几何生成**，不生成材质，也不接入后续精配准。`align/` 是为后续材质模型配准保留的独立模块，当前流程不用运行它。`--bbox` 是测试用的额外配准参数，第一步几何生成暂时不用，下面的运行命令不传该参数。默认仍会保存几何生成所需的完整坐标逆变换。

## 流程

```text
Gaussian PLY
  → 属性预过滤
  → SAM3D 五视角朝向估计
  → 高斯四视图渲染
  → Fal 清理与补全四视图
  → 前景分割与高斯过滤
  → 高斯体积采样、FPS 4096 与独立点云归一化
  → Rodin 几何生成
  → 保存完整逆变换与结果
```

Fal 默认使用 `openai/gpt-image-2.6/sunburst/edit`，prompt 为 [src/fal_prompt.txt](src/fal_prompt.txt)，与当前用户修订版 [V1 英文](prompts/fal_v1.en.txt) 一致，[中文对照](prompts/fal_v1.zh.txt) 用于阅读和修订。

以属性预过滤后的高斯数量为基准，如果 GPT 输出导致高斯删除超过 70%，自动改用 `google/nano-banana-2.1/edit` 最多尝试两次。两次均使用相同原始输入图、prompt 和相机，seed 分别为该物体的 seed 与 `(seed + 1) % 65536`；任一次保留至少 30% 即使用对应点云继续。两次 Nano 均保留不足 30%（包括全部删空）时，使用第一次 GPT Image 的四视图继续生成 Rodin，仅上传图片，省略 `pcd_condition`、`pcd_condition_cfg`、`pcd_condition_weight`，跳过 FPS、点云归一化和测试用 bbox 配准。选择结果与原因保存在工作目录日志中。

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
  --output ./result --work-dir ./.work \
  --credentials .env --device cuda:1 --seed 22101
```

此命令执行完整的单物体几何生成，**不启用 bbox 配准，也不运行 `align/`**。`--device` 必须包含设备编号，例如 `cuda:0` 或 `cuda:1`。`scene-id` 和 `object-id` 决定输出子目录，seed 范围为 `0..65535`。

输入为 binary little-endian Gaussian PLY，属性使用 float32：`x/y/z`、`opacity`、`scale_0..2`、`rot_0..3`（wxyz）、`f_dc_0..2`，支持 SH degree 0 或 3。`opacity` 存 logit，`scale_*` 存 log scale；源坐标约定为 −Y 向上、+X 向前、+Z 向右。

## 输出目录

上述命令的结果位于 **`result/scene_001/object_001/`**：

```text
result/
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
| `rodin.glb` | Rodin 返回的原始几何模型，保持生成坐标，尚未烘焙世界坐标变换 |
| `transform.json` | 有点云时保存 `rodinToWorld` 完整逆变换；仅图像模式保存 canonical 坐标系的粗放置矩阵，并标记 `imageOnly=true`、`alignment=canonical-frame-only`，需要后续配准 |
| `gaussian.png` | 前景过滤前的高斯四视图，也是 Fal 的原始输入 |
| `fal_refine.png` | 最终采用的 GPT 或 Nano Banana 补全四视图 |
| `gaussian_filter.ply` | 实际过滤后的高斯，保留原世界坐标和源属性；仅图像模式保存第二次 Nano 的过滤结果，可能为 0 点 |
| `gaussian_filter.png` | 过滤高斯四视图，与原图共用相机；0 点时为空白白底图 |
| `rodin.png` | 原始模型应用保存矩阵后的四视图，无材质；仅图像模式的放置不代表精确配准 |
| `timing.json` | 各阶段处理耗时、排队及缓存记录 |

四视图拼图为 2048×2048，顺序是左上 front、右上 right、左下 back、右下 left。`transform.json` 中矩阵按 column-major 存储，加载 `rodin.glb` 后应用一次即可；不要重复归一化或重复应用逆变换。仅图像模式没有点云条件约束，保存矩阵仅用于按原始 canonical 坐标系粗放置，不保证生成模型的世界坐标尺度和位置准确。

工作缓存保存在 **`.work/scene_001/object_001/`**，包含朝向、实际上传点云（有点云模式）、请求日志和高斯保留率记录，用于断点续跑和后续配准。仅图像模式另存 `image-only.json`，不生成 `condition.npz`。相同命令可继续已有请求；更改输入、prompt、seed 或参数时应使用新的工作目录与结果目录。

## 代码目录与后续配准

| 目录 | 用途 |
|---|---|
| `src/` | 当前几何生成流水线、单物体与批量入口、结果 viewer |
| `prompts/` | Fal prompt 的中英文版本及修订说明 |
| `manifests/` | 本机批量示例；示例中的输入路径需按实际数据修改 |
| `third_party/` | 几何流程使用的 SAM3D、DINOv2 源码；不包含 viewer 浏览器依赖 |
| `align/` | 后续材质模型的 bbox 与 DINOv3 / SIFT 配准，独立运行，当前几何阶段暂不接入 |

`src/run_pipeline.py --bbox` 仅用于测试几何生成后的 bbox 等比缩放与平移配准，默认关闭。第一步几何输出使用上面的无 `--bbox` 命令；后续配准保留原始 `rodin.glb`、`transform.json`、过滤高斯及对应 `.work/` 即可。

批量入口、viewer、完整 API 参数及变换说明见 [src/README.md](src/README.md)。viewer 为可选工具，其 Three.js / Spark 浏览器依赖不随仓库提交，使用时需本地另行提供；单物体几何生成不需要这些依赖。后续配准的独立输入输出约定见 [align/README.md](align/README.md)。
