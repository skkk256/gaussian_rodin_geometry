# Gaussian → Rodin

独立项目，代码、第三方 Python 源码和权重入口都位于 `gaussian_rodin/` 内。不依赖旧 deploy、Docker、viewer 或内部 16127 服务。

流程固定为：**Gaussian PLY → 属性预过滤 → SAM3D 五视角朝向投票 → canonical 四视图 → Fal 清理补全 → 前景分割过滤高斯 → 高斯体积候选采样 → FPS 4096 → 独立点云归一化 → Rodin → 保存完整逆变换。**

默认只执行完整逆变换；`--bbox` 是测试用的额外配准参数，第一步几何生成暂时不用。`align/` 留给后续材质模型配准，当前阶段暂不接入，不执行 ICP、DINO/SIFT 配准，也不生成材质。后续纹理化若保持原始 Rodin 顶点坐标，可直接使用同一个 `transform.json` 恢复世界坐标。

## 环境与权重

本机可使用 `/root/anaconda3/envs/sam3d-objects/bin/python`，已验证的关键依赖见项目根目录 `requirements.txt`。PyTorch、gsplat、PyTorch3D 和 attention 扩展需要匹配 CUDA 环境；已有环境无需重新安装。

`third_party/` 包含本次流程使用的 SAM3D 与 DINOv2 Python 源码。SAM3D 不启用 `torch.compile`；本项目推理前冻结模型参数。渲染和 FPS 的原生扩展使用环境已有安装，首次运行可能在系统缓存目录构建扩展，不能将编译缓存设置到本项目目录。

权重不复制，通过 `ckpt/` 下的符号链接引用：

| 链接 | 当前目标 | 用途 |
|---|---|---|
| `ckpt/sam3d` | `/root/Lab/hyperhuman-cast/weights/s3o` | `pipeline_ssonly.yaml`、SS generator / decoder 配置和权重 |
| `ckpt/dinov2_vitl14_reg4_pretrain.pth` | `/root/Lab/hyperhuman-cast/model/dinov2/checkpoints/dinov2_vitl14_reg4_pretrain.pth` | SAM3D 条件编码器 |
| `ckpt/u2net.onnx` | `/root/.u2net/u2net.onnx` | 前景分割 |
| `ckpt/isnet-general-use.onnx` | `/root/.u2net/isnet-general-use.onnx` | 细节前景分割 |

迁移项目时更新这些链接即可；配置通过 `ckpt/sam3d` 加载，其相邻权重路径保持有效。DINOv2 在这里仅用于 SAM3D 朝向估计，不是后续图像配准。

凭据使用环境变量 `FAL_API_KEY`、`RODIN_API_KEY`，或用 `--credentials /private/keys.env` 加载私密 dotenv。仓库提供不含密钥的 `.env.template`，复制到 `.env` 后填写自己的 key，并传 `--credentials .env`。本机已有的 `.env.example` 是私密文件，和 `.env` 一样被 Git 忽略，不随仓库发布。不要把密钥放进 manifest 或结果目录。

## 单物体运行

以下命令在 `gaussian_rodin/` 目录运行；脚本也支持从其他工作目录使用绝对路径调用。

```bash
/root/anaconda3/envs/sam3d-objects/bin/python -B src/run_pipeline.py \
  --input /absolute/path/object.ply \
  --scene-id scene_001 --object-id object_001 \
  --output ./result --work-dir ./.work \
  --device cuda:1 --seed 22101 \
  --credentials /private/keys.env
```

输入固定为单物体、binary little-endian Gaussian PLY，属性为 float32：`x/y/z`、`opacity`、`scale_0..2`、`rot_0..3`（wxyz）、`f_dc_0..2`；支持 SH degree 0 或 3（后者另含 `f_rest_0..44`）。源坐标约定为 **−Y 向上、+X 向前、+Z 向右**。`opacity` 存 logit，`scale_*` 存 log scale。

`--device` 必须带 CUDA index；SAM3D、gsplat 与 FPS 共用这一个明确设备。默认 seed 为 `22101`，允许 `0..65535`。GPT 接口不发送 seed，Nano Banana 重试使用该物体的 seed；SAM3D 与候选采样 seed 固定为 `42`。

测试配准时，可在上述命令末尾添加 `--bbox`。此时 `transform.json` 直接保存「完整逆变换 + bbox 修正」的最终矩阵，加载原始 `rodin.glb` 后应用一次即可。第一步几何生成不添加该参数，保持只做逆变换。

## 批量运行

准备 JSON manifest，每行明确四个字段。相对输入路径相对于 manifest 所在目录解析：

```json
[
  {"scene_id": "scene_001", "object_id": "object_001", "input": "data/chair.ply", "seed": 22101},
  {"scene_id": "scene_001", "object_id": "object_002", "input": "data/table.ply", "seed": 22102},
  {"scene_id": "scene_002", "object_id": "object_001", "input": "data/sofa.ply", "seed": 22103}
]
```

```bash
/root/anaconda3/envs/sam3d-objects/bin/python -B src/run_batch.py \
  --manifest /absolute/path/cases.json \
  --output ./result --work-dir ./.work \
  --device cuda:1 --credentials /private/keys.env
```

批量按 manifest 顺序执行，复用同一进程中的 SAM3D、分割模型和 CUDA warmup；每个 `(scene_id, object_id)` 必须唯一。某项失败会打印错误，继续处理其他项，最后返回非零状态。重跑相同命令会复用已完成任务并继续已有请求。

批量命令同样支持 `--bbox`，对该批全部物体启用，无需修改 manifest。

### 本机「待制作物体_分割GS」批量运行

已准备 `manifests/待制作物体_分割GS.json`，包含 `/root/Lab/hyperhuman-cast/gaussian_rodin_standalone/data/待制作物体_分割GS` 直接目录下的 34 个 `.ply`，忽略其中的 `.glb`。`scene_id` 为 `待制作物体_分割GS`，`object_id` 为 PLY 文件名去掉扩展名，按文件名排序，Rodin seed 从 `22101` 逐项递增。

```bash
cd /root/Lab/gaussian_rodin
/root/anaconda3/envs/sam3d-objects/bin/python -B -u src/run_batch_concurrent.py \
  --manifest manifests/待制作物体_分割GS.json \
  --output result_geometry --work-dir .work_geometry \
  --credentials .env --device cuda:1 \
  --fal-workers 4 --rodin-workers 4 --max-inflight 16
```

这条命令只执行几何生成及完整逆变换，不启用测试用 bbox；`result_geometry` 和 `.work_geometry` 应选用尚未被其他参数占用的目录。本机历史 V1 测试使用过 `--bbox`，不能直接用此命令续跑那些工作目录。单个和批量脚本的默认输出根目录均为项目下的 `result/`。更换 prompt、重试策略或 bbox 设置必须使用不同的工作目录和结果目录；后续续跑保持命令不变。

### 四段流水并发

`run_batch_concurrent.py` 使用与单物体、串行批量相同的处理代码、参数和工作目录，可以直接续跑原来的串行任务：

| 流程 | 同时执行数 | 内容 |
|---|---|---|
| `preprocess_sam3d` | 1 | 读取、属性预过滤、SAM3D 五视角投票、原高斯四视图 |
| `fal` | `--fal-workers`，默认 2 | Fal 提交/等待/下载、前景分割过滤、过滤高斯渲染、FPS 与归一化 |
| `rodin` | `--rodin-workers`，默认 2 | Rodin 提交/等待/下载与原始 GLB 校验 |
| `postprocess` | 1 | 完整逆变换、可选 bbox、对齐 Mesh 渲染、八文件导出 |

四个流程处理不同物体时可以重叠。SAM3D 和 CPU 前景模型在同一进程内各常驻一份；每个 API 工作线程独享 HTTP session。共享 GPU 操作使用 FIFO 锁，避免同时修改模型状态、随机数状态及渲染资源；GPU 与云端等待、CPU 处理可重叠。前景分割模型也按 FIFO 串行使用。`--max-inflight 8` 限制进入流水线但尚未完成的物体数，其余物体未开始计时。

已有完成项校验 hash 后跳过；未完成项继续原 request ID，付费提交不重复发送。单项失败保留计时并继续其他项，最后返回非零。日志直接输出到终端 / tmux。`Ctrl+C` 停止接收新物体，已进入的物体完成后退出。`.work/batch-timing.json` 的 `totalSeconds` 累加本批已结束任务的处理耗时，包含失败尝试、不含跳过项；`wallSeconds` 单独记录整个批次实际经过时间。物体并发执行时，处理耗时之和可大于批次墙钟时间。

## 精简输出

每个物体最终有以下八个文件：

```text
result/
  scene_001/
    object_001/
      rodin.glb
      transform.json
      fal_refine.png
      gaussian.png
      gaussian_filter.png
      gaussian_filter.ply
      rodin.png
      timing.json
```

| 文件 | 内容 |
|---|---|
| `rodin.glb` | Rodin 下载的原始模型，字节不修改，保持 condition 坐标 |
| `transform.json` | Rodin condition → 原高斯世界坐标的最终矩阵；默认完整逆变换，`--bbox` 时包含 bbox 修正 |
| `fal_refine.png` | 通过高斯保留率检查的 GPT 或 Nano Banana 补全结果，2048×2048 四视图拼图；与过滤高斯及 Rodin 输入对应 |
| `gaussian.png` | 属性预过滤后、前景过滤前的高斯渲染；也是 Fal 的精确输入，2048×2048 四视图拼图 |
| `gaussian_filter.png` | 前景过滤后的高斯渲染；与 `gaussian.png` 使用完全相同的 frame、相机、分辨率与拼图顺序 |
| `rodin.png` | 原始 GLB 应用最终 `transform.json` 后的 Mesh 四视图；包含可选 bbox 修正，灰蓝色几何明暗显示，无材质 |
| `gaussian_filter.ply` | 过滤后的高斯，保留原世界坐标及源 PLY 属性顺序 |
| `timing.json` | 各次执行尝试、逐阶段处理与排队耗时、缓存状态、其他开销与累计处理耗时，单位为秒 |

四张图的四格顺序均为：左上 front、右上 right、左下 back、右下 left。每格 1024×1024，白背景，透视相机仰角 30°；各阶段使用同一组相机。图像接口和内存数据使用 RGB，OpenCV 编解码边界做颜色转换。

这些文件已足够后续使用原始模型和恢复矩阵，也足够独立浏览图片、耗时和 3D 对比。精确上传点云、canonical frame、请求状态及输入 hash 保存在独立的 `.work/<scene_id>/<object_id>/`，用于断点恢复和点击重跑，不属于交付输出；不导出对齐 GLB、诊断图或 pointmap。bbox 配准结果直接包含在 `transform.json` 中。清理工作目录后仍可查看结果，但不能从 viewer 重跑；重新生成需要新的结果目录。

同一工作目录要求源文件内容、ID、seed、prompt、API 参数和 `--bbox` 开关一致；修改时使用新工作目录和输出目录。付费生成的 POST 不自动重试。提交前写入状态，若网络中断导致提交状态不确定，明确报错，需检查服务端任务后处理该工作目录；不会悄悄创建第二次付费任务。已有 request ID 的任务可以继续轮询、下载。工作目录锁防止同时处理同一物体。

## 通用结果 viewer

在项目目录中，仅指定结果根目录即可启动；浏览器打开 `http://服务器地址:7880`：

```bash
/root/anaconda3/envs/sam3d-objects/bin/python -B -u src/viewer.py --results ./result
```

默认 `--work-dir` 为项目 `.work/`，`--credentials` 为项目 `.env.example`，`--device cuda:1`，`--host 0.0.0.0`，`--port 7880`。自定义输出位置时指定它对应的工作目录：

```bash
/root/anaconda3/envs/sam3d-objects/bin/python -B -u src/viewer.py \
  --results /absolute/path/result --work-dir /absolute/path/work \
  --credentials /private/keys.env --device cuda:1 --port 7880
```

viewer 直接扫描 `result/<scene_id>/<object_id>/`，无需旧版 `progress.json`、comparison manifest 或手工生成 index。只展示原高斯 / Fal / 过滤高斯 / 对齐 Mesh 四视图、模型及变换下载、`timing.json` 耗时和 3D 对比。支持搜索场景或物体；批量运行中点击刷新即可看到已完成项。3D 页面保留并排、叠加、单独查看及前后左右顶视角；对原始 GLB 应用 `transform.json`，与世界坐标中的过滤高斯保持同一变换基准。浏览只依赖八个结果文件，不需要 GPU、原输入或工作目录；重跑才需要这些资源。

点击「重新生成」提交一次新请求，范围可选「仅重跑 Rodin」或「重跑 Fal + Rodin」。前者复用 Fal、过滤高斯和精确条件点，后者复用原四视图与朝向并重做 Fal、过滤和采样。均保留原来的 bbox 开关和固定 API 参数，Rodin 使用一个新 seed。重跑串行执行，页面显示排队 / 运行 / 成功 / 失败及日志；完成后选择相应版本即可比较。

重跑不会覆盖首次结果；独立结果和工作目录保存为 `.viewer/<结果路径hash>/<run_id>/result/<scene_id>/<object_id>/` 与对应的 `work/`，每个结果仍是相同八文件结构，且有自己的 `timing.json`。viewer 的 HTML、JS 和本地 Three.js / Spark 依赖都随项目提供，不写进结果目录，也不通过网页暴露凭据或工作目录。

## 完整耗时记录

`timing.json` 在后端自动生成，viewer 直接读取显示，不从图片时间或请求日志估算。`totalSeconds` 是该物体在同一工作目录中的所有尝试的累计处理秒数，扣除本地排队；`attempts[-1].totalSeconds` 是最近一次处理耗时。每条新尝试另存 `wallSeconds`（实际经过时间）、`queueSeconds`（本地排队），满足 `totalSeconds = wallSeconds - queueSeconds`，并包含 `startedAt / finishedAt`（UTC）、`status`、`stages` 和 `overheadSeconds`。每阶段有 `name / seconds / status`：`completed` 实际执行，`cached` 复用已有结果，`skipped` 未启用 bbox，失败或中断明确标记。

记录完整覆盖：管线初始化、输入读取与属性预过滤、SAM3D 加载 / warmup / 五视角朝向估计、原四视图渲染、Fal 提交 / 轮询等待 / 下载 / 图片解码、分割模型加载与前景过滤、过滤后四视图渲染、体积采样 / FPS / 独立归一化、逆变换与恢复校验、Rodin 提交 / 轮询等待 / 下载 / GLB 校验、可选 bbox、变换文件写入、对齐 Mesh 渲染及结果导出。模型加载和 CUDA warmup 计入实际发生的阶段；同一批共享的管线初始化仅计入首个物体一次。配置校验、状态保存等未归入单阶段的开销放在 `overheadSeconds`，各阶段扣除本地排队后加该开销等于该次总处理时长；总时长计至结果导出结束，最终计时文件与完成状态的落盘不递归计时。

并发模式下，每次执行的 `stages` 正好对应上述四个流程，每项保存 `seconds / processingSeconds / queueSeconds / overheadSeconds / details`。`details` 保留实际子步骤耗时；`queueSeconds` 包括线程池等待、流程交接及 GPU / 分割器资源等待，`processingSeconds` 包括 API 请求的服务端等待和下载。原始阶段记录满足 `seconds = processingSeconds + queueSeconds + overheadSeconds`；物体 `totalSeconds = sum(seconds - queueSeconds) + 顶层 overheadSeconds`。串行模式同样扣除 GPU / 分割器等待。viewer 和阶段完成日志以扣除排队后的时间显示阶段耗时，排队仅单独展示；历史记录也按已有阶段明细计算显示，不改写结果文件。

给已有七文件结果补 Mesh 图，无需重新生成或调用 API：

```bash
/root/anaconda3/envs/sam3d-objects/bin/python -B -u src/backfill_mesh.py \
  --results ./result --work-dir ./.work --device cuda:1
```

仅补已完成且缺少 `rodin.png` 的物体，正在运行的锁定项跳过。原生成计时保留，追加 `kind=mesh_backfill` 的实际补图尝试，累计时间包含补图；原生成文件保持原样，更新计时和完成 hash。

失败时保留 `.work/<scene_id>/<object_id>/timing.json`；同一命令续跑会追加一次尝试并保留失败记录，成功后整体导出到结果目录。进程被强制结束时只能保留最后落盘的计时检查点，续跑标记为 `interrupted`，不将停机等待虚构为运行时长。已完成任务直接跳过时保持原计时不变。旧五文件结果没有真实计时记录，需重新运行新版流程，viewer 不补造历史耗时。

## 处理步骤

1. **属性预过滤。** 去掉非有限记录、opacity `<0.01`、协方差 3σ 半径超过稳健物体对角线四倍，以及离主体极远的记录。各向异性只统计，不作为删除条件，保留桌面等扁平高斯。
2. **朝向。** 在水平面求稳健最小面积 OBB，按源坐标约定锚定轴向。从方位角 `0/72/144/216/288°`、仰角 `30°` 渲染五张 512×512 图；每视角用 SAM3D stage1 coarse（10 steps、seed 42）投一票，选水平 quarter turn。平票按 coarse / Gaussian 体素的 Chamfer 最小值决定。体素仅参与朝向判断，不上传 Rodin。保存 canonical→world 矩阵 **P**。
3. **四视图与 Fal。** 按确定的 canonical 朝向渲染四视图，拼成 2×2 图，请求下面列出的 Fal 模型。通用 prompt 完整保存在 `fal_prompt.txt`：清理扫描杂质、补全缺失，保持物体身份、数量、真实结构、材质和各格相机投影一致，不添加文字或边框。
4. **前景与高斯过滤。** Fal 每格同时运行 U2Net 320×320 和 ISNet 1024×1024，取前景概率逐像素最大值。使用 `alpha >= 0.25`，二值膨胀 4 次。将高斯中心投影到四张图，**任意一个有效投影在 mask 外即删除**；相机背后或图像外的无效投影不投票。以属性预过滤后的高斯数为分母，GPT 过滤后保留不足 30%（删除超过 70%，包括全部删除）时，明确记录并换 `google/nano-banana-2.1/edit` 重试一次。重试使用原始 `gaussian.png`、相同的当前默认 prompt、原始待过滤高斯和相机；不对 GPT 结果二次编辑或继续裁减。Nano 也必须保留至少 30%，否则明确失败，不提交 Rodin。不使用 IoU 门槛、连通域清理或回滚，保留记录不移动。工作目录 `foreground_filter.json` 保存两次的数量、保留率、request ID 和选择结果；`fal_gpt/`、`fal_nano/` 分别保存图片和请求日志，续跑复用原请求。
5. **采样。** 从过滤后 opacity `>0.2` 的高斯随机等概率选择 65,536 次，在各自 `opacity × exp(-0.5 × Mahalanobis²) >= 0.2` 椭球体积内采样。seed 42。转到 canonical 坐标，保留 XYZ 绝对值均 `<0.98` 的候选，使用确定性 greedy GPU FPS（固定首点）取 4096 点。这是体积点，可能含内部点。
6. **独立点云归一化。** 根据实际 4096 点自己的 AABB 中心 `c` 和最长边 `L`，计算 `condition=(canonical−c)×1.96/L`，保存 canonical→condition 矩阵 **N**。只做等比缩放和中心平移；最长轴范围 `[-0.98,0.98]`，其他轴保持比例。不能以高斯渲染归一化代替这一步。
7. **Rodin。** 上传 Fal 四张图和这批归一化的 4096×3 点，下载原始 GLB。模型参数固定如下。
8. **完整逆变换。** 计算 **`M_inverse = P × inverse(N)`**，包括点云归一化的反向缩放和平移，以及 canonical 的轴向、朝向、缩放和平移。内部校验该矩阵能把实际上传点云恢复为原始世界坐标的 FPS 点。默认保存 `rodinToWorld = M_inverse`。
9. **可选 bbox 配准（`--bbox`）。** 从原始 GLB 表面按面积确定性采样 8000 点，seed 为 0，包含 GLB 内部节点变换。先应用 `M_inverse`，然后在**过滤后高斯的语义 OBB 轴系**中，将模型与**同一批未归一化的 4096 个 FPS 世界坐标点**比较。两侧逐轴取 `0.25% / 99.75%` 分位范围，用最长边比计算一个等比缩放，并将范围中心对齐；固定旋转，不搜索朝向，不做逐轴非等比缩放。把 OBB 内修正转换为世界坐标修正 **C**，最终保存 **`rodinToWorld = C × M_inverse`**。原始 GLB、高斯与三张输入/过滤图片保持原样；`rodin.png` 使用 bbox 后的最终矩阵。

`transform.json` 的矩阵是 glTF / Three.js 使用的 column-major 列表：

```text
{"matrixOrder": "column-major", "rodinToWorld": {"matrix": [16个数]}}
```

应用方式（列向量约定）：`[world_x,world_y,world_z,1]ᵀ = M × [rodin_x,rodin_y,rodin_z,1]ᵀ`。NumPy 使用 `M=np.asarray(data['rodinToWorld']['matrix']).reshape(4,4,order='F')`；一批行向量顶点使用 `vertices @ M[:3,:3].T + M[:3,3]`。两种模式均只应用一次；`--bbox` 时矩阵已包含完整逆变换，不应再叠加另一份逆矩阵。不额外翻轴或重新归一化。bbox 配准对齐稳健范围，不保证生成几何与高斯逐表面一致。

## Fal 完整参数

默认使用 **GPT Image 2.6 Sunburst edit**，精确模型 ID 为 **`openai/gpt-image-2.6/sunburst/edit`**。`src/fal_prompt.txt` 与 `prompts/fal_v1.en.txt` 完全一致，默认使用当前用户修订版 V1。

| 字段 | 值 |
|---|---|
| 提交地址 | `POST https://queue.fal.run/openai/gpt-image-2.6/sunburst/edit` |
| `prompt` | `src/fal_prompt.txt` 全文，原样发送 |
| `image_urls` | 一张 `gaussian.png` 的 PNG data URL |
| `num_images` | `1` |
| `image_size` | `{"width":2048,"height":2048}` |
| `quality` | `high` |
| `reasoning_effort` | `high` |
| `output_format` | `png` |
| `background` | `opaque` |
| seed | 不发送；该接口没有沿用 Rodin seed |

提交 HTTP 超时 180 秒，状态 GET 超时 60 秒，每 5 秒轮询，单次等待最多 60 分钟；可续跑已有任务。

仅当 GPT 高斯保留率不足 30% 时，使用 `POST https://queue.fal.run/google/nano-banana-2.1/edit` 重试一次：`num_images=1`、`aspect_ratio=1:1`、`resolution=2K`、`output_format=png`、`thinking_level=high`、`seed` 为该物体的 manifest/命令行 seed；prompt 和输入图片与 GPT 完全相同。两次请求各有独立日志，最终仅将通过检查的图片和对应过滤高斯用于后续流程。当前已有 34 个结果中，叉子删除 99.70%、餐刀删除 88.73%，其他物体最高删除 47.31%，因此 70% 阈值只会标记这两个已有异常案例；重新生成后的触发情况由新结果决定。

## Rodin 完整参数

使用公网 **`https://api.hyper3d.com/api/v2`**，multipart `POST /rodin`；保存返回 UUID 和 subscription key，再 `POST /status`、`POST /download` 获取原始 GLB。

| 字段 | 值 |
|---|---|
| `images` | 四个独立 PNG，按四视图拼图的逐行顺序上传 |
| `image_label` | `["F","L","B","R"]` |
| `tier` | `Gen-2.5-Medium` |
| `quality` | `medium` |
| `mesh_mode` | `Raw` |
| `geometry_file_format` | `glb` |
| `material` | 字符串 `None` |
| `use_original_alpha` | 字符串 `true` |
| `is_symmetric` | `unknown` |
| `pcd_condition_cfg` | 字符串 `true` |
| `pcd_condition_weight` | `0.3` |
| `pcd_condition` | 实际归一化后的 `4096×3` XYZ JSON 列表 |
| `seed` | 单物体 `--seed`，默认 22101；批量 manifest 每行的 `seed` |

`image_label` 按 Rodin 物体局部方向定义；图中右上是相机位于 +X 的一侧，对应标签 L，右下相机位于 −X，对应 R。不交换图片，也不将标签改成 F/R/B/L。不发送内部服务的 `normalize`、`point_count`、`pcd_tokens`、`cfg`、`steps` 或 `cutoff` 参数。

生成 POST HTTP 超时 600 秒；状态 / 下载地址查询超时 180 秒，文件下载超时 300 秒。每 5 秒轮询，单次等待最多 90 分钟；超时后用同一命令继续已有任务。
