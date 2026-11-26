# 配准模块

主流程已通过 `src/pipeline.py` 接入本目录，在每次 Rodin 生成后自动完成 bbox、条件 ICP、六视图 DINOv3/SIFT 与轮廓联合配准，以及评分和重试。运行入口见[项目 README](../README.md)，指标公式、相机和阈值见[配准与重试说明](../ALIGNMENT_RETRY.md)。

## 主流程实现

| 模块 | 职责 |
|---|---|
| `generation.py` | 每次生成的候选调度、评分与 DINO 选择 |
| `geometry.py` | GLB 读取、完整逆变换后的固定旋转 bbox 配准 |
| `icp.py` | 可见表面 ICP，优化统一缩放和平移 |
| `reference.py` | 六视图共享相机：四侧 30°、顶底 ±90° |
| `routing.py` | FAL 漂移检查、侧面共同选择与顶底独立选择，背面固定高斯 |
| `rendering.py` | 高斯与 mesh 在共享相机下渲染 |
| `matching.py` | DINOv3 512/768 双尺度特征与 SIFT 对应 |
| `iterative.py` | 联合轮廓、鲁棒特征重投影与姿态先验，最多三轮刷新对应 |
| `rotation_math.py`、`contour_math.py` | 有界相似变换与可微轮廓优化 |
| `scoring.py` | 分块正向距离、六视图 IoU、原始反向距离 |

bbox 使用网格采样点和过滤高斯 FPS 的 0.25%–99.75% 分位边界。统一缩放同时最小化三个方向的归一化尺寸误差，各方向归一化分母下限为目标最长尺寸的 5%；固定旋转并对齐中心，不做逐轴拉伸。

DINO 使用六张完整 RGB，以 mask 筛选特征点。前／右／左全部通过 FAL 漂移检查才一起用 FAL；顶／底各自判断；背面固定使用 30° 高斯，排除 0° FAL 背面。FAL 使用完整分割轮廓，高斯使用过滤高斯局部轮廓；轮廓不足 64 像素时屏蔽该视角特征和轮廓约束。配准和评分使用相同相机，评分始终参考过滤高斯。DINO 比 bbox/ICP 最佳结果下降不超过 5 分时采用，最终得分不足 70 时由主流程重试。

每轮诊断保存在工作目录 `attempts/<序号>/`，包括 `alignment.json`、`matches.json`、`correspondences.npz`、`iterations.json` 与朝向初始化记录。交付的原始 `rodin.glb` 保持生成坐标，只应用一次最终 `transform.json` 中的 `rodinToWorld`。

## 独立配准工具

`run.py` 与 `pipeline.py` 提供独立的八视角、固定旋转配准工具，不参与上述自动六视图联合流程。它要求 `rodin_textured.glb`、尚未叠加任何配准修正的完整 condition→world 逆变换，以及同批 `condition.npz` 和过滤高斯记录；不能把自动流程的最终矩阵当作原始逆变换再次输入。

此工具不调用生成 API，也不执行自动评分重试。输入 GLB 须保留 Rodin condition 坐标；默认读取 emissive 纹理，base-color 纹理需显式指定 `--texture-slot base-color`。仅图像生成缺少上传点云记录，不能直接使用该工具。参数见 `python -B align/run.py --help`。

独立工具输出的 `final.glb`、`bbox.glb` 已烘焙相应变换，不能再次叠加外部矩阵；这与主流程交付原始 GLB 的约定不同。

## 依赖

权重使用项目 `ckpt/dinov3/`，由 Transformers 从本地配置与权重加载。Python 和原生依赖见 [requirements.txt](../requirements.txt)；CUDA 扩展须与 PyTorch/CUDA 匹配，构建与缓存位于项目目录外。配准计算不需要 viewer 浏览器依赖。
