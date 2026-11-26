# 第一阶段材质、配准与重试

单物体、串行批量和并发批量均在第一阶段材质生成后自动执行配准、评分与重试。

## 生成输入

- 六视图顺序仍为 front、right、back、left、top、bottom，拼图为三行两列。
- front、right、left 的俯视角为 30°；back 为 0°；top、bottom 为 ±90°。
- Rodin 上传索引 `[0,1,2,3,5]`，标签 `[F,L,B,R,D]`。底面是右下角的第 5 格，不上传左下角 top。
- 几何保持 `Gen-2.5-Medium / medium / Raw`，同时请求 `material=Shaded`、`texture_mode=extreme-low`，在该模型上配准。纹理渲染读取 GLB 实际声明的有效 emissive 或 base-color 纹理。

参数取值参考 [Rodin Gen-2.5 API](https://docs.hyper3d.ai/en/api-specification/rodin-gen2-5)。

## 评分与配准

总分（0–100）沿用会话 `01a21703-21fa-7d31-9a7c-31f11773f25f` 最后的权重：

`0.4 × 分块高斯→mesh + 0.5 × 六视角平均 mask IoU + 0.1 × 原始 mesh→高斯`

固定参考是本轮过滤高斯的体积 FPS4096。有点云条件时直接恢复本轮实际上传的同批 FPS，不能使用高斯中心替代。距离用这些世界坐标点的 AABB 对角线 L 归一化，容差为 0.02 L。

- 分块正向：高斯点到精确 mesh 三角面的距离；格宽 0.05 L；每格取最近 ceil(20% × 点数) 的距离中位数；保留误差最小的 ceil(80% × 占用格数)；分数为保留格中误差 ≤0.02 L 的格数比例。格子等权。
- 原始反向：mesh 表面按面积采样 65536 点，seed42，FPS4096；计算到固定高斯 FPS 的最近点距离；分数为距离 ≤0.02 L 的点数比例。
- mask：与 DINO 共用六视图相机（四侧均 30°，顶底 ±90°）渲染过滤高斯和 mesh，以 alpha>0.5 二值化，逐视角 IoU 等权平均。双空视角不参与平均，不膨胀、不腐蚀、不删除小块。评分 mask 与 DINO 优化中的轮廓处理相互独立。

每次生成依次执行：

1. 完整逆变换后做 bbox：0.25%–99.75% 分位边界、三轴归一化尺寸最小二乘、统一缩放和平移，固定旋转。
2. bbox 总分 <70 时，从 bbox 启动可见表面 ICP；三层体素、互为最近邻、Tukey 权重、空间均衡；固定旋转，只优化缩放和平移。ICP 没有足够对应时明确记录状态，不伪装成成功。
3. 在 bbox、ICP 中分数较高的基础上运行六视图 DINOv3/SIFT + 轮廓 + 姿态先验联合配准。配准和评分的四个侧面均为 30°，顶、底为 ±90°；背面重新从高斯渲染为 30°。参考图沿用旧版路由：前／右／左逐张检查 FAL 漂移，三张全部通过才一起使用 FAL；顶／底各自判断；背面固定使用 30° 原高斯，排除 0° FAL 背面。漂移检测沿用 512 特征及旧阈值（至少 20 内点、内点率 ≥35%、覆盖率 ≥12%、尺度偏差 ≤10%、旋转 ≤5°、中心位移 ≤20 像素），不变形 FAL 图片。完整 RGB 用于特征提取，再以 mask 筛选匹配点；FAL 使用完整分割轮廓，高斯使用过滤高斯局部轮廓。轮廓不足 64 像素时屏蔽该视角特征与轮廓约束。评分始终参考过滤高斯。特征提取保留 `align/matching.py` 的 512/768 两尺度。沿用 13 起点轮廓初始化、最多三轮对应刷新；相对起点缩放限 0.8–1.2、各轴平移限 ±0.15 个归一化单位、旋转限 30°。
4. 用同一套六视图相机和总分评估 DINO。DINO 比 bbox/ICP 最佳结果下降不超过 5 分时采用 DINO（包括同分和恰好下降 5 分）；超过 5 分才保留之前候选。以实际选中结果的得分判断是否 ≥70；不足时继续重试，不将原候选得分冒充 DINO 得分。
5. 仍 <70 时重生成。重试耗尽后保留全部轮次的最高分；低分结果明确标记 `below_threshold`。

第一阶段保留既有仅图像生成分支。有可采样过滤高斯时独立采样用于配准评分，不追加点云到仅图像 Rodin 请求；没有 opacity>0.2 的过滤高斯时标记 `unscorable`、分数 null 并进入重试，不冒充达标。

## 运行

单物体、串行批量、并发批量入口均支持 `--max-retry`，默认 1，表示初次生成之外最多额外尝试一次。

```bash
python -B -u src/run_pipeline.py \
  --input /absolute/path/object.ply --scene-id scene --object-id object \
  --output ./result --work-dir ./.work_aligned \
  --credentials .env --device cuda:1 --max-retry 1
```

重试 1 只重跑 Rodin，复用本轮 Fal、过滤高斯与点云条件；重试 2 从 Fal 开始；更大次数按相同两次一组交替。Rodin seed 为 `(原 seed + 尝试序号) % 65536`。重新执行相同命令复用已保存请求；更改相机、prompt、材质或重试参数必须使用新的 work/result 目录。

## 记录与输出

- `.work/<scene>/<object>/refinements/000/`：初次 Fal、过滤和点云条件；`001/` 是第二次重试重跑 Fal 的产物。
- `attempts/000/`：初次 Rodin 请求、原始模型及配准诊断；后续尝试独立存放在 `001/` 等目录。
- 每轮 `alignment.json` 保存 bbox/ICP/DINO 分数、矩阵、求解状态、参考路由和来源 hash；DINO 另存 iterations、orientation-search 和 correspondences。
- 工作目录根部 `alignment.json` 保存跨轮选优。最终八个交付文件保持原有文件名，模型、过滤高斯、Fal 图片和 condition 均来自被选中的同一轮。
- `transform.json` 的 `alignmentScore`、`selectedAttempt`、`finalStage` 和 `alignmentReport` 提供最终总分与各轮各阶段分项。原始 `rodin.glb` 不烘焙变换，使用 `rodinToWorld` 一次恢复世界坐标。

实现位于 [align/generation.py](align/generation.py)、[align/icp.py](align/icp.py)、[align/iterative.py](align/iterative.py) 和 [align/scoring.py](align/scoring.py)，不依赖实验目录。
