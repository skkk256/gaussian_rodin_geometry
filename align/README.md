# 材质模型配准

本目录为后续配准保留的独立模块，当前第一步几何生成暂不接入，也不需要执行本目录命令。几何入口中的 `--bbox` 是测试用配准参数，第一步几何生成同样暂时不用。

独立执行已有流程：**完整逆变换 → 过滤高斯 FPS bbox → 八视角 DINOv3 / SIFT**。只优化等比缩放和平移，固定旋转；不调用生成或材质 API。

`transform.json` 按未包含 bbox / DINO 的完整逆变换读取。`rodin_textured.glb` 必须保持 Rodin condition 坐标，不能提前应用该矩阵。原始生成结果不被改写。

## 输入

单个物体使用生成流程的这两组文件：

```text
result/<scene>/<object>/
  rodin_textured.glb       # 后续材质步骤新增的文件
  transform.json          # 完整 condition → world 逆变换
  gaussian_filter.ply     # 过滤后的高斯，世界坐标

.work/<scene>/<object>/
  config.json             # ID、原始高斯路径和 hash
  canonical.json          # 生成时的采样坐标系
  condition.npz           # 实际上传的 FPS、归一化矩阵
  gaussian_filter.ply     # 生成时的过滤高斯，核对来源
```

**配准前保留对应 `.work`。** 从过滤高斯按原算法重新采集 4096 FPS，并核对实际上传记录；使用逆变换恢复同批 FPS 的世界坐标做 bbox，不使用高斯中心代替。读取记录也用于显示准确的 condition FPS 图层。

输入 GLB 的已验证结构为一个节点、一个 mesh、一个三角面 primitive，内嵌纹理和 `TEXCOORD_0`。保留模型内部节点变换；导出时合并配准矩阵，二进制几何、UV 和贴图数据保持原样。默认使用已有材质文件的 `emissiveTexture`。如果材质步骤输出的是 PBR `baseColorTexture`，显式加 `--texture-slot base-color`，不自动切换纹理来源。

## 运行

在 `gaussian_rodin/` 下执行，也可使用脚本绝对路径从任意目录调用：

```bash
/root/anaconda3/envs/sam3d-objects/bin/python -B align/run.py \
  --result result/scene_001/object_001 \
  --device cuda:1
```

默认从 `.work/scene_001/object_001` 读取采样记录。生成时用了其他工作目录则添加 `--work-dir /absolute/path/work/scene_001/object_001`。

批量直接复用生成流程的 manifest：

```bash
/root/anaconda3/envs/sam3d-objects/bin/python -B align/run.py \
  --manifest manifests/待制作物体_分割GS.json \
  --results-root result --work-root .work \
  --device cuda:1 --resume
```

每行使用 `scene_id` / `object_id` 定位文件。单物体缺少材质文件或发生程序错误会保存完整 traceback，批量继续处理其他物体，最后以非零状态退出。`--resume` 只复用输入、代码、权重入口、设置和输出 hash 均一致的完成结果；变化后去掉 `--resume` 重跑。

细长餐具可加 `--per-view-framing`，分别确定每个视角的取景距离。每个视角内高斯、网格、特征匹配及反投影始终使用同一相机。默认按全部视角所需范围统一取景；两种模式均包含高斯 3σ 椭球、逆变换模型、bbox 模型及优化允许范围，预留 28 px 边距。

## 输出

默认全部放在本文件夹下，可用 `--output` 修改根目录：

```text
align/output/
  index.html, viewer.js, overlay.js, manifest.json
  <scene>/<object>/
    final.glb                    # 最终模型，已经应用最终变换
    bbox.glb                     # 逆变换 + FPS bbox 模型
    transform.json               # condition → world 最终矩阵
    result.json                  # 各阶段矩阵、相机、指标、状态和用时
    stages.json, matches.json, correspondences.npz
    pointcloud-fps-world.json     # 同批 FPS 的世界坐标
    pointcloud-condition.json    # 实际上传的 condition FPS
    gaussian-<view>.png           # 八视角，view = 0/2/4/6/8/10/12/14
    inverse-<view>.png, bbox-<view>.png, final-<view>.png
    matches-<view>.png            # 绿线内点，红线剔除的对应
    original-gaussian.ply         # 原始高斯文件的链接
    filtered-gaussian.ply         # 过滤高斯文件的链接
    model.glb                    # 输入材质文件的链接
```

输出 `transform.json` 是原始 condition 模型到世界坐标的完整最终矩阵，column-major，与生成流程格式一致。输入模型已包含内部节点变换时，加载它后只应用一次该外部矩阵。**`final.glb` 和 `bbox.glb` 已经应用相应矩阵，不再叠加 `transform.json`。**

上下 ±30° 两层、每层四个方位共八视角参与特征配准；视角 1/5/9/13 仅计算轮廓指标。空视角和少于六个角点的视角可正常跳过，不中断其他视角。候选不足、共识不足或几何退化时，`finalStage=bbox`，最终模型和全部调试图片照常生成，报告及页面明确显示原因。程序异常保存为 `error.json`，不伪装成正常匹配不足。DINO 求解完成但检查指标下降时保留实际结果及诊断标记，供手动查看。

## Viewer 与共享依赖

```bash
python -m http.server 8910 --bind 0.0.0.0 --directory /absolute/path/gaussian_rodin
```

访问 `http://localhost:8910/align/output/`。八个图层为原始高斯、过滤高斯、未归一化 FPS、condition FPS、原始 Rodin、逆变换、逆变换加 bbox、最终结果。默认显示原始高斯和最终结果。四个模型阶段共享一次几何加载，贴图与调试图片按需加载，静止时停止连续渲染。

输出中的原始资产使用文件链接；移动生成结果或源高斯后需重新生成这些链接。前端库复用项目 `third_party/three` 和 `third_party/spark`，保留上游许可证，不依赖 CDN 或旧 viewer 服务。

配准使用共享的 `ckpt/dinov3` 链接，当前指向 `/root/Lab/EditLocalization/sk_release/ckpt/dinov3`；迁移时更新链接或传 `--weights`。Python 额外依赖见 `align/requirements.txt`：`transformers==5.2.0`、`nvdiffrast==0.4.0`，已在现有 `sam3d-objects` 环境安装。CUDA 扩展使用环境已有安装；如需构建，只能在测试容器的项目目录外进行。
