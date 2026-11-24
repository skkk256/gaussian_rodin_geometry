# Fal prompt 版本

- V1：当前用户修订版。第一段删除“严格保留输入中显示的原物体颜色”和“不要提亮、增白、漂白、重新着色或改变物体的白平衡”及对应英文句子；第三段末尾保留“主体物体的位置保持和原图片完全对齐”，其余段落不变。
- V2：在 V1 第三段末尾增加“主体物体的位置保持和原图片完全对齐”。其他内容一致。
- V3：在 V2 全文结尾增加“保留不同面及不同材质之间真实的色差”。其他内容一致。
- V4：在 V2 第一段结尾增加“保留不同面及不同材质之间真实的色差”。其他内容一致。
- V5：替换 V4 第一段中的保色要求，以各视图清晰、未受污染区域的可见颜色和明暗为基准，保留色相、饱和度、明度和物体自身明暗，禁止提亮较暗表面、重新打光、补光及统一各面亮度。其余三段不变。
- V6：删除 V5 第一段中的“以每个视图中清晰、未受污染的物体区域为颜色和明暗基准。”及对应英文，其余全文不变。

每版保存 `fal_vN.en.txt`（实际发送英文）和 `fal_vN.zh.txt`（中文对照）。V2–V6 为历史版本存档；V1 为用户当前修订版。2026-11-25 起生产流水线默认 `src/fal_prompt.txt` 使用 `fal_six_views.en.txt`：沿用当前 V1 的修复要求，改为三行两列的六视图说明，明确顶视图从正上方垂直向下、底视图从正下方垂直向上拍摄。前四张保持原 30° 俯视角。对应中文为 `fal_six_views.zh.txt`。GPT 与 Nano Banana 重试均原样使用这份 prompt；Rodin 仅上传前四张和底视图，标签 F/L/B/R/D。

修订前 V1 已测试 G013、G027：`/root/Lab/hyperhuman-cast/temp/g013_fal_preserve_color_short`、`/root/Lab/hyperhuman-cast/temp/g027_fal_preserve_color_short`；实验目录保留当时实际使用的 prompt。

2026-11-23 本次修订前的 34 个物体整批结果归档为 `result_v1_old/`，对应工作目录为 `.work_v1_old/`；修订后的整批使用 `result_v1/` 与 `.work_v1/`。两批沿用相同原始朝向、输入高斯四视图、seed 和 bbox 设置，重新执行 Fal、过滤、Rodin 及后处理。

随后再次删除第一段的禁止提亮等要求，上一轮中断结果归档为 `result_v1_stopped_20261123_2312/`，工作目录为 `.work_v1_stopped_20261123_2312/`；当前 `result_v1/` 与 `.work_v1/` 留给最新修订版，付费生成结果不复用。

V2 实验：`/root/Lab/hyperhuman-cast/temp/fal_prompt_v2_repeat3`，四个物体各三次独立请求；叉子、餐刀逐次用对应 Fal 输出重新执行前景过滤和高斯渲染。

V3 实验：`/root/Lab/hyperhuman-cast/temp/g013_fal_v3_color_difference_repeat3`，G013 书桌上的书三次独立请求，复用 V2 的精确原始输入及 API 参数。

V4 实验：`/root/Lab/hyperhuman-cast/temp/fal_v4_first_paragraph_gpt_nano`，原 GPT 模型对 G013 书桌上的书、G027 绿色餐垫各请求三次；`google/nano-banana-2.1/edit` 对叉子、餐刀分别使用 seed 22101、22102、22103 请求，逐次执行前景过滤及高斯渲染。

V5 实验：`/root/Lab/hyperhuman-cast/temp/fal_v5_observed_color_repeat3`，G013 书桌上的书、G027 绿色餐垫各三次独立请求，与 V4 使用相同原始输入和 GPT 模型参数。

V6 实验：`/root/Lab/hyperhuman-cast/temp/fal_v6_without_color_reference_repeat3`，G013 书桌上的书、G027 绿色餐垫各三次独立请求，与 V5 使用相同原始输入和 GPT 模型参数。
