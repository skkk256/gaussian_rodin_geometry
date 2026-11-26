# FAL prompt

生产流水线实际发送 [src/fal_prompt.txt](../src/fal_prompt.txt) 全文。GPT Image 与 Nano Banana 重试使用相同 prompt，运行时不会从本目录选择版本。

当前输入为三行两列六视图：

```text
front  right
back   left
top    bottom
```

生成输入的 front/right/left 俯视 30°，back 平视 0°，top/bottom 为垂直视图。prompt 要求保留每格实际投影、位置、尺度和物体身份，清理污染并补全缺失结构。Rodin 上传索引 `[0,1,2,3,5]`，底面取右下角，标签为 `[F,L,B,R,D]`。

DINO 配准与评分独立从高斯渲染四侧均 30° 的六视图，不使用 FAL 图片，相关设置见[配准与重试说明](../ALIGNMENT_RETRY.md)。

## 版本参考

本目录的 `fal_v1` 至 `fal_v6` 中英文文件保存不同保色要求，`fal_six_views.en.txt` 与 `.zh.txt` 保存六视图版本参考。这些文件不保证与生产 prompt 一致；修改运行行为应更新 `src/fal_prompt.txt`，并使用新的工作目录。
