# video-color-constancy

## 题面

自动白平衡开启后，暖光场景反而更黄、整体亮度还忽高忽低：① `mean_rgb` 不排除过曝和死黑像素，高光把白点拉偏；② `gray_world_gains` 对绿通道也乘增益，整体亮度漂移；③ `gray_world_gains` 用增益和的算术平均归一，而不是把绿通道固定为 1；④ `kelvin_curve` 的色温方向反了，低 K 偏蓝；⑤ `estimate_kelvin` 用 R/B 比值但未取对数，色温区间被压缩；⑥ `apply_gains` 在 sRGB 域应用，色偏校正量非线性；⑦ `scene_correction` 先应用增益再估计，结果恒为无偏、自我实现；⑧ `validate` 不检查增益乘积是否约等于 1，整体曝光不守恒；⑨ `awb_batch` 复用第一帧均值，第二帧用第一帧统计；⑩ `kelvin_curve` 的插值端点不连续，跨段色温跳变。难点是闭环：统计→增益→应用必须严格有序，且绿通道是锚点。awb/estimate.py 是核心，tests/test_estimate.py 15个红。要修的是：统计前剔除过曝/死黑；灰世界增益把绿通道固定为 1；色温曲线方向与跨段连续；R/B 对数反推与 CCT 单调；线性光域施加增益；先估计后校正；校验三通道增益积；批量逐帧独立统计；让批量与逐帧单独结果一致；补齐 6200 K 中性锚点并满足全部 15 条断言。

## 运行方式

```powershell
cd D:\gsb-20261119\video-color-constancy
python -m pytest tests -q
```

初始状态输出恰好为 `15 failed`，没有 collection/import/syntax error；15 条失败都来自 `awb/estimate.py` 中 8 个桩的 `NotImplementedError`。修复核心模块后测试应全部转绿。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 音视频处理 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 核心契约

- 输入 RGB 是 `[0, 1]` 的线性光值；`apply_gains` 不做 sRGB gamma 变换。
- 中性色温锚点是 **6200 K**；该温度下 `kelvin_curve(6200.0) == (1.0, 1.0, 1.0)`。
- 灰世界增益和场景校正增益都把绿通道固定为 `1.0`，绿通道是整条链路的锚点。
- `kelvin_curve` 的 R/B 权重与温度成对数线性关系，低 K 偏暖、高 K 偏冷，且跨段连续。
- `estimate_kelvin` 对 R/B 取对数后反推 CCT，保证不同色温场景的估计单调。
- `scene_correction` 的顺序固定为“原始帧统计→估计色温→生成增益”，不得先校正再估计。
- 三通道增益乘积应接近 `1.0`（容差 `0.01`），避免白平衡改变整体曝光。
- `awb_batch` 每帧独立统计，缓存任何跨帧均值都会让后续帧串色。

## 约束

`tests/test_estimate.py` 是评测基准，**不可修改**；只允许修改 `awb/estimate.py`。实现只使用 Python 标准库（`math`）与 pytest，离线运行，不使用随机数。不要执行 `git init`、提交或推送。
