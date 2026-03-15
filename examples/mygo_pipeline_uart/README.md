# mygo_pipeline_uart

纯 C++ 的目标追踪管线示例（无遗传算法），支持串口输出云台指令。

## 功能

- 使用 `TargetTrackingPipeline` 进行目标/激光闭环追踪
- 启动后自动进入正弦扫描（`yaw` 为 x 轴、`pitch` 为 y 轴）
- 扫描中持续检测目标，稳定锁定 ROI 后自动进入追踪
- 将控制命令通过串口发送到云台控制器

## 参数

- `--width <int>`：相机宽度（默认 `640`）
- `--height <int>`：相机高度（默认 `480`）
- `--uart <device>`：串口设备（默认 MaixCAM2 使用 `/dev/ttyS4`）
- `--baud <int>`：串口波特率（默认 `115200`）
- `--no-uart`：关闭串口输出

> 说明：已移除空格键触发扫描/追踪逻辑，运行后自动扫描并自动进入追踪。

## Pinmux

- MaixCAM2 默认：`A21 -> UART4_TX`, `A22 -> UART4_RX`（`/dev/ttyS4`）
- 其他平台默认回退：`A16 -> UART0_TX`, `A17 -> UART0_RX`（`/dev/ttyS0`）

> 如果你使用了其他 UART 设备，请按硬件连接自行设置 pin function。
