# mygo_pipeline_uart

纯 C++ 的目标追踪管线示例（无遗传算法），支持通过 TCP 接收上位机控制命令，并将视觉伺服指令回传给 ROS2；同时保留本地 UART/蓝牙串口直连能力。

## 功能

- 使用 `TargetTrackingPipeline` 进行目标/激光闭环追踪
- 程序启动后常驻运行 TCP 服务端，默认监听 `5555`
- 上位机可通过 TCP 命令控制视觉任务 `START/STOP`
- 运行中返回状态字段 `vision_app_state / vision_track_state / latest_servo_command`
- 仍可选用本地 UART 或蓝牙串口直连输出控制帧

## 参数

- `--width <int>`：相机宽度（默认 `640`）
- `--height <int>`：相机高度（默认 `480`）
- `--uart <device>`：串口设备（默认 MaixCAM2 使用 `/dev/ttyS4`）
- `--bt-rfcomm <device>`：蓝牙串口设备，例如 `/dev/rfcomm0`
- `--baud <int>`：串口波特率（默认 `115200`）
- `--tcp-port <int>`：TCP 服务端监听端口（默认 `5555`）
- `--no-uart`：关闭串口输出

> 说明：当前默认行为是启动后等待上位机 `START` 命令，不会自动进入扫描。

## TCP 控制

- 默认启动后处于 `IDLE/STOPPED`，等待上位机命令。
- 上位机发送 `APP_CMD_VISION_START(0x10)` 后，管线进入搜索并在锁定后自动进入追踪。
- 上位机发送 `APP_CMD_VISION_STOP(0x11)` 后，管线停止识别，但应用进程保持运行。
- 上位机轮询 `APP_CMD_VISION_STATUS(0x12)` 可获取：
  - `vision_app_state`
  - `vision_track_state`
  - `active`
  - `target_found`
  - `laser_found`
  - `latest_servo_command`

## ROS2 对接

- ROS2 `camera_serial_node` 通过 TCP 连接本应用，并将状态发布到：
  - `/status/camera/connection`
  - `/status/camera/vision_app_state`
  - `/status/camera/vision_track_state`
  - `/status/camera/protocol`
- 视觉侧返回的原始云台帧，如 `{#000P1500T0100!#003P1500T0100!}`，会由 ROS2 通信节点翻译后发布到 `/cmd/arm/direct_pwm_command`。

## 蓝牙转发

- 当输出设备使用 `/dev/rfcomm*` 时，程序会按蓝牙串口处理，不再配置 UART pinmux。
- 推荐仅在不走 TCP 控制链路时使用蓝牙转发。

## Pinmux

- MaixCAM2 默认：`A21 -> UART4_TX`, `A22 -> UART4_RX`（`/dev/ttyS4`）
- 其他平台默认回退：`A16 -> UART0_TX`, `A17 -> UART0_RX`（`/dev/ttyS0`）

> 如果你使用的是蓝牙串口 `/dev/rfcomm*`，这里的 pinmux 配置不会生效。
