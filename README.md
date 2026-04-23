# MyG0 视觉任务开源

这个仓库是我们学校机甲杯校内赛中我们战队的视觉项目工程，基于MaixCDK开发，部署于Sipeed家的Maixcam2平台。
本人懒得对原仓库进行剔除后提交，遂直接把整个编译链上传了。本人修改位于components中的TargetTracking目录，创建了自己需要用到的识别算法、云台舵机驱动、遗传算法模板和包装好的识别-追踪管线。
主函数位于examples中的Mygo_pipeline_uart，虽然叫这个名字，但其实指令传输链路是和上位机建立TCP连接，最开始想直接通过相机模块的串口线直接驱动舵机的，但是由于各种工程原因未能实现。TCP连接为我贡献了更多可以调试的信息和更多可以与ROS对接的接口。
部署后按照MaixCDK官方要求添加工具链到环境变量，再项目目录下执行``` maixcdk build -p maixcam2 ```即可。待工具链下载完成后即自动开始编译，编译完成后在dist目录中能够找到可执行文件，通过MaixVision官方IDE或是其他方式安装至maixcam2并运行即可。
注意如果直接运行文件需要先``` chmod +x ```赋予权限。如果直接用``` maixcdk deploy ```也可以生成安装二维码并安装至相机，通过应用界面调用。
哦对了，这个项目启动需要接受TCP协议传过来的启动指令，请自行阅读协议并考虑用其他方式启动。
## 依赖

本项目依赖以下第三方组件：

### MaixCDK
- **版权方**: Copyright (c) 2023- Sipeed Ltd.
- **开源许可证**: Apache 2.0
- **原始仓库**: [[MaixCDK 官方仓库地址]](https://github.com/sipeed/MaixCDK)

### c_cpp_project_framework
- **版权方**: Copyright (c) 2019- Neucrack (CZD666666@gmail.com)
- **开源许可证**: MIT
- **原始仓库**: https://github.com/neutree/c_cpp_project_framework

> **注意**: MaixCDK 可能包含其他子模块，各子模块的许可证信息请参见各子模块中的 LICENSE 文件。

## 许可证

### 本项目代码

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### 第三方组件许可证声明

本项目中包含以下第三方组件的代码（或依赖）：

#### MaixCDK - Apache 2.0
