# Visualization Enhancement PR

## Enhanced UAV Network Visualization (English)

Developed a new interactive visualizer module for the UAV network simulator, enhencing the previous static plotting approach. The visualizer provides an interactive dynamic plot that can be launched after simulation completion, allowing users to analyze routing protocol behaviors by combining visualization with log analysis. Users can configure the visualization time granularity in main.py to control the temporal resolution of displayed communication events. For each pair of UAVs, the visualizer displays their latest DATA and ACK packets within the time window, and shows all concurrent communications between different UAV pairs, though for the same UAV pair only the latest packet is shown. This enhancement enables better understanding of protocol dynamics through visual inspection of network topology and packet routing during simulation playback.

## 可视化系统增强 (中文)

为无人机网络模拟器开发了新的交互式可视化模块，增强了之前的静态绘图方式。该可视化器在仿真完成后提供了一个可交互的动态绘图界面，用户可以结合日志分析来研究路由协议的行为。用户可以在main.py中配置可视化的时间粒度来控制通信事件显示的时间分辨率。对于每对无人机，可视化器会显示它们在时间窗口内最新的DATA和ACK包，并同时显示不同无人机对之间的所有并发通信，但对于同一对无人机只显示最新的数据包。这一增强功能通过对网络拓扑和数据包路由的可视化回放，使用户能够更好地理解协议动态过程。
