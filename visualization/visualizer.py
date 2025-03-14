import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import networkx as nx
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import tqdm
from matplotlib.gridspec import GridSpec
from utils import config  # 确保导入config模块
from matplotlib.animation import FuncAnimation

# 定义顶级函数 (在类外部定义，使其可以被pickle)
def process_frame_wrapper(args):
    """适用于多处理的包装函数，调用实例方法"""
    visualizer_obj, time_idx, time_points, position_cache, metrics_cache = args
    return visualizer_obj._generate_frame(time_idx, time_points, position_cache, metrics_cache)

class SimulationVisualizer:
    """
    可视化UAV网络仿真过程，包括移动轨迹和通信状态
    """
    
    def __init__(self, simulator, output_dir="log_results"):
        """
        初始化可视化器
        
        参数:
            simulator: 仿真器实例
            output_dir: 输出目录
        """
        self.simulator = simulator
        self.output_dir = output_dir
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "frames"), exist_ok=True)
        
        # 初始化存储数据的结构
        self.drone_positions = {i: [] for i in range(self.simulator.n_drones)}
        self.timestamps = []
        self.active_links = []
        self.link_timestamps = []
        
        # 为每个UAV分配一个固定颜色
        self.colors = plt.cm.tab10(np.linspace(0, 1, self.simulator.n_drones))
        
        # 添加碰撞和丢包事件跟踪
        self.collision_events = []
        self.packet_drop_events = []
        
        # 存储PDR历史记录用于绘制动态折线图
        self.pdr_history = []
        self.time_history = []
        
        # Initialize arrows list
        self.arrows = []
        
        # Set up figure and axes for 3D visualization
        self.fig = plt.figure(figsize=(12, 8))
        self.gs = GridSpec(2, 2, height_ratios=[3, 1], width_ratios=[3, 1])
        self.ax = self.fig.add_subplot(self.gs[0, 0], projection='3d')
        self.pdr_ax = self.fig.add_subplot(self.gs[0, 1])
        self.pdr_ax.set_title('Packet Delivery Ratio')
        self.pdr_ax.set_xlabel('Time (s)')
        self.pdr_ax.set_ylabel('PDR (%)')
        self.pdr_ax.set_ylim([0, 100])
        self.pdr_line, = self.pdr_ax.plot([], [], 'g-')
        
        # Set initial axis labels and limits
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        
        # 初始化碰撞计数
        self.collision_count = 0
        
        # 保存初始时间
        self.start_time = self.simulator.env.now
    
    def track_drone_positions(self):
        """
        记录当前无人机位置
        """
        current_time = self.simulator.env.now / 1e6  # 转换为秒
        self.timestamps.append(current_time)
        
        for i, drone in enumerate(self.simulator.drones):
            position = drone.coords  # This already contains (x, y, z) coordinates as a list/tuple
            self.drone_positions[i].append(position)
    
    def track_communication(self, src_id, dst_id):
        """
        记录通信事件
        
        参数:
            src_id: 源无人机ID
            dst_id: 目标无人机ID
        """
        current_time = self.simulator.env.now / 1e6  # 转换为秒
        self.active_links.append((src_id, dst_id))
        self.link_timestamps.append(current_time)
    
    def track_collision(self, location, time):
        """记录碰撞事件"""
        self.collision_events.append((location, time))
        self.collision_count += 1  # 更新碰撞计数
    
    def track_packet_drop(self, source_id, packet_id, reason, time):
        """记录丢包事件"""
        current_time = time / 1e6  # 转换为秒
        self.packet_drop_events.append((source_id, packet_id, reason, current_time))
    
    def save_trajectory_plot(self):
        """
        保存无人机轨迹图
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制每个无人机的轨迹
        for i in range(self.simulator.n_drones):
            if not self.drone_positions[i]:
                continue
                
            x, y, z = zip(*self.drone_positions[i])
            ax.plot(x, y, z, color=self.colors[i], label=f'UAV {i}', linewidth=2)
            
            # 标记起点和终点
            ax.scatter(x[0], y[0], z[0], color=self.colors[i], marker='o', s=100)
            ax.scatter(x[-1], y[-1], z[-1], color=self.colors[i], marker='s', s=100)
        
        # 设置图形参数
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('UAV 3D Trajectories')
        ax.legend()
        
        # 保存图形
        plt.savefig(os.path.join(self.output_dir, 'uav_trajectories.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    def _generate_frame(self, time_idx, time_points, position_cache, metrics_cache):
        """
        生成单个可视化帧的辅助方法
        """
        current_time = time_points[time_idx]
        idx = time_idx  # 保持索引为文件命名
        
        # 记录当前时间点的PDR数据用于绘制动态曲线
        metrics_data = metrics_cache[current_time]
        sent_packets = metrics_data['sent_packets']
        received_packets = metrics_data['received_packets']
        total_collisions = metrics_data['total_collisions']
        
        # 计算PDR
        if sent_packets > 0:
            pdr = received_packets / sent_packets * 100
        else:
            pdr = 0
            
        # 存储PDR历史数据
        if time_idx > 0:
            self.pdr_history.append(pdr)
            self.time_history.append(current_time)

        # 创建图形对象
        fig = plt.figure(figsize=(15, 12))
        
        # 设置子图布局
        gs = fig.add_gridspec(3, 4)
        ax = fig.add_subplot(gs[:, :3], projection='3d')
        info_ax = fig.add_subplot(gs[0, 3])
        stats_ax = fig.add_subplot(gs[1, 3])
        events_ax = fig.add_subplot(gs[2, 3])
        
        # 获取当前帧中实际存在的无人机及其位置
        active_drones = {}
        for i in range(self.simulator.n_drones):
            if i in position_cache[current_time]:
                active_drones[i] = position_cache[current_time][i]
        
        # 绘制无人机位置
        for drone_id, position in active_drones.items():
            x, y, z = position
            color = self.colors[drone_id % len(self.colors)]
            
            # 绘制散点而不添加标签(以便不生成图例)
            ax.scatter(x, y, z, c=[color], s=50, alpha=1.0)
            
            # 直接在无人机旁边添加ID文本，增大字体大小
            ax.text(x, y, z, f"{drone_id}", color='black', fontsize=18, 
                    ha='center', va='bottom', weight='bold', 
                    bbox=dict(facecolor='white', alpha=0.8, pad=2, edgecolor='black'))
        
        # 绘制通信链路
        for link_idx, (src_id, dst_id) in enumerate(self.active_links):
            link_time = self.link_timestamps[link_idx]
            # 只显示当前时间点附近的链路
            time_diff = abs(link_time - current_time)
            if time_diff <= 0.05:  # 显示50ms内的链路
                if src_id in active_drones and dst_id in active_drones:
                    src_pos = active_drones[src_id]
                    dst_pos = active_drones[dst_id]
                    
                    # 绘制通信链路线
                    ax.plot([src_pos[0], dst_pos[0]], 
                            [src_pos[1], dst_pos[1]], 
                            [src_pos[2], dst_pos[2]], 'b-', linewidth=1.5)
                    
                    # 添加箭头方向表示数据流向
                    arrow_pos = [(src_pos[0] + dst_pos[0])/2, 
                                (src_pos[1] + dst_pos[1])/2, 
                                (src_pos[2] + dst_pos[2])/2]
                    arrow_direction = [dst_pos[0] - src_pos[0], 
                                    dst_pos[1] - src_pos[1], 
                                    dst_pos[2] - src_pos[2]]
                    length = np.sqrt(sum([d**2 for d in arrow_direction]))
                    arrow_direction = [d/length * 15 for d in arrow_direction]  # 标准化并缩放
                    
                    # 添加通信方向箭头
                    ax.quiver(arrow_pos[0], arrow_pos[1], arrow_pos[2], 
                            arrow_direction[0], arrow_direction[1], arrow_direction[2], 
                            color='r', arrow_length_ratio=0.3, linewidth=1.5)
        
        # 添加通信链路图例但不包含无人机
        ax.plot([], [], 'b-', label='Communication Links')
        ax.plot([], [], 'r-', label='Data Transmission Direction')
        ax.legend(loc='upper right', fontsize=8)
        
        # 设置图形标题和轴标签
        ax.set_title(f'UAV Network Simulation at t={current_time:.2f}s')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        
        # 设置坐标轴范围
        ax.set_xlim(0, config.MAP_LENGTH)
        ax.set_ylim(0, config.MAP_WIDTH)
        ax.set_zlim(0, config.MAP_HEIGHT)
        
        # 添加网格
        ax.grid(True)
        
        # 更新信息面板
        info_ax.axis('off')
        info_text = (
            f"Simulation Time: {current_time:.2f}s\n\n"
            f"Sent Packets: {sent_packets}\n"
            f"Received Packets: {received_packets}\n"
            f"Packet Delivery Ratio: {pdr:.2f}%\n"
            f"Total Collisions: {total_collisions}\n"
        )
        info_ax.text(0.05, 0.95, info_text, transform=info_ax.transAxes, 
                    fontsize=12, verticalalignment='top')
        
        # 绘制PDR折线图替代柱状图
        stats_ax.clear()
        
        # 如果有足够的历史数据，绘制折线图
        if len(self.time_history) > 1:
            # 绘制PDR历史折线
            stats_ax.plot(self.time_history, self.pdr_history, 'g-', linewidth=2)
            
            # 标记当前点
            stats_ax.plot(current_time, pdr, 'ro', markersize=8)
            
            # 添加网格线和标签
            stats_ax.grid(True, linestyle='--', alpha=0.7)
            stats_ax.set_xlabel('Time (s)')
            stats_ax.set_ylabel('PDR (%)')
            stats_ax.set_title('Packet Delivery Ratio')
            
            # 设置y轴范围
            stats_ax.set_ylim(0, 105)  # 给PDR留出一点空间
        else:
            # 如果没有足够的历史数据，绘制单个点
            stats_ax.bar(['PDR'], [pdr], color='g')
            stats_ax.set_ylabel('PDR (%)')
            stats_ax.set_ylim(0, 105)
        
        # 显示事件
        events_ax.axis('off')
        events_ax.set_title('Simulation Status')
        events_ax.text(0.05, 0.95, f"Frame {idx+1}/{len(time_points)}", 
                    transform=events_ax.transAxes, fontsize=10, verticalalignment='top')
        
        # 确保布局一致
        plt.tight_layout()
        
        # 保存为固定尺寸，确保所有图像大小一致
        output_path = os.path.join(self.output_dir, 'frames', f'frame_{idx:04d}.png')
        plt.savefig(output_path, dpi=100, bbox_inches=None)  # 不使用tight以保证固定大小
        plt.close(fig)
        
    def save_frame_visualization(self, interval=0.5):
        """
        保存帧可视化（定期快照）
        
        参数:
            interval: 帧间隔（秒）
        """
        if not self.timestamps:
            return
            
        max_time = max(self.timestamps)
        time_points = np.arange(0, max_time + interval, interval)
        
        # 优化1: 如果帧数过多，自动增加间隔进行降采样
        if len(time_points) > 100:
            sampling_factor = len(time_points) // 100 + 1
            time_points = time_points[::sampling_factor]
            print(f"Too many frames, reducing from {len(time_points)*sampling_factor} to {len(time_points)} frames")
        
        # 创建frames目录（如果不存在）
        frames_dir = os.path.join(self.output_dir, 'frames')
        os.makedirs(frames_dir, exist_ok=True)
        
        total_frames = len(time_points)
        print(f"Starting frame generation: {total_frames} frames to create")
        
        # 重置PDR历史数据
        self.pdr_history = []
        self.time_history = []
        
        # 预计算和缓存所有时间点的位置数据
        position_cache = {}
        for idx, current_time in enumerate(time_points):
            position_cache[current_time] = {}
            for i in range(self.simulator.n_drones):
                if not self.drone_positions[i]:
                    continue
                
                # 找到最接近当前时间的位置记录
                pos_times = np.array(self.timestamps)
                if len(pos_times) == 0:
                    continue
                    
                closest_idx = np.argmin(np.abs(pos_times - current_time))
                if closest_idx < len(self.drone_positions[i]):
                    position_cache[current_time][i] = self.drone_positions[i][closest_idx]
        
        # 预计算和缓存性能指标
        metrics_cache = {}
        for idx, current_time in enumerate(time_points):
            metrics = self.simulator.metrics
            
            try:
                sent_packets = sum(1 for t in metrics.datapacket_generated_time if t <= current_time*1e6)
            except AttributeError:
                try:
                    sent_packets = metrics.datapacket_generated_num
                except AttributeError:
                    sent_packets = 0
            
            try:
                received_packets = sum(1 for t in metrics.deliver_time_dict.values() if t <= current_time*1e6)
            except AttributeError:
                try:
                    received_packets = len(metrics.datapacket_arrived)
                except AttributeError:
                    received_packets = 0
            
            try:
                total_collisions = sum(1 for t in metrics.collision_time if t <= current_time*1e6)
            except AttributeError:
                try:
                    total_collisions = metrics.collision_num
                except AttributeError:
                    total_collisions = 0
                    
            metrics_cache[current_time] = {
                'sent_packets': sent_packets,
                'received_packets': received_packets,
                'total_collisions': total_collisions
            }
        
        # 直接使用单线程处理，不尝试多处理（避免错误消息）
        for idx, current_time in enumerate(time_points):
            if idx % max(1, total_frames // 10) == 0 or idx == total_frames-1:
                print(f"Generating frame {idx+1}/{total_frames} at time {current_time:.1f}s")
            
            # 直接调用帧生成方法
            self._generate_frame(idx, time_points, position_cache, metrics_cache)
        
        print(f"Completed all {total_frames} frames")
    
    def create_animations(self):
        """
        从保存的帧创建动画
        """
        try:
            # 创建3D轨迹动画
            frames_path = os.path.join(self.output_dir, 'frames')
            frames = sorted([os.path.join(frames_path, f) for f in os.listdir(frames_path) if f.endswith('.png')])
            
            if frames:
                print(f"Creating animation from {len(frames)} frames...")
                gif_path = os.path.join(self.output_dir, 'uav_simulation.gif')
                
                # 确保所有帧具有相同的尺寸
                standard_size = None
                resized_frames = []
                
                for frame_path in frames:
                    img = Image.open(frame_path)
                    if standard_size is None:
                        standard_size = img.size
                    
                    # 如果尺寸不同，则调整为标准尺寸
                    if img.size != standard_size:
                        img = img.resize(standard_size, Image.LANCZOS)
                    
                    resized_frames.append(img)
                
                # 设置帧率 - 减半
                fps = 5  # 原来是10
                if len(frames) > 100:
                    fps = 8  # 原来是15
                
                # 使用PIL保存GIF
                resized_frames[0].save(
                    gif_path, 
                    save_all=True, 
                    append_images=resized_frames[1:], 
                    duration=int(1000/fps), 
                    loop=0
                )
                
                print(f"Created animation: {gif_path}")
                
        except Exception as e:
            print(f"Error creating animation: {e}")
            import traceback
            traceback.print_exc()
    
    def run_visualization(self, tracking_interval=0.5, save_interval=1.0):
        """
        运行可视化过程
        
        参数:
            tracking_interval: 跟踪位置的时间间隔（秒）
            save_interval: 保存可视化结果的时间间隔（秒）
        """
        # 转换为微秒
        tracking_interval_us = tracking_interval * 1e6
        save_interval_us = save_interval * 1e6
        
        # 启动位置跟踪进程
        def track_process():
            while True:
                self.track_drone_positions()
                yield self.simulator.env.timeout(tracking_interval_us)
        
        # 启动定期保存进程
        def save_process():
            while True:
                yield self.simulator.env.timeout(save_interval_us)
                # 什么也不做，仅在最后保存
        
        # 注册进程
        self.simulator.env.process(track_process())
        self.simulator.env.process(save_process())
    
    def finalize(self):
        """
        完成可视化处理并生成最终输出
        """
        print("Saving visualization results...")
        
        # 保存帧可视化
        self.save_frame_visualization()
        
        # 创建动画
        self.create_animations()
        
        print("Visualization complete!")