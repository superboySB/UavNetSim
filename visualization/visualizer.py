import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import networkx as nx
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import tqdm

# 定义顶级函数 (在类外部定义，使其可以被pickle)
def process_frame_wrapper(args):
    """适用于多处理的包装函数，调用实例方法"""
    visualizer_obj, time_idx, time_points, position_cache, metrics_cache = args
    return visualizer_obj._generate_frame(time_idx, time_points, position_cache, metrics_cache)

class SimulationVisualizer:
    """
    可视化UAV网络仿真过程，包括移动轨迹和通信状态
    """
    
    def __init__(self, simulator, output_dir="visualization"):
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
        os.makedirs(os.path.join(output_dir, "communications"), exist_ok=True)
        
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
        current_time = time / 1e6  # 转换为秒
        self.collision_events.append((location, current_time))
    
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
    
    def save_communication_graph(self, time_window=0.5):
        """
        保存通信图（在给定时间窗口内的通信连接）
        
        参数:
            time_window: 时间窗口大小（秒）
        """
        # 创建一系列时间点
        if not self.timestamps:
            return
            
        max_time = max(self.timestamps)
        time_points = np.arange(0, max_time + time_window, time_window)
        
        # 初始化变量
        previous_collisions = 0
        performance_history = {
            'sent': [], 
            'received': [], 
            'pdr': [], 
            'collisions': [],
            'window_collisions': []
        }
        recent_events = []
        
        # 确保目录存在
        comm_dir = os.path.join(self.output_dir, 'communications')
        os.makedirs(comm_dir, exist_ok=True)
        
        # 为每个时间点创建通信图
        for idx, current_time in enumerate(time_points):
            if idx % 5 == 0:
                print(f"Saving communication graph {idx+1}/{len(time_points)} at time {current_time:.1f}s")
            
            # 找到该时间点之前的最后一个通信记录
            current_links = []
            
            # 安全获取通信链接
            for t, l in self.active_links:
                if t <= current_time:
                    if isinstance(l, list):  # 确保l是一个列表
                        current_links = l
                    break
                
            # 创建图形
            fig = plt.figure(figsize=(12, 8))
            gs = fig.add_gridspec(1, 5)
            ax = fig.add_subplot(gs[0, :4])
            info_ax = fig.add_subplot(gs[0, 4])
            
            # 创建通信网络图
            G = nx.Graph()
            
            # 添加无人机节点
            for i in range(self.simulator.n_drones):
                if not self.drone_positions[i]:
                    continue
                
                # 找到最接近当前时间的位置记录
                pos_times = np.array(self.timestamps)
                if len(pos_times) == 0:
                    continue
                
                closest_idx = np.argmin(np.abs(pos_times - current_time))
                if closest_idx < len(self.drone_positions[i]):
                    pos = self.drone_positions[i][closest_idx]
                    # 只用2D坐标作为图形位置
                    G.add_node(i, pos=(pos[0], pos[1]))
            
            # 添加通信链接
            for i, j in current_links:
                if i in G.nodes and j in G.nodes:
                    G.add_edge(i, j)
            
            # 获取节点位置以便绘图
            pos = nx.get_node_attributes(G, 'pos')
            
            # 计算整体边界框
            if pos:
                x_coords = [p[0] for p in pos.values()]
                y_coords = [p[1] for p in pos.values()]
                
                # 添加一些边距
                margin_x = max(50, (max(x_coords) - min(x_coords)) * 0.1)
                margin_y = max(50, (max(y_coords) - min(y_coords)) * 0.1)
                
                # 计算边界
                x_min, x_max = min(x_coords) - margin_x, max(x_coords) + margin_x
                y_min, y_max = min(y_coords) - margin_y, max(y_coords) + margin_y
                
                # 确保边界框是正方形（保持比例）
                width = x_max - x_min
                height = y_max - y_min
                if width > height:
                    # 如果宽度大于高度，增加高度
                    diff = (width - height) / 2
                    y_min -= diff
                    y_max += diff
                else:
                    # 如果高度大于宽度，增加宽度
                    diff = (height - width) / 2
                    x_min -= diff
                    x_max += diff
            else:
                # 默认边界
                x_min, x_max = 0, 500
                y_min, y_max = 0, 500
            
            # 绘制节点（无人机）
            for i, (x, y) in pos.items():
                ax.scatter(x, y, s=200, color=plt.cm.tab10(i / 10), edgecolors='black', linewidths=1)
                ax.text(x, y, f"{i}", horizontalalignment='center', verticalalignment='center', fontweight='bold')
            
            # 绘制边（通信链接）
            for (i, j) in G.edges():
                ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]], 'b-', alpha=0.7, linewidth=1.5)
            
            # 显示信息面板
            info_ax.axis('off')
            metrics = self.simulator.metrics
            
            # 计算性能指标
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
                
            # 计算PDR
            if sent_packets > 0:
                pdr = received_packets / sent_packets * 100
            else:
                pdr = 0
                
            # 更新性能历史记录
            performance_history['sent'].append(sent_packets)
            performance_history['received'].append(received_packets)
            performance_history['pdr'].append(pdr)
            performance_history['collisions'].append(total_collisions)
            
            # 计算当前时间窗口内的碰撞
            window_collisions = total_collisions - previous_collisions
            previous_collisions = total_collisions
            performance_history['window_collisions'].append(window_collisions)
            
            # 更新事件历史
            if window_collisions > 0:
                recent_events.append(f"t={current_time:.1f}s: {window_collisions} collisions")
            
            # 保留最近10个事件
            recent_events = recent_events[-10:]
            
            # 绘制信息文本
            info_text = (
                f"Time: {current_time:.1f}s\n\n"
                f"Network Summary:\n"
                f"Nodes: {len(G.nodes)}\n"
                f"Links: {len(G.edges)}\n\n"
                f"Performance:\n"
                f"PDR: {pdr:.1f}%\n"
                f"Total Collisions: {total_collisions}\n\n"
                f"Link Color Legend:\n"
                f"Blue: Normal Link\n"
            )
            info_ax.text(0.05, 0.95, info_text, transform=info_ax.transAxes, 
                        fontsize=12, verticalalignment='top')
            
            # 设置图形参数
            ax.set_title(f'UAV Communication Network at t={current_time:.1f}s')
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            
            # 设置坐标轴范围
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            
            # 确保图像尺寸适配
            plt.tight_layout()
            
            # 保存图形
            plt.savefig(os.path.join(self.output_dir, 'communications', f'comm_graph_{idx:04d}.png'), 
                        dpi=200, bbox_inches='tight')
            plt.close(fig)
        
        # 输出完成消息
        print(f"Saved communication graphs for {len(time_points)} time points")
    
    def _generate_frame(self, time_idx, time_points, position_cache, metrics_cache):
        """
        生成单个可视化帧的辅助方法
        
        参数:
            time_idx: 时间点索引
            time_points: 所有时间点列表
            position_cache: 预计算的位置缓存
            metrics_cache: 预计算的指标缓存
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
        
        # 绘制无人机和路径
        for i in range(self.simulator.n_drones):
            if i not in position_cache[current_time]:
                continue
            
            pos = position_cache[current_time][i]
            ax.scatter(pos[0], pos[1], pos[2], color=f'C{i}', s=100, marker='o', label=f'UAV {i}')
            
            # 绘制UAV历史轨迹
            past_positions = []
            for t in time_points[:time_idx+1]:
                if i in position_cache[t]:
                    past_positions.append(position_cache[t][i])
            
            if past_positions:
                past_positions = np.array(past_positions)
                ax.plot(past_positions[:, 0], past_positions[:, 1], past_positions[:, 2], 
                      color=f'C{i}', linestyle='-', alpha=0.7)
        
        # 显示通信链接
        link_time = None
        links = []
        
        # 找到最接近当前时间的通信链接记录
        for t, l in self.active_links:
            if t <= current_time:
                link_time = t
                links = l
        
        # 绘制通信链接
        if links and isinstance(links, list):  # 确保links是一个列表
            for i, j in links:
                if i in position_cache[current_time] and j in position_cache[current_time]:
                    pos_i = position_cache[current_time][i]
                    pos_j = position_cache[current_time][j]
                    ax.plot([pos_i[0], pos_j[0]], [pos_i[1], pos_j[1]], [pos_i[2], pos_j[2]], 
                          'k--', alpha=0.4, linewidth=1)
        
        # 设置图表标题和标签
        ax.set_title(f'UAV Network Simulation at t={current_time:.1f}s')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.legend()
        
        # 优化3D视图
        x_min, x_max = 0, 500
        y_min, y_max = 300, 650
        z_min, z_max = 150, 350
        
        # 根据实际坐标数据调整视图范围
        pos_data = []
        for i in range(self.simulator.n_drones):
            if i in position_cache[current_time]:
                pos_data.append(position_cache[current_time][i])
        
        if pos_data:
            pos_data = np.array(pos_data)
            
            # 添加一些边距
            margin = 50
            x_min, x_max = min(pos_data[:, 0]) - margin, max(pos_data[:, 0]) + margin
            y_min, y_max = min(pos_data[:, 1]) - margin, max(pos_data[:, 1]) + margin
            z_min, z_max = min(pos_data[:, 2]) - margin, max(pos_data[:, 2]) + margin
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        
        # 绘制性能信息面板
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
        plt.close('all')  # 确保所有图形对象都被释放
        
        return idx
        
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
                
                # 设置帧率
                fps = 10
                if len(frames) > 100:
                    fps = 15
                
                # 使用PIL保存GIF
                resized_frames[0].save(
                    gif_path, 
                    save_all=True, 
                    append_images=resized_frames[1:], 
                    duration=int(1000/fps), 
                    loop=0
                )
                
                print(f"Created animation: {gif_path}")
                
            # 创建通信网络动画
            comm_frames_path = os.path.join(self.output_dir, 'communications')
            comm_frames = sorted([os.path.join(comm_frames_path, f) for f in os.listdir(comm_frames_path) 
                                if f.endswith('.png')])
            
            if comm_frames:
                print(f"Creating communication network animation from {len(comm_frames)} frames...")
                comm_gif_path = os.path.join(self.output_dir, 'communication_network.gif')
                
                # 确保所有帧具有相同的尺寸
                standard_size = None
                resized_comm_frames = []
                
                for frame_path in comm_frames:
                    img = Image.open(frame_path)
                    if standard_size is None:
                        standard_size = img.size
                    
                    # 如果尺寸不同，则调整为标准尺寸
                    if img.size != standard_size:
                        img = img.resize(standard_size, Image.LANCZOS)
                    
                    resized_comm_frames.append(img)
                
                # 使用PIL保存GIF
                resized_comm_frames[0].save(
                    comm_gif_path, 
                    save_all=True, 
                    append_images=resized_comm_frames[1:], 
                    duration=int(1000/fps), 
                    loop=0
                )
                
                print(f"Created communication network animation: {comm_gif_path}")
                
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
        
        # 保存通信图 - 现在已修复，不再需要额外参数
        self.save_communication_graph()
        
        # 创建动画
        self.create_animations()
        
        print("Visualization complete!")