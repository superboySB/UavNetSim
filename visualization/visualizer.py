import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import networkx as nx
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

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
        
        for idx, current_time in enumerate(time_points):
            # 创建图形 - 使用子图以便添加指标面板
            fig = plt.figure(figsize=(14, 10))
            gs = fig.add_gridspec(1, 4)
            ax = fig.add_subplot(gs[0, :3])
            info_ax = fig.add_subplot(gs[0, 3])
            
            # 创建网络图
            G = nx.DiGraph()
            
            # 添加所有无人机节点
            for i in range(self.simulator.n_drones):
                G.add_node(i)
            
            # 找出时间窗口内的通信链路
            window_links = []
            for link, timestamp in zip(self.active_links, self.link_timestamps):
                if current_time - time_window <= timestamp <= current_time:
                    window_links.append(link)
            
            # 添加链路
            for src, dst in window_links:
                G.add_edge(src, dst)
            
            # 获取当前时间点的无人机位置
            current_positions = {}
            for i in range(self.simulator.n_drones):
                if not self.drone_positions[i]:
                    continue
                
                # 找到最接近当前时间的位置记录
                pos_times = np.array(self.timestamps)
                if len(pos_times) == 0:
                    continue
                
                closest_idx = np.argmin(np.abs(pos_times - current_time))
                if closest_idx < len(self.drone_positions[i]):
                    x, y, _ = self.drone_positions[i][closest_idx]
                    current_positions[i] = (x, y)
            
            if not current_positions:
                continue
            
            # 计算当前位置的边界
            x_coords = [pos[0] for pos in current_positions.values()]
            y_coords = [pos[1] for pos in current_positions.values()]
            
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            # 添加边距
            margin = max((x_max - x_min), (y_max - y_min)) * 0.1
            if margin < 10:  # 确保最小边距
                margin = 10
            
            x_min -= margin
            x_max += margin
            y_min -= margin
            y_max += margin
            
            # 确保坐标范围不为零
            if x_min == x_max:
                x_min -= 10
                x_max += 10
            if y_min == y_max:
                y_min -= 10
                y_max += 10
            
            # 设置坐标轴范围（在绘制之前）
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            
            # 确保图形是正方形，以保持比例一致
            width = x_max - x_min
            height = y_max - y_min
            if width > height:
                extra = (width - height) / 2
                ax.set_ylim(y_min - extra, y_max + extra)
            else:
                extra = (height - width) / 2
                ax.set_xlim(x_min - extra, x_max + extra)
            
            # 绘制网络图
            try:
                nodes = nx.draw_networkx_nodes(G, current_positions, node_size=500, 
                                       node_color=[to_rgba(self.colors[i]) for i in range(self.simulator.n_drones)],
                                       ax=ax)
                
                # 获取链路效率信息
                edge_colors = []
                for src, dst in G.edges():
                    edge_colors.append('blue')  # 默认为蓝色
                
                edges = nx.draw_networkx_edges(G, current_positions, width=2, alpha=0.7, arrows=True, 
                                      arrowsize=15, arrowstyle='->', edge_color=edge_colors,
                                      ax=ax)
                
                labels = nx.draw_networkx_labels(G, current_positions, font_size=12, font_weight='bold',
                                       ax=ax)
            except Exception as e:
                print(f"Error drawing network: {e}")
                continue
            
            # 确保绘图精确包含节点
            ax.update_datalim([(x_min, y_min), (x_max, y_max)])
            ax.autoscale_view()
            
            # 获取当前时间点的性能指标
            metrics = self.simulator.metrics
            
            # 尝试安全地获取数据包统计信息
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
            
            # 计算当前窗口内新增的碰撞数
            window_collisions = total_collisions - previous_collisions
            previous_collisions = total_collisions
            
            # 计算PDR
            if sent_packets > 0:
                pdr = received_packets / sent_packets * 100
            else:
                pdr = 0
                
            # 添加到历史记录 - 存储累计值
            performance_history['sent'].append(sent_packets)
            performance_history['received'].append(received_packets)
            performance_history['pdr'].append(pdr)
            performance_history['collisions'].append(total_collisions)
            performance_history['window_collisions'].append(window_collisions)
            
            # 添加性能指标面板
            info_ax.axis('off')
            info_text = (
                f"Simulation Time: {current_time:.2f}s\n\n"
                f"Active Links: {len(window_links)}\n"
                f"Sent Packets: {sent_packets}\n"
                f"Received Packets: {received_packets}\n"
                f"Packet Delivery Ratio: {pdr:.2f}%\n"
                f"Collisions: {total_collisions}\n\n"
                f"Link Color Legend:\n"
                f"Blue: Normal Link\n"
            )
            info_ax.text(0.05, 0.95, info_text, transform=info_ax.transAxes, 
                        fontsize=12, verticalalignment='top')
            
            # 设置图形参数
            ax.set_title(f'UAV Communication Network at t={current_time:.1f}s')
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            
            # 重新确认轴的范围
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            
            # 确保图像尺寸适配
            plt.tight_layout()
            
            # 保存图形
            plt.savefig(os.path.join(self.output_dir, 'communications', f'comm_graph_{idx:04d}.png'), 
                        dpi=200, bbox_inches='tight')
            plt.close(fig)
            
            # 每5个图像输出一条进度消息
            if idx % 5 == 0:
                print(f"Saving communication graph {idx}/{len(time_points)} at time {current_time:.1f}s")
    
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
        
        # 跟踪性能指标 - 直接存储累计值
        performance_history = {
            'sent': [], 
            'received': [], 
            'pdr': [], 
            'collisions': [],
            'window_collisions': []  # 存储每个窗口新增的碰撞数
        }
        
        recent_events = []  # 用于存储最近的通信事件
        previous_collisions = 0
        
        total_frames = len(time_points)
        print(f"Starting frame generation: {total_frames} frames to create")
        
        for idx, current_time in enumerate(time_points):
            # 每5个图像输出一条进度消息，以便跟踪进度
            if idx % 5 == 0:
                print(f"Generating frame {idx+1}/{total_frames} at time {current_time:.1f}s")
            
            fig = plt.figure(figsize=(15, 12))
            
            # 创建主3D图和信息面板
            gs = fig.add_gridspec(3, 4)
            ax = fig.add_subplot(gs[:, :3], projection='3d')
            info_ax = fig.add_subplot(gs[0, 3])
            stats_ax = fig.add_subplot(gs[1, 3])
            events_ax = fig.add_subplot(gs[2, 3])
            
            # 获取当前时间点的性能指标
            metrics = self.simulator.metrics
            
            # 安全获取数据包统计信息
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
            
            # 获取碰撞的累计总数
            try:
                total_collisions = sum(1 for t in metrics.collision_time if t <= current_time*1e6)
            except AttributeError:
                try:
                    total_collisions = metrics.collision_num
                except AttributeError:
                    total_collisions = 0
            
            # 计算当前窗口内新增的碰撞数
            window_collisions = total_collisions - previous_collisions
            previous_collisions = total_collisions
            
            # 计算PDR
            if sent_packets > 0:
                pdr = received_packets / sent_packets * 100
            else:
                pdr = 0
                
            # 添加到历史记录 - 存储累计值
            performance_history['sent'].append(sent_packets)
            performance_history['received'].append(received_packets)
            performance_history['pdr'].append(pdr)
            performance_history['collisions'].append(total_collisions)  # 存储累计碰撞数
            performance_history['window_collisions'].append(window_collisions)  # 存储窗口增量
            
            # 绘制无人机和路径
            for i in range(self.simulator.n_drones):
                if not self.drone_positions[i]:
                    continue
                
                # 找到最接近当前时间的位置记录
                pos_times = np.array(self.timestamps)
                if len(pos_times) == 0:
                    continue
                    
                closest_idx = np.argmin(np.abs(pos_times - current_time))
                if closest_idx < len(self.drone_positions[i]):
                    # 绘制无人机位置
                    x, y, z = self.drone_positions[i][closest_idx]
                    ax.scatter(x, y, z, color=self.colors[i], s=100, label=f'UAV {i}')
                    
                    # 绘制无人机轨迹
                    if closest_idx > 0:
                        x_history, y_history, z_history = zip(*self.drone_positions[i][:closest_idx+1])
                        ax.plot(x_history, y_history, z_history, color=self.colors[i], linewidth=1.5, alpha=0.5)
            
            # 绘制当前活跃的通信链路
            for src, dst in self.active_links:
                # 找到链路对应的时间戳
                link_times = np.array(self.link_timestamps)
                if len(link_times) == 0:
                    continue
                    
                # 仅绘制当前时间窗口内的链路
                valid_links = np.where((link_times <= current_time) & (link_times >= current_time - interval))[0]
                
                for link_idx in valid_links:
                    src = self.active_links[link_idx][0]
                    dst = self.active_links[link_idx][1]
                    
                    # 获取源节点和目标节点的位置
                    src_times = np.array(self.timestamps)
                    dst_times = np.array(self.timestamps)
                    
                    if len(src_times) == 0 or len(dst_times) == 0:
                        continue
                    
                    src_idx = np.argmin(np.abs(src_times - current_time))
                    dst_idx = np.argmin(np.abs(dst_times - current_time))
                    
                    if (src_idx < len(self.drone_positions[src]) and 
                        dst_idx < len(self.drone_positions[dst])):
                        
                        src_pos = self.drone_positions[src][src_idx]
                        dst_pos = self.drone_positions[dst][dst_idx]
                        
                        ax.plot([src_pos[0], dst_pos[0]], 
                            [src_pos[1], dst_pos[1]], 
                            [src_pos[2], dst_pos[2]], 
                            'k-', alpha=0.7, linewidth=1.5)
                        
                        # 记录最近的通信事件
                        recent_events.append(f"UAV {src} -> UAV {dst}")
                        if len(recent_events) > 10:  # 限制事件列表长度
                            recent_events.pop(0)
            
            # 设置3D图参数
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title(f'UAV Network Simulation at t={current_time:.1f}s')
            ax.legend(loc='upper right')
            
            # 绘制性能信息面板
            info_ax.axis('off')
            info_text = (
                f"Simulation Time: {current_time:.2f}s\n\n"
                f"Sent Packets: {sent_packets}\n"
                f"Received Packets: {received_packets}\n"
                f"Packet Delivery Ratio: {pdr:.2f}%\n"
                f"Total Collisions: {total_collisions}\n"
                f"New Collisions: {window_collisions}\n"
            )
            info_ax.text(0.05, 0.95, info_text, transform=info_ax.transAxes, 
                        fontsize=12, verticalalignment='top')
            
            # 绘制性能指标历史图表
            stats_ax.plot(time_points[:idx+1], performance_history['pdr'], 'g-', label='PDR (%)')
            stats_ax.set_ylabel('PDR (%)', color='g')
            stats_ax.tick_params(axis='y', labelcolor='g')
            stats_ax.set_ylim(0, 100)
            
            ax2 = stats_ax.twinx()
            # 绘制累计碰撞数 - 直接使用存储的累计值
            ax2.plot(time_points[:idx+1], performance_history['collisions'], 'r-', label='Collisions')
            ax2.set_ylabel('Collision Count', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            
            stats_ax.set_title('Network Performance Metrics')
            stats_ax.set_xlabel('Time (s)')
            
            # 显示最近事件
            events_ax.axis('off')
            events_ax.set_title('Recent Communication Events')
            
            event_text = "\n".join(recent_events[-8:]) if recent_events else "No recent events"
            events_ax.text(0.05, 0.95, event_text, transform=events_ax.transAxes,
                        fontsize=10, verticalalignment='top')
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图形
            plt.savefig(os.path.join(self.output_dir, 'frames', f'frame_{idx:04d}.png'), 
                        dpi=200, bbox_inches='tight')
            
            # 确保内存管理：清除不再需要的大型对象
            plt.close(fig)
            fig = None  # 明确释放
            ax = None
            info_ax = None
            stats_ax = None
            events_ax = None
        
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
                
                # 使用PIL创建GIF，但限制每批处理的图像数量以减少内存使用
                batch_size = 20  # 每批处理的最大帧数
                
                if len(frames) <= batch_size:
                    # 如果帧数较少，直接创建GIF
                    imgs = [Image.open(f) for f in frames]
                    imgs[0].save(gif_path, save_all=True, append_images=imgs[1:], duration=200, loop=0)
                else:
                    # 如果帧数较多，分批处理
                    first_img = Image.open(frames[0])
                    first_img.save(gif_path, save_all=True, append_images=[], duration=200, loop=0)
                    
                    for i in range(1, len(frames), batch_size):
                        batch_end = min(i + batch_size, len(frames))
                        batch = [Image.open(f) for f in frames[i:batch_end]]
                        
                        # 追加到现有GIF
                        with open(gif_path, 'rb') as f:
                            gif = Image.open(f)
                            gif.seek(0)  # 回到开头
                            frames_in_gif = 0
                            try:
                                while True:
                                    frames_in_gif += 1
                                    gif.seek(frames_in_gif)
                            except EOFError:
                                pass
                        
                        # 保存更新后的GIF
                        first_img.save(gif_path, save_all=True, 
                                    append_images=batch, duration=200, loop=0)
                        
                        print(f"Added frames {i}-{batch_end} to animation")
                
                print(f"Created animation: {gif_path}")
                
            # 创建通信网络动画 (同样使用批处理方法)
            comm_path = os.path.join(self.output_dir, 'communications')
            comm_frames = sorted([os.path.join(comm_path, f) for f in os.listdir(comm_path) if f.endswith('.png')])
            
            if comm_frames:
                print(f"Creating communication network animation from {len(comm_frames)} frames...")
                comm_gif_path = os.path.join(self.output_dir, 'communication_network.gif')
                
                # 使用PIL创建GIF，但限制每批处理的图像数量以减少内存使用
                batch_size = 20  # 每批处理的最大帧数
                
                if len(comm_frames) <= batch_size:
                    # 如果帧数较少，直接创建GIF
                    comm_imgs = [Image.open(f) for f in comm_frames]
                    comm_imgs[0].save(comm_gif_path, save_all=True, append_images=comm_imgs[1:], duration=200, loop=0)
                else:
                    # 如果帧数较多，分批处理
                    first_img = Image.open(comm_frames[0])
                    first_img.save(comm_gif_path, save_all=True, append_images=[], duration=200, loop=0)
                    
                    for i in range(1, len(comm_frames), batch_size):
                        batch_end = min(i + batch_size, len(comm_frames))
                        batch = [Image.open(f) for f in comm_frames[i:batch_end]]
                        
                        # 追加到现有GIF
                        with open(comm_gif_path, 'rb') as f:
                            gif = Image.open(f)
                            gif.seek(0)  # 回到开头
                            frames_in_gif = 0
                            try:
                                while True:
                                    frames_in_gif += 1
                                    gif.seek(frames_in_gif)
                            except EOFError:
                                pass
                        
                        # 保存更新后的GIF
                        first_img.save(comm_gif_path, save_all=True, 
                                    append_images=batch, duration=200, loop=0)
                        
                        print(f"Added frames {i}-{batch_end} to communication network animation")
                
                print(f"Created communication network animation: {comm_gif_path}")
                
        except Exception as e:
            print(f"Error creating animation: {e}")
            import traceback
            traceback.print_exc()  # 打印完整的错误堆栈跟踪
    
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
        完成可视化，保存所有图形
        """
        print("Saving visualization results...")
        self.save_trajectory_plot()
        self.save_communication_graph()
        self.save_frame_visualization()
        self.create_animations()
        print("Visualization completed! Results saved in directory:", self.output_dir)