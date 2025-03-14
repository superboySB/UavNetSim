import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tqdm
from matplotlib.gridspec import GridSpec
from utils import config  # 确保导入config模块
from matplotlib.lines import Line2D  # Add this import
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

# 添加3D箭头类定义，处理3D视图中的箭头，正确地实现do_3d_projection方法
class Arrow3D(FancyArrowPatch):
    """
    用于在3D视图中绘制箭头的类
    """
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        # 计算平均z值作为深度
        avg_z = np.mean(zs)
        return avg_z
        
    def draw(self, renderer):
        FancyArrowPatch.draw(self, renderer)

# 定义顶级函数 (在类外部定义，使其可以被pickle)
def process_frame_wrapper(args):
    """适用于多处理的包装函数，调用实例方法"""
    visualizer_obj, time_idx, time_points, position_cache, metrics_cache = args
    return visualizer_obj._generate_frame(time_idx, time_points, position_cache, metrics_cache)

class SimulationVisualizer:
    """
    可视化UAV网络仿真过程，包括移动轨迹和通信状态
    """
    
    def __init__(self, simulator, output_dir="vis_results"):
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
        
        # 增强通信事件的记录细节
        self.comm_events = []  # 存储元组 (src_id, dst_id, packet_id, packet_type, timestamp)
        
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
        
        # 通信类型的颜色映射
        self.comm_colors = {
            "DATA": "blue",
            "ACK": "green",
            "HELLO": "orange"
        }
        
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
    
    def track_communication(self, src_id, dst_id, packet_id, packet_type="DATA"):
        """
        记录通信事件
        
        参数:
            src_id: 源无人机ID
            dst_id: 目标无人机ID
            packet_id: 数据包ID
            packet_type: 包类型 (DATA, ACK, HELLO)
        """
        current_time = self.simulator.env.now / 1e6  # 转换为秒
        # 记录完整的通信事件信息
        self.comm_events.append((src_id, dst_id, packet_id, packet_type, current_time))
    
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
        
        # 获取当前时间点的度量数据
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
            
            # 增大散点大小
            ax.scatter(x, y, z, c=[color], s=150, alpha=1.0)  # 增大到150让无人机更明显
            
            # 增大无人机ID文本
            ax.text(x, y, z + 15, f"{drone_id}", color='black', fontsize=20,  
                    ha='center', va='center', weight='bold', 
                    bbox=dict(facecolor='white', alpha=0.8, pad=3, edgecolor='black'))
        
        # 创建通信组字典，用于分组并错开同类型通信
        comm_groups = {packet_type: {} for packet_type in self.comm_colors.keys()}
        
        # 首先分组所有通信，以便更好地错开它们
        time_window = 0.10  # 保持100ms时间窗口
        for src_id, dst_id, packet_id, packet_type, event_time in self.comm_events:
            if abs(event_time - current_time) <= time_window:
                if src_id in active_drones and dst_id in active_drones:
                    # 使用src-dst对作为键
                    pair_key = (min(src_id, dst_id), max(src_id, dst_id))
                    if pair_key not in comm_groups[packet_type]:
                        comm_groups[packet_type][pair_key] = []
                    comm_groups[packet_type][pair_key].append((src_id, dst_id, packet_id))
        
        # 现在为每个通信类型按组绘制
        for packet_type, groups in comm_groups.items():
            color = self.comm_colors[packet_type]
            linewidth = 2.5 if packet_type == "DATA" else 2.0
            alpha = 0.9 if packet_type == "DATA" else 0.7
            style = '-' if packet_type == "DATA" else '--' if packet_type == "ACK" else ':'
            
            for pair_key, comms in groups.items():
                # 对于每对无人机，计算每个通信包应该有多大的偏移
                for idx, (src_id, dst_id, packet_id) in enumerate(comms):
                    src_pos = active_drones[src_id]
                    dst_pos = active_drones[dst_id]
                    
                    # 计算不同通信包类型之间的垂直偏移
                    packet_type_offset = 0
                    if packet_type == "DATA":
                        packet_type_offset = 0
                    elif packet_type == "ACK":
                        packet_type_offset = 12  # ACK在DATA上方
                    elif packet_type == "HELLO":
                        packet_type_offset = -12  # HELLO在DATA下方
                    
                    # 对同类型通信进行进一步错开
                    offset_multiplier = idx % 3 - 1  # -1, 0, 1 循环
                    
                    # 计算垂直偏移向量
                    dx = dst_pos[0] - src_pos[0]
                    dy = dst_pos[1] - src_pos[1]
                    dz = dst_pos[2] - src_pos[2]
                    
                    # 创建一个垂直于xy平面的偏移
                    normal_x = -dy
                    normal_y = dx
                    normal_z = 0
                    
                    # 归一化并缩放
                    normal_length = np.sqrt(normal_x**2 + normal_y**2)
                    if normal_length > 0:
                        normal_x = normal_x / normal_length
                        normal_y = normal_y / normal_length
                    
                    # 应用基本类型偏移和同类型偏移
                    base_offset = 8  # 基本偏移距离
                    offset_x = normal_x * (packet_type_offset + offset_multiplier * base_offset)
                    offset_y = normal_y * (packet_type_offset + offset_multiplier * base_offset)
                    offset_z = packet_type_offset + offset_multiplier * 5  # z方向也有偏移
                    
                    # 应用偏移
                    src_pos_offset = [src_pos[0] + offset_x, src_pos[1] + offset_y, src_pos[2] + offset_z]
                    dst_pos_offset = [dst_pos[0] + offset_x, dst_pos[1] + offset_y, dst_pos[2] + offset_z]
                    
                    # 绘制通信线
                    ax.plot([src_pos_offset[0], dst_pos_offset[0]], 
                            [src_pos_offset[1], dst_pos_offset[1]], 
                            [src_pos_offset[2], dst_pos_offset[2]], 
                            linestyle=style, color=color, linewidth=linewidth, alpha=alpha)
                    
                    # 箭头位置
                    arrow_pos = [(src_pos_offset[0] + dst_pos_offset[0])/2, 
                            (src_pos_offset[1] + dst_pos_offset[1])/2, 
                            (src_pos_offset[2] + dst_pos_offset[2])/2]
                    arrow_direction = [dst_pos_offset[0] - src_pos_offset[0], 
                                    dst_pos_offset[1] - src_pos_offset[1], 
                                    dst_pos_offset[2] - src_pos_offset[2]]
                    
                    # 标准化箭头方向
                    length = np.sqrt(sum([d**2 for d in arrow_direction]))
                    if length > 0:
                        arrow_direction = [d/length * 20 for d in arrow_direction]
                        ax.quiver(arrow_pos[0], arrow_pos[1], arrow_pos[2], 
                                arrow_direction[0], arrow_direction[1], arrow_direction[2], 
                                color=color, arrow_length_ratio=0.4, linewidth=linewidth)
                    
                    # 仅为DATA包添加标签，位置更精确
                    if packet_type == "DATA" and packet_id < 10000:
                        # 在线的1/3处放置标签，远离无人机
                        label_pos = [
                            src_pos_offset[0] + (dst_pos_offset[0] - src_pos_offset[0]) * 0.33,
                            src_pos_offset[1] + (dst_pos_offset[1] - src_pos_offset[1]) * 0.33,
                            src_pos_offset[2] + (dst_pos_offset[2] - src_pos_offset[2]) * 0.33 + 5
                        ]
                        
                        # 使用不透明背景确保标签清晰可见
                        ax.text(label_pos[0], label_pos[1], label_pos[2], 
                            f"Pkt:{packet_id}", color='black', fontsize=14,
                            bbox=dict(facecolor='white', alpha=0.9, pad=2, edgecolor=color),
                            ha='center', va='center', zorder=100)  # 设置高zorder确保在最上层
        
        # 从comm_groups构建active_comms以便处理路由
        active_comms = {
            "DATA": [],
            "ACK": [],
            "HELLO": []
        }
        for packet_type, groups in comm_groups.items():
            for pair_key, comms in groups.items():
                for src_id, dst_id, packet_id in comms:
                    active_comms[packet_type].append((src_id, dst_id, packet_id))
        
        # 构建活跃的端到端路径
        active_routes = {}
        for packet_type, comms in active_comms.items():
            if packet_type == "DATA":
                for src_id, dst_id, packet_id in comms:
                    if packet_id < 10000:  # 只跟踪用户数据包
                        if packet_id not in active_routes:
                            active_routes[packet_id] = []
                        active_routes[packet_id].append((src_id, dst_id))
        
        # 绘制路由路径
        drawn_routes = set()
        for packet_id, hops in active_routes.items():
            if len(hops) > 1:  # 多跳路径
                # 按源节点组织跳数
                route_segments = {}
                for src, dst in hops:
                    route_segments[src] = dst
                
                # 尝试构建完整路径
                try:
                    # 假设第一跳的源是路径起点
                    start_node = hops[0][0]
                    path = [start_node]
                    current = start_node
                    
                    # 最多尝试n_drones次跳数（避免潜在的循环）
                    for _ in range(self.simulator.n_drones):
                        if current in route_segments:
                            next_node = route_segments[current]
                            path.append(next_node)
                            current = next_node
                        else:
                            break
                    
                    # 绘制完整路径，如果至少有3个节点
                    if len(path) >= 3:
                        route_key = tuple(path)
                        if route_key not in drawn_routes:
                            positions = [active_drones[node] for node in path]
                            xs, ys, zs = zip(*positions)
                            
                            # 绘制背景路径线
                            ax.plot(xs, ys, zs, 'y-', linewidth=4, alpha=0.3)
                            
                            # 完全改变路径标签位置 - 单独放置在图表侧面
                            # 而不是放在UAV或线路上
                            route_label = f"Route: {path[0]}→{path[-1]}"
                            
                            # 直接在终点位置上方显示路由标签
                            end_point = positions[-1]
                            ax.text(end_point[0], end_point[1], end_point[2] + 30, 
                                route_label, color='purple', fontsize=14,
                                bbox=dict(facecolor='yellow', alpha=0.9, pad=3, edgecolor='black'),
                                ha='center', va='center', weight='bold', zorder=101)  # 高zorder确保始终可见
                            
                            drawn_routes.add(route_key)
                except Exception as e:
                    # 如果路径构建失败，跳过
                    pass
        
        # 只包含实际可见的通信类型在图例中
        legend_elements = []
        for packet_type, color in self.comm_colors.items():
            # 只有在当前帧中有这种类型的通信时才添加到图例
            if packet_type in comm_groups and comm_groups[packet_type]:
                style = '-' if packet_type == "DATA" else '--' if packet_type == "ACK" else ':'
                legend_elements.append(Line2D([0], [0], color=color, linestyle=style, 
                                            label=f'{packet_type} Packets'))
        
        # 只有在当前帧有多跳路径时才添加该图例
        if drawn_routes:
            legend_elements.append(Line2D([0], [0], color='yellow', linewidth=4, alpha=0.3,
                                        label='Multi-hop Route'))
        
        # 添加图例，增大字号
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
        
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
        
        # 更新信息面板 - 强调是当前帧的值
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
        
    def save_frame_visualization(self, interval=0.1):
        """
        保存帧可视化（定期快照）
        
        参数:
            interval: 帧间隔（秒）
        """
        if not self.timestamps:
            print("No timestamps available")
            return []
        
        max_time = max(self.timestamps)
        min_time = min(self.timestamps)
        print(f"Time range: {min_time:.2f}s to {max_time:.2f}s")
        
        # 使用等间隔的时间点，确保覆盖整个时间范围
        time_points = np.arange(min_time, max_time + interval, interval)
        
        # 创建frames目录（如果不存在）
        frames_dir = os.path.join(self.output_dir, 'frames')
        os.makedirs(frames_dir, exist_ok=True)
        
        # 先清空frames目录中的所有旧帧
        for old_frame in os.listdir(frames_dir):
            if old_frame.endswith('.png'):
                os.remove(os.path.join(frames_dir, old_frame))
        
        total_frames = len(time_points)
        print(f"Starting frame generation: {total_frames} frames to create")
        
        # 重置历史数据
        self.pdr_history = []
        self.time_history = []
        self.collision_history = []
        self.sent_history = []
        self.received_history = []
        
        # 直接从日志中获取最终结果
        print("Checking for final metrics in simulator output...")
        final_sent = 0
        final_received = 0
        final_pdr = 0.0
        final_collisions = 0
        
        # 从属性中获取
        if hasattr(self.simulator, 'metrics'):
            metrics = self.simulator.metrics
            if hasattr(metrics, 'sent_packets'):
                final_sent = metrics.sent_packets
                print(f"Using metrics.sent_packets: {final_sent}")
                
            if hasattr(metrics, 'received_packets'):
                final_received = metrics.received_packets
                print(f"Using metrics.received_packets: {final_received}")
                
            if hasattr(metrics, 'pdr'):
                final_pdr = metrics.pdr
                print(f"Using metrics.pdr: {final_pdr}")
                
            if hasattr(metrics, 'collision_num'):
                final_collisions = metrics.collision_num
                print(f"Using metrics.collision_num: {final_collisions}")
        
        # 记录通信事件 - 确保它们具有正确的格式
        data_events = []  # 记录 (src_id, dst_id, packet_id, timestamp) 元组
        ack_events = []   # 记录 (packet_id, timestamp) 元组
        
        for event in self.comm_events:
            if len(event) != 5:
                continue
            
            src_id, dst_id, packet_id, packet_type, event_time = event
            
            if packet_type == "DATA":
                data_events.append((src_id, dst_id, packet_id, event_time))
            elif packet_type == "ACK":
                ack_events.append((packet_id, event_time))
        
        # 处理碰撞事件
        collision_events = []
        if hasattr(self, 'collision_events') and self.collision_events:
            for event in self.collision_events:
                if len(event) >= 2:
                    collision_events.append((event[1], event[0]))  # (time, location)
        collision_events.sort()  # 按时间排序
        
        # 使用实际数据或创建合成数据
        sent_by_time = []
        received_by_time = []
        
        # 获取已确认接收的包ID集合
        received_packet_ids = set(packet_id for packet_id, _ in ack_events)
        
        # 计算每个时间点的数据
        for t in time_points:
            # 计算当前时间的累计发送和接收数
            sent_at_time = sum(1 for _, _, _, time in data_events if time <= t)
            received_at_time = sum(1 for packet_id, time in ack_events if time <= t)
            
            # 如果有最终值但事件不足，使用线性插值
            if final_sent > 0 and (len(data_events) < final_sent):
                sent_growth_rate = final_sent / (max_time - min_time)
                elapsed_time = t - min_time
                sent_at_time = min(int(sent_growth_rate * elapsed_time), final_sent)
                
            if final_received > 0 and (len(ack_events) < final_received):
                received_growth_rate = final_received / (max_time - min_time)
                elapsed_time = t - min_time
                received_at_time = min(int(received_growth_rate * elapsed_time), final_received)
            
            sent_by_time.append(sent_at_time)
            received_by_time.append(received_at_time)
        
        # 计算每个时间点的PDR
        pdr_by_time = []
        for i in range(len(time_points)):
            if sent_by_time[i] > 0:
                pdr = (received_by_time[i] / sent_by_time[i]) * 100
            else:
                pdr = 0.0
            pdr_by_time.append(pdr)
        
        # 计算每个时间点的碰撞累计数
        collisions_by_time = []
        for t in time_points:
            # 计算当前时间的累计碰撞数
            collisions_at_time = sum(1 for time, _ in collision_events if time <= t)
            
            # 如果有最终值但事件不足，使用线性插值
            if final_collisions > 0 and (len(collision_events) < final_collisions):
                collision_growth_rate = final_collisions / (max_time - min_time)
                elapsed_time = t - min_time
                collisions_at_time = min(int(collision_growth_rate * elapsed_time), final_collisions)
                
            collisions_by_time.append(collisions_at_time)
        
        # 创建帧
        frames = []
        successful_frames = 0
        
        for time_idx, current_time in enumerate(time_points):
            frame_path = os.path.join(frames_dir, f'frame_{time_idx:04d}.png')
            
            # 创建图形
            fig = plt.figure(figsize=(12, 10))
            
            # 创建子图布局
            gs = fig.add_gridspec(3, 4)
            ax = fig.add_subplot(gs[:2, :3], projection='3d')
            info_ax = fig.add_subplot(gs[0, 3])
            pdr_ax = fig.add_subplot(gs[1, 3])
            collision_ax = fig.add_subplot(gs[2, :])
            
            # 设置标题
            ax.set_title(f'UAV Network Simulation at t={current_time:.2f}s')
            
            # 获取当前帧的无人机位置
            drone_positions = {}
            
            # 检查我们使用的数据结构类型
            for drone_id in range(len(self.drone_positions)):
                trajectory = self.drone_positions[drone_id]
                
                if isinstance(trajectory, dict):
                    # 字典类型 {timestamp: position}
                    closest_time = self._find_closest_time(trajectory, current_time)
                    if closest_time:
                        drone_positions[drone_id] = trajectory[closest_time]
                
                elif isinstance(trajectory, list):
                    # 列表类型 [position]
                    frame_idx = self._find_closest_time(trajectory, current_time)
                    if frame_idx is not None and 0 <= frame_idx < len(trajectory):
                        drone_positions[drone_id] = trajectory[frame_idx]
            
            # 筛选出当前时间点之前的有效无人机
            active_drones = {}
            for drone_id, position in drone_positions.items():
                if position is not None:
                    active_drones[drone_id] = position
            
            # 绘制无人机
            for drone_id, position in active_drones.items():
                ax.scatter(position[0], position[1], position[2], color='red', s=100, marker='o')
                
                # 为每个无人机添加标签
                ax.text(position[0], position[1], position[2] + 30, 
                       str(drone_id), color='black', fontsize=12, 
                       ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7))
            
            # 找出当前时间窗口内的通信事件
            current_comm_events = []
            
            for event in self.comm_events:
                if len(event) != 5:
                    continue
                
                src_id, dst_id, packet_id, packet_type, event_time = event
                
                # 仅包括当前时间窗口内发生的事件
                time_window = 0.3  # 显示过去0.3秒内的通信事件
                if current_time - time_window <= event_time <= current_time:
                    current_comm_events.append((src_id, dst_id, packet_id, packet_type, event_time))
            
            # 绘制通信事件
            packet_labels = {}
            for src_id, dst_id, packet_id, packet_type, event_time in current_comm_events:
                if src_id in active_drones and dst_id in active_drones:
                    src_pos = active_drones[src_id]
                    dst_pos = active_drones[dst_id]
                    
                    # 绘制从源到目的地的箭头
                    if packet_type == "DATA":
                        arrow = Arrow3D([src_pos[0], dst_pos[0]],
                                       [src_pos[1], dst_pos[1]],
                                       [src_pos[2], dst_pos[2]],
                                       mutation_scale=20,
                                       lw=2, arrowstyle="-|>", color="b")
                        ax.add_artist(arrow)
                        
                        # 记录当前正在发送的包
                        mid_x = (src_pos[0] + dst_pos[0]) / 2
                        mid_y = (src_pos[1] + dst_pos[1]) / 2
                        mid_z = (src_pos[2] + dst_pos[2]) / 2
                        
                        # 确保标签不重叠
                        offset = 15 * (len(packet_labels) % 3 + 1)
                        packet_labels[packet_id] = (mid_x, mid_y, mid_z + offset)
                    elif packet_type == "ACK":
                        arrow = Arrow3D([dst_pos[0], src_pos[0]],
                                       [dst_pos[1], src_pos[1]],
                                       [dst_pos[2], src_pos[2]],
                                       mutation_scale=20,
                                       lw=2, arrowstyle="-|>", color="g", linestyle='dashed')
                        ax.add_artist(arrow)
            
            # 为最近发送的数据包添加标签
            for packet_id, (x, y, z) in packet_labels.items():
                ax.text(x, y, z, f"Pkt:{packet_id}", color='blue', fontsize=10,
                       ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
            
            # 设置坐标轴范围 - 使用config如果可用，否则使用默认值
            config = getattr(self, 'config', None)
            map_length = getattr(config, 'MAP_LENGTH', 600) if config else 600
            map_width = getattr(config, 'MAP_WIDTH', 600) if config else 600
            map_height = getattr(config, 'MAP_HEIGHT', 600) if config else 600
            
            ax.set_xlim(0, map_length)
            ax.set_ylim(0, map_width)
            ax.set_zlim(0, map_height)
            
            # 设置轴标签
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            
            # 添加网格
            ax.grid(True)
            
            # 添加图例
            legend_elements = [
                Line2D([0], [0], color='b', lw=2, label='DATA Packets'),
                Line2D([0], [0], color='g', lw=2, linestyle='--', label='ACK Packets')
            ]
            ax.legend(handles=legend_elements, loc='upper right')
            
            # 显示仿真信息
            info_ax.axis('off')
            info_text = (
                f"Simulation Time: {current_time:.2f}s\n\n"
                f"Sent Packets: {sent_by_time[time_idx]}\n"
                f"Received Packets: {received_by_time[time_idx]}\n"
                f"Packet Delivery Ratio: {pdr_by_time[time_idx]:.2f}%\n"
                f"Total Collisions: {collisions_by_time[time_idx]}\n"
            )
            info_ax.text(0.05, 0.95, info_text, transform=info_ax.transAxes, 
                        fontsize=12, verticalalignment='top')
            
            # 绘制PDR历史图表
            if len(time_points) > 1:
                # 绘制历史曲线
                pdr_ax.plot(time_points[:time_idx+1], pdr_by_time[:time_idx+1], 'g-', linewidth=2)
                
                # 标记当前点
                pdr_ax.plot(current_time, pdr_by_time[time_idx], 'ro', markersize=8)
                
                # 美化图表
                pdr_ax.grid(True, linestyle='--', alpha=0.7)
                pdr_ax.set_xlabel('Time (s)')
                pdr_ax.set_ylabel('PDR (%)')
                pdr_ax.set_title('Packet Delivery Ratio')
                
                # 设置y轴范围，确保显示整个PDR范围
                pdr_ax.set_ylim(0, 105)
                
                # 设置x轴范围为整个时间段
                pdr_ax.set_xlim(min_time, max_time)
            
            # 绘制碰撞历史图表
            if len(time_points) > 1:
                # 绘制碰撞历史曲线
                collision_ax.plot(time_points[:time_idx+1], 
                                collisions_by_time[:time_idx+1], 'r-', linewidth=2)
                
                # 标记当前点
                collision_ax.plot(current_time, collisions_by_time[time_idx], 'bo', markersize=8)
                
                # 美化图表
                collision_ax.grid(True, linestyle='--', alpha=0.7)
                collision_ax.set_xlabel('Time (s)')
                collision_ax.set_ylabel('Collisions')
                collision_ax.set_title('Cumulative Collisions')
                
                # 确保Y轴上限能容纳最终的碰撞总数
                max_collisions = max(collisions_by_time)
                collision_ax.set_ylim(0, max_collisions * 1.1 or 1)  # 如果为0则设为1
                
                # 设置x轴范围为整个时间段
                collision_ax.set_xlim(min_time, max_time)
            
            # 显示状态信息
            status_text = f"Frame {time_idx+1}/{total_frames}"
            plt.figtext(0.5, 0.01, status_text, ha='center', fontsize=10)
            
            # 确保布局紧凑
            plt.tight_layout()
            
            # 保存帧
            plt.savefig(frame_path, dpi=100)
            plt.close(fig)
            
            frames.append(frame_path)
            successful_frames += 1
            
            # 每生成10个帧输出一次进度
            if (time_idx + 1) % 10 == 0 or time_idx == len(time_points) - 1:
                print(f"Generated {successful_frames}/{time_idx+1} frames")
        
        print(f"Successfully generated {successful_frames} frames out of {total_frames} time points")
        return frames
    
    def create_animations(self):
        """创建动画，从帧可视化快照创建GIF动画"""
        # 查找所有帧文件
        frames_dir = os.path.join(self.output_dir, 'frames')
        if not os.path.exists(frames_dir):
            print("No frames directory found. Run save_frame_visualization first.")
            return
        
        # 获取所有PNG帧，并按编号排序
        frame_files = [f for f in os.listdir(frames_dir) if f.endswith('.png') and f.startswith('frame_')]
        print(f"Found {len(frame_files)} frame files in {frames_dir}")
        
        if not frame_files:
            print("No frame files found")
            return
        
        # 按帧编号排序
        frames = sorted([
            os.path.join(frames_dir, f) for f in frame_files
        ], key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
        
        if frames:
            print(f"Creating animation from {len(frames)} frames...")
            gif_path = os.path.join(self.output_dir, 'uav_simulation.gif')
            
            # 使用PIL创建GIF
            # 读取第一帧以获取大小
            first_img = Image.open(frames[0])
            standard_size = first_img.size
            
            # 加载所有帧
            images = []
            for i, frame_path in enumerate(frames):
                img = Image.open(frame_path)
                # 确保所有帧大小一致
                if img.size != standard_size:
                    img = img.resize(standard_size, Image.LANCZOS)
                images.append(img)
                
                # 每100帧显示一次进度
                if (i+1) % 100 == 0 or i == len(frames) - 1:
                    print(f"Loaded {i+1}/{len(frames)} frames")
            
            if images:
                # 调整帧率，根据帧数
                if len(images) > 200:
                    fps = 20
                elif len(images) > 100:
                    fps = 15
                elif len(images) > 50:
                    fps = 10
                else:
                    fps = 5
                    
                print(f"Saving GIF with {len(images)} frames at {fps} fps...")
                
                # 保存GIF
                images[0].save(
                    gif_path, 
                    save_all=True, 
                    append_images=images[1:], 
                    duration=int(1000/fps), 
                    loop=0
                )
                print(f"Created animation: {gif_path}")
            else:
                print("No valid frames to create animation")
        else:
            print("No frames found for animation")
    
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
        
        # 保存帧可视化并获取帧列表
        frames = self.save_frame_visualization(interval=0.1)  # 使用0.1秒的间隔
        
        # 创建动画
        if frames:
            self.create_animations()
        else:
            print("No frames were generated for animation")
        
        print("Visualization complete!")

    def _find_closest_time(self, trajectory, target_time):
        """
        查找轨迹中最接近给定时间点的时间戳或索引
        
        参数:
            trajectory: 轨迹数据，可以是字典 {timestamp: position} 或列表 [position]
            target_time: 目标时间点
            
        返回:
            如果轨迹是字典，返回最接近的时间戳
            如果轨迹是列表，返回当前帧的索引 (按比例计算)
        """
        if not trajectory:
            return None
        
        # 检查轨迹类型
        if isinstance(trajectory, dict):
            # 字典类型 {timestamp: position}
            timestamps = sorted(trajectory.keys())
            
            # 如果目标时间小于第一个时间戳，返回第一个
            if target_time <= timestamps[0]:
                return timestamps[0]
            
            # 如果目标时间大于最后一个时间戳，返回最后一个
            if target_time >= timestamps[-1]:
                return timestamps[-1]
            
            # 线性查找最接近的时间戳
            for i in range(len(timestamps) - 1):
                if timestamps[i] <= target_time <= timestamps[i + 1]:
                    # 返回更接近的那个
                    if target_time - timestamps[i] < timestamps[i + 1] - target_time:
                        return timestamps[i]
                    else:
                        return timestamps[i + 1]
                    
            return timestamps[-1]  # 默认返回最后一个
        
        elif isinstance(trajectory, list):
            # 列表类型 [position]
            # 根据当前时间在时间范围内的比例，计算对应的索引
            # 假设轨迹中的每个点对应一个均匀的时间点
            total_frames = len(trajectory)
            
            if total_frames == 0:
                return None
            
            # 获取时间范围
            min_time = min(self.timestamps) if self.timestamps else 0
            max_time = max(self.timestamps) if self.timestamps else 1
            time_range = max_time - min_time
            
            if time_range <= 0:
                return 0  # 避免除以零错误
            
            # 计算目标时间对应的比例
            time_ratio = (target_time - min_time) / time_range
            
            # 将比例转换为索引
            frame_idx = int(time_ratio * (total_frames - 1))
            
            # 确保索引在有效范围内
            frame_idx = max(0, min(frame_idx, total_frames - 1))
            
            return frame_idx
        
        else:
            # 不支持的类型
            print(f"Warning: Unsupported trajectory type: {type(trajectory)}")
            return None