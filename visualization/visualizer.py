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
        
        for idx, current_time in enumerate(time_points):
            # 创建图形
            fig, ax = plt.subplots(figsize=(10, 8))
            
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
                
            # 绘制网络图
            nx.draw_networkx_nodes(G, current_positions, node_size=500, 
                                  node_color=[to_rgba(self.colors[i]) for i in range(self.simulator.n_drones)])
            
            nx.draw_networkx_edges(G, current_positions, width=2, alpha=0.7, arrows=True, 
                                  arrowsize=15, arrowstyle='->')
            
            nx.draw_networkx_labels(G, current_positions, font_size=12, font_weight='bold')
            
            # 设置图形参数
            ax.set_title(f'UAV Communication Network at t={current_time:.1f}s')
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.axis('off')
            
            # 保存图形
            plt.savefig(os.path.join(self.output_dir, 'communications', f'comm_graph_{idx:04d}.png'), 
                        dpi=200, bbox_inches='tight')
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
        
        for idx, current_time in enumerate(time_points):
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # 为每个无人机绘制当前位置
            for i in range(self.simulator.n_drones):
                if not self.drone_positions[i]:
                    continue
                
                # 找到最接近当前时间的位置记录
                pos_times = np.array(self.timestamps)
                if len(pos_times) == 0:
                    continue
                    
                closest_idx = np.argmin(np.abs(pos_times - current_time))
                if closest_idx < len(self.drone_positions[i]):
                    x, y, z = self.drone_positions[i][closest_idx]
                    ax.scatter(x, y, z, color=self.colors[i], marker='o', s=300, label=f'UAV {i}')
                    
                    # 绘制轨迹线（显示历史路径）
                    if closest_idx > 0:
                        path_x, path_y, path_z = zip(*self.drone_positions[i][:closest_idx+1])
                        ax.plot(path_x, path_y, path_z, color=self.colors[i], alpha=0.5, linewidth=1)
            
            # 添加通信线
            window_links = []
            for link, timestamp in zip(self.active_links, self.link_timestamps):
                if current_time - interval <= timestamp <= current_time:
                    window_links.append(link)
            
            for src, dst in window_links:
                if (src < len(self.simulator.drones) and dst < len(self.simulator.drones) and 
                    self.drone_positions[src] and self.drone_positions[dst]):
                    
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
            
            # 设置图形参数
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title(f'UAV Network Simulation at t={current_time:.1f}s')
            ax.legend(loc='upper right')
            
            # 保存图形
            plt.savefig(os.path.join(self.output_dir, 'frames', f'frame_{idx:04d}.png'), 
                        dpi=200, bbox_inches='tight')
            plt.close(fig)
    
    def create_animations(self):
        """
        从保存的帧创建动画
        """
        try:
            # 创建3D轨迹动画
            frames_path = os.path.join(self.output_dir, 'frames')
            frames = sorted([os.path.join(frames_path, f) for f in os.listdir(frames_path) if f.endswith('.png')])
            
            if frames:
                gif_path = os.path.join(self.output_dir, 'uav_simulation.gif')
                
                # 使用PIL创建GIF
                imgs = [Image.open(f) for f in frames]
                imgs[0].save(gif_path, save_all=True, append_images=imgs[1:], duration=200, loop=0)
                
                print(f"创建动画: {gif_path}")
                
            # 创建通信网络动画
            comm_path = os.path.join(self.output_dir, 'communications')
            comm_frames = sorted([os.path.join(comm_path, f) for f in os.listdir(comm_path) if f.endswith('.png')])
            
            if comm_frames:
                comm_gif_path = os.path.join(self.output_dir, 'communication_network.gif')
                
                # 使用PIL创建GIF
                comm_imgs = [Image.open(f) for f in comm_frames]
                comm_imgs[0].save(comm_gif_path, save_all=True, append_images=comm_imgs[1:], duration=200, loop=0)
                
                print(f"创建通信网络动画: {comm_gif_path}")
                
        except Exception as e:
            print(f"创建动画时出错: {e}")
    
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
        print("保存可视化结果...")
        self.save_trajectory_plot()
        self.save_communication_graph()
        self.save_frame_visualization()
        self.create_animations()
        print("可视化完成！结果保存在目录:", self.output_dir)