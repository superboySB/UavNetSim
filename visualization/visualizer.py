import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

# Add 3D arrow class definition that handles arrows in 3D view
class Arrow3D(FancyArrowPatch):
    """
    Class for drawing arrows in 3D view
    """
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        # Calculate average z value as depth
        avg_z = np.mean(zs)
        return avg_z
        
    def draw(self, renderer):
        FancyArrowPatch.draw(self, renderer)

class SimulationVisualizer:
    """
    Visualize UAV network simulation process, including movement trajectories and communication status
    """
    
    def __init__(self, simulator, output_dir="vis_results"):
        """
        Initialize visualizer
        
        Parameters:
            simulator: simulator instance
            output_dir: output directory
        """
        self.simulator = simulator
        self.output_dir = output_dir
        self.config = simulator.config if hasattr(simulator, 'config') else None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "frames"), exist_ok=True)
        
        # Initialize data storage structures
        self.drone_positions = {i: [] for i in range(self.simulator.n_drones)}
        self.timestamps = []
        
        # Comm events tracking
        self.comm_events = []  # Store tuples (src_id, dst_id, packet_id, packet_type, timestamp)
        
        # Assign a fixed color to each UAV
        self.colors = plt.cm.tab10(np.linspace(0, 1, self.simulator.n_drones))
        
        # Color mapping for communication types
        self.comm_colors = {
            "DATA": "blue",
            "ACK": "green",
            "HELLO": "orange"
        }
        
        # Setup communication tracking
        self._setup_communication_tracking()
    
    def _setup_communication_tracking(self):
        """Setup tracking for communication events"""
        # Save the original unicast_put method
        original_unicast_put = self.simulator.channel.unicast_put
        
        # Rewrite unicast_put method to track communications
        def tracked_unicast_put(message, dst_drone_id):
            # Call the original method
            result = original_unicast_put(message, dst_drone_id)
            
            # Record communication event
            packet, _, src_drone_id, _ = message
            
            # Add packet type differentiation
            packet_id = packet.packet_id
            packet_type = "DATA"
            
            # Identify different types of packets
            if packet_id >= 20000:
                packet_type = "ACK"
            elif packet_id >= 10000:
                packet_type = "HELLO"
            
            self.track_communication(src_drone_id, dst_drone_id, packet_id, packet_type)
            
            return result
        
        # Replace the method
        self.simulator.channel.unicast_put = tracked_unicast_put
    
    def track_drone_positions(self):
        """
        Record current drone positions
        """
        current_time = self.simulator.env.now / 1e6  # Convert to seconds
        self.timestamps.append(current_time)
        
        for i, drone in enumerate(self.simulator.drones):
            position = drone.coords  # This already contains (x, y, z) coordinates
            self.drone_positions[i].append(position)
    
    def track_communication(self, src_id, dst_id, packet_id, packet_type="DATA"):
        """
        Record communication event
        """
        current_time = self.simulator.env.now / 1e6  # Convert to seconds
        # Record complete communication event information
        self.comm_events.append((src_id, dst_id, packet_id, packet_type, current_time))
    
    def _generate_frame(self, time_idx, time_points):
        """Generate a single frame for the animation"""
        current_time = time_points[time_idx]
        
        # Create figure for this frame
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Set plot title with timestamp
        ax.set_title(f"UAV Network Simulation at t={int(current_time*1e6)}μs")
        
        # Set axis labels
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        
        # Get current frame drone positions
        drone_positions = {}
        
        # Find closest recorded position for each drone at current time
        for drone_id in range(len(self.drone_positions)):
            positions = self.drone_positions[drone_id]
            timestamps = self.timestamps
            
            if positions and timestamps:
                # Find closest timestamp
                closest_idx = min(range(len(timestamps)), 
                                key=lambda i: abs(timestamps[i] - current_time))
                
                # Get position at closest timestamp
                if 0 <= closest_idx < len(positions):
                    drone_positions[drone_id] = positions[closest_idx]
        
        # Draw drones
        for drone_id, position in drone_positions.items():
            ax.scatter(position[0], position[1], position[2], 
                      color=self.colors[drone_id], s=100, marker='o')
            
            # Add drone ID labels
            ax.text(position[0], position[1], position[2]+10, f"{drone_id}", 
                   color='black', fontsize=10, 
                   ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7))
        
        # Draw communication links
        recent_window = 0.05  # Show events from the last 50ms
        recent_comms = [e for e in self.comm_events 
                       if current_time - recent_window <= e[4] <= current_time]
        
        # Add arrows for communications
        for src_id, dst_id, packet_id, packet_type, event_time in recent_comms:
            if src_id in drone_positions and dst_id in drone_positions:
                src_pos = drone_positions[src_id]
                dst_pos = drone_positions[dst_id]
                
                # Choose color based on packet type
                arrow_color = self.comm_colors.get(packet_type, "gray")
                
                # Draw arrow for DATA packets
                if packet_type == "DATA":
                    arrow = Arrow3D([src_pos[0], dst_pos[0]], 
                                  [src_pos[1], dst_pos[1]], 
                                  [src_pos[2], dst_pos[2]],
                                  mutation_scale=15, 
                                  lw=2, arrowstyle="-|>", color=arrow_color)
                    ax.add_artist(arrow)
                    
                    # Add packet ID label near the middle of the arrow
                    mid_x = (src_pos[0] + dst_pos[0]) / 2
                    mid_y = (src_pos[1] + dst_pos[1]) / 2
                    mid_z = (src_pos[2] + dst_pos[2]) / 2
                    ax.text(mid_x, mid_y, mid_z, f"pkt:{packet_id}", 
                           color='blue', fontsize=8, 
                           bbox=dict(facecolor='white', alpha=0.7))
                
                # Draw curved arrow for ACK packets
                elif packet_type == "ACK":
                    # Create curved path (simple arc in 3D)
                    # Calculate midpoint with an offset
                    mid_x = (src_pos[0] + dst_pos[0]) / 2
                    mid_y = (src_pos[1] + dst_pos[1]) / 2
                    mid_z = (src_pos[2] + dst_pos[2]) / 2 + 30  # Offset upward
                    
                    ax.plot([src_pos[0], mid_x, dst_pos[0]], 
                           [src_pos[1], mid_y, dst_pos[1]], 
                           [src_pos[2], mid_z, dst_pos[2]], 
                           color=arrow_color, linestyle='-', marker='v', markevery=[1])
                
                # Draw star burst for HELLO packets (broadcast)
                elif packet_type == "HELLO":
                    ax.scatter(src_pos[0], src_pos[1], src_pos[2], 
                              color=arrow_color, marker='*', s=200, alpha=0.7)
        
        # Set axis range from map dimensions
        map_length = 600  # Default map dimensions
        map_width = 600
        map_height = 600
        
        ax.set_xlim(0, map_length)
        ax.set_ylim(0, map_width)
        ax.set_zlim(0, map_height)
        
        # Add grid
        ax.grid(True)
        
        # Add legend
        legend_elements = [
            Line2D([0], [0], color='blue', lw=2, label='DATA Packets'),
            Line2D([0], [0], color='green', marker='v', label='ACK Packets (curved)'),
            Line2D([0], [0], color='orange', marker='*', label='HELLO Packets (broadcast)')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Save frame
        frame_path = os.path.join(self.output_dir, 'frames', f'frame_{time_idx:04d}.png')
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        return frame_path

    def save_frame_visualization(self, interval=0.02):
        """
        Save frame visualization (periodic snapshot)
        
        Parameters:
            interval: Frame interval (seconds)
        """
        if not self.timestamps:
            print("No timestamps available")
            return []
        
        max_time = max(self.timestamps)
        min_time = min(self.timestamps)
        print(f"Time range: {min_time:.2f}s to {max_time:.2f}s")
        
        # Use evenly spaced time points to cover entire time range
        time_points = np.arange(min_time, max_time + interval, interval)
        
        # Create frames directory (if not exist)
        frames_dir = os.path.join(self.output_dir, 'frames')
        os.makedirs(frames_dir, exist_ok=True)
        
        # Clear all old frames in frames directory
        for old_frame in os.listdir(frames_dir):
            if old_frame.endswith('.png'):
                os.remove(os.path.join(frames_dir, old_frame))
        
        total_frames = len(time_points)
        print(f"Starting frame generation: {total_frames} frames to create")
        
        # Create frames
        frames = []
        successful_frames = 0
        
        for time_idx, current_time in enumerate(time_points):
            frame_path = self._generate_frame(time_idx, time_points)
            frames.append(frame_path)
            successful_frames += 1
            
            # Output progress every 10 frames
            if (time_idx + 1) % 10 == 0 or time_idx == len(time_points) - 1:
                print(f"Generated {successful_frames}/{total_frames} frames")
        
        print(f"Successfully generated {successful_frames} frames out of {total_frames} time points")
        return frames
    
    def create_animations(self):
        """Create animation, create GIF animation from frame visualization snapshot"""
        # Find all frame files
        frames_dir = os.path.join(self.output_dir, 'frames')
        if not os.path.exists(frames_dir):
            print("No frames directory found. Run save_frame_visualization first.")
            return
        
        # Get all PNG frames, and sort by number
        frame_files = [f for f in os.listdir(frames_dir) if f.endswith('.png') and f.startswith('frame_')]
        print(f"Found {len(frame_files)} frame files in {frames_dir}")
        
        if not frame_files:
            print("No frame files found")
            return
        
        # Sort frames by number
        frames = sorted([
            os.path.join(frames_dir, f) for f in frame_files
        ], key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
        
        if frames:
            print(f"Creating animation from {len(frames)} frames...")
            gif_path = os.path.join(self.output_dir, 'uav_simulation.gif')
            
            # Use PIL to create GIF
            # Read first frame to get size
            first_img = Image.open(frames[0])
            standard_size = first_img.size
            
            # Load all frames
            images = []
            for i, frame_path in enumerate(frames):
                img = Image.open(frame_path)
                # Ensure all frames consistent size
                if img.size != standard_size:
                    img = img.resize(standard_size, Image.LANCZOS)
                images.append(img)
                
                # Output progress every 100 frames
                if (i+1) % 100 == 0 or i == len(frames) - 1:
                    print(f"Loaded {i+1}/{len(frames)} frames")
            
            if images:
                # Adjust frame rate
                fps = 10
                    
                print(f"Saving GIF with {len(images)} frames at {fps} fps...")
                
                # Save GIF
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
    
    def run_visualization(self, tracking_interval=0.01):
        """
        Run visualization process
        
        Parameters:
            tracking_interval: Tracking position time interval (seconds)
        """
        # Convert to microseconds
        tracking_interval_us = tracking_interval * 1e6
        
        # Start tracking drone positions
        def track_positions():
            while True:
                self.track_drone_positions()
                yield self.simulator.env.timeout(tracking_interval_us)
        
        # Register tracking process
        self.simulator.env.process(track_positions())
    
    def finalize(self):
        """
        Finalize visualization, generate frames and animations
        """
        print("Finalizing visualization...")
        
        # Get frames (save frame graphics)
        self.save_frame_visualization()
        
        # Create animation
        self.create_animations()
        
        print("Visualization complete. Output saved to:", self.output_dir)