import random
import numpy as np
from phy.channel import Channel
from entities.drone import Drone
from simulator.metrics import Metrics
from mobility import start_coords
from utils import config
from visualization.scatter import scatter_plot


class Simulator:
    """
    Description: simulation environment

    Attributes:
        env: simpy environment
        total_simulation_time: discrete time steps, in nanosecond
        n_drones: number of the drones
        channel_states: a dictionary, used to describe the channel usage
        channel: wireless channel
        metrics: Metrics class, used to record the network performance
        drones: a list, contains all drone instances

    Author: Zihao Zhou, eezihaozhou@gmail.com
    Created at: 2024/1/11
    Updated at: 2024/8/16
    """

    def __init__(self,
                 seed,
                 env,
                 channel_states,
                 n_drones,
                 total_simulation_time=config.SIM_TIME):

        self.env = env
        self.seed = seed
        self.total_simulation_time = total_simulation_time  # total simulation time (ns)

        self.n_drones = n_drones  # total number of drones in the simulation
        self.channel_states = channel_states
        self.channel = Channel(self.env)

        self.metrics = Metrics(self)  # use to record the network performance

        start_position = start_coords.get_random_start_point_3d(seed)

        self.drones = []
        for i in range(n_drones):
            if config.HETEROGENEOUS:
                speed = random.randint(5, 60)
            else:
                speed = 20

            print('UAV: ', i, ' initial location is at: ', start_position[i], ' speed is: ', speed)
            drone = Drone(env=env, node_id=i, coords=start_position[i], speed=speed,
                          inbox=self.channel.create_inbox_for_receiver(i), simulator=self)
            self.drones.append(drone)

        scatter_plot(self)

        self.env.process(self.show_performance())
        self.env.process(self.show_time())

    def show_time(self):
        while True:
            print('At time: ', self.env.now / 1e6, ' s.')
            yield self.env.timeout(0.5*1e6)  # the simulation process is displayed every 0.5s

    def show_performance(self):
        yield self.env.timeout(self.total_simulation_time - 1)

        scatter_plot(self)

        self.metrics.print_metrics()

    def add_communication_tracking(self, visualizer):
        """Add tracking for communication events"""
        self.visualizer = visualizer
        
        # Save the original unicast_put method
        original_unicast_put = self.channel.unicast_put
        
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
            
            self.visualizer.track_communication(src_drone_id, dst_drone_id, packet_id, packet_type)
            
            return result
        
        # Replace the method
        self.channel.unicast_put = tracked_unicast_put
