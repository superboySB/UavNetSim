import simpy
from utils import config
from simulator.simulator import Simulator
from visualization.visualizer import SimulationVisualizer

"""
  _   _                   _   _          _     ____    _             
 | | | |   __ _  __   __ | \ | |   ___  | |_  / ___|  (_)  _ __ ___  
 | | | |  / _` | \ \ / / |  \| |  / _ \ | __| \___ \  | | | '_ ` _ \ 
 | |_| | | (_| |  \ V /  | |\  | |  __/ | |_   ___) | | | | | | | | |
  \___/   \__,_|   \_/   |_| \_|  \___|  \__| |____/  |_| |_| |_| |_|
                                                                                                                                                                                                                                                                                           
"""

if __name__ == "__main__":
    # Simulation setup
    env = simpy.Environment()
    channel_states = {i: simpy.Resource(env, capacity=1) for i in range(config.NUMBER_OF_DRONES)}
    sim = Simulator(seed=2024, env=env, channel_states=channel_states, n_drones=config.NUMBER_OF_DRONES)
    
    # Add the visualizer to the simulator
    # Use 20000 microseconds (0.02s) as the visualization frame interval
    visualizer = SimulationVisualizer(sim, output_dir=".", vis_frame_interval=20000)
    visualizer.run_visualization()

    # Run simulation
    env.run(until=config.SIM_TIME)
    
    # Finalize visualization
    visualizer.finalize()