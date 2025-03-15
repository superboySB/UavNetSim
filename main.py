import simpy
import sys
from io import StringIO
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
    # Capture console output for later parsing
    original_stdout = sys.stdout
    console_output = StringIO()
    sys.stdout = console_output
    
    env = simpy.Environment()
    channel_states = {i: simpy.Resource(env, capacity=1) for i in range(config.NUMBER_OF_DRONES)}
    sim = Simulator(seed=2024, env=env, channel_states=channel_states, n_drones=config.NUMBER_OF_DRONES)
    
    visualizer = SimulationVisualizer(sim, output_dir="vis_results")
    sim.add_communication_tracking(visualizer)
    visualizer.run_visualization()

    env.run(until=config.SIM_TIME)
    
    # Store the console output in the simulator for the visualizer to access
    sim.output_log = console_output.getvalue()
    
    # Restore original stdout
    sys.stdout = original_stdout
    
    # Print the original output
    print(sim.output_log)
    
    # Store the console output in the visualizer as well (belt and suspenders)
    visualizer.console_output = sim.output_log
    
    visualizer.finalize()
