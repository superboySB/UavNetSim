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

# Custom output stream class that writes to both console and StringIO
class TeeOutput:
    def __init__(self, original_stream, capture_stream):
        self.original_stream = original_stream
        self.capture_stream = capture_stream
    
    def write(self, message):
        self.original_stream.write(message)
        self.capture_stream.write(message)
        self.original_stream.flush()  # Ensure real-time output
    
    def flush(self):
        self.original_stream.flush()
        self.capture_stream.flush()

if __name__ == "__main__":
    # Capture console output while still displaying it
    original_stdout = sys.stdout
    console_output = StringIO()
    sys.stdout = TeeOutput(original_stdout, console_output)
    
    env = simpy.Environment()
    channel_states = {i: simpy.Resource(env, capacity=1) for i in range(config.NUMBER_OF_DRONES)}
    sim = Simulator(seed=2024, env=env, channel_states=channel_states, n_drones=config.NUMBER_OF_DRONES)
    
    # Add the visualizer to the simulator
    visualizer = SimulationVisualizer(sim, output_dir="vis_results")
    sim.add_communication_tracking(visualizer)
    visualizer.run_visualization()

    env.run(until=config.SIM_TIME)
    
    # Store the console output in the simulator for the visualizer to access
    visualizer.console_output = console_output.getvalue()
    visualizer.finalize()