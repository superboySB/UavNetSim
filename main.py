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
    env = simpy.Environment()
    channel_states = {i: simpy.Resource(env, capacity=1) for i in range(config.NUMBER_OF_DRONES)}
    sim = Simulator(seed=2024, env=env, channel_states=channel_states, n_drones=config.NUMBER_OF_DRONES)
    
    # 创建可视化器并设置
    visualizer = SimulationVisualizer(sim, output_dir="visualization_results")
    
    # 添加通信跟踪
    sim.add_communication_tracking(visualizer)
    
    # 启动可视化
    visualizer.run_visualization()
    
    # 运行仿真
    env.run(until=config.SIM_TIME)
    
    # 完成可视化
    visualizer.finalize()
    
    print("仿真完成！可视化结果已保存。")
    
    # 在仿真结束后添加以下代码
    print("----------- Metrics 类的属性 -----------")
    for attr in dir(sim.metrics):
        if not attr.startswith('__'):
            value = getattr(sim.metrics, attr)
            if not callable(value):  # 只打印非方法属性
                print(f"{attr}: {value}")