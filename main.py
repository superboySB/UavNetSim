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

# 指定要运行的路由协议列表
ROUTING_PROTOCOLS = ["q_routing", "qgeo", "greedy", "dsdv", "grad", "opar"]

if __name__ == "__main__":
    # 遍历每种路由协议，分别进行仿真
    for protocol in ROUTING_PROTOCOLS:
        print(f"\n{'='*50}")
        print(f"开始执行 {protocol} 路由协议仿真...")
        print(f"{'='*50}\n")
        
        # 为每个协议创建一个专用的输出目录
        output_dir = f"visualization_results/{protocol}"
        
        env = simpy.Environment()
        channel_states = {i: simpy.Resource(env, capacity=1) for i in range(config.NUMBER_OF_DRONES)}
        
        # 创建仿真器实例，并传递当前路由协议
        sim = Simulator(
            seed=2024, 
            env=env, 
            channel_states=channel_states, 
            n_drones=config.NUMBER_OF_DRONES,
            routing_protocol=protocol
        )
        
        # 创建可视化器并设置，使用协议专用的输出目录
        visualizer = SimulationVisualizer(sim, output_dir=output_dir)
        
        # 添加通信跟踪
        sim.add_communication_tracking(visualizer)
        
        # 启动可视化
        visualizer.run_visualization()
        
        # 运行仿真
        env.run(until=config.SIM_TIME)
        
        # 完成可视化
        visualizer.finalize()
        
        print(f"仿真完成！使用{protocol}路由协议。可视化结果已保存到 {output_dir}。")
        
        # 在仿真结束后打印指标
        print("----------- Metrics 类的属性 -----------")
        for attr in dir(sim.metrics):
            if not attr.startswith('__'):
                value = getattr(sim.metrics, attr)
                if not callable(value):  # 只打印非方法属性
                    print(f"{attr}: {value}")
        
        # 清理资源，为下一个协议准备
        del sim
        del visualizer
        del env
        del channel_states