import simpy
import logging
import os
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

# 指定要运行的路由协议列表
ROUTING_PROTOCOLS = ["q_routing", "qgeo", "greedy", "dsdv", "grad", "opar"]

# 配置根日志记录器的通用设置
def configure_logger(protocol):
    # 创建日志目录
    log_dir = f"log_results/{protocol}"
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建日志文件路径
    log_file = f"{log_dir}/running_log.log"
    
    # 重置根日志记录器
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # 配置新的日志处理器
    logging.basicConfig(
        filename=log_file,
        filemode='w',  # 覆盖模式
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=config.LOGGING_LEVEL
    )
    
    logging.info(f"开始记录 {protocol} 路由协议的日志信息")

# 创建一个用于记录控制台输出的类
class OutputRecorder:
    def __init__(self, output_file):
        self.terminal = sys.stdout
        self.output_file = output_file
        self.log = []

    def write(self, message):
        self.terminal.write(message)
        self.log.append(message)
        
    def flush(self):
        self.terminal.flush()
        
    def save_to_file(self):
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write(''.join(self.log))

if __name__ == "__main__":
    # 创建日志根目录
    os.makedirs("log_results", exist_ok=True)
    
    # 遍历每种路由协议，分别进行仿真
    for protocol in ROUTING_PROTOCOLS:
        print(f"\n{'='*50}")
        print(f"开始执行 {protocol} 路由协议仿真...")
        print(f"{'='*50}\n")
        
        # 为每个协议配置专用的日志文件
        configure_logger(protocol)
        
        # 为每个协议创建一个专用的输出目录
        output_dir = f"log_results/{protocol}"
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置输出记录器
        original_stdout = sys.stdout
        recorder = OutputRecorder(f"{output_dir}/results.log")
        sys.stdout = recorder
        
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
        
        # 获取和保存关键指标
        metrics_output = f"""Totally send: {sim.metrics.datapacket_generated_num} data packets
Packet delivery ratio is: {len(sim.metrics.datapacket_arrived) / max(1, sim.metrics.datapacket_generated_num) * 100} %
Average end-to-end delay is: {sum(sim.metrics.deliver_time_dict.values()) / max(1, len(sim.metrics.deliver_time_dict)) / 1000} ms
Routing load is: {sim.metrics.control_packet_num / max(1, sim.metrics.datapacket_generated_num)}
Average throughput is: {sum(sim.metrics.throughput_dict.values()) / max(1, len(sim.metrics.throughput_dict))} Kbps
Average hop count is: {sum(sim.metrics.hop_cnt_dict.values()) / max(1, len(sim.metrics.hop_cnt_dict))}
Collision num is: {sim.metrics.collision_num}
Average mac delay is: {sum(sim.metrics.mac_delay) / max(1, len(sim.metrics.mac_delay))} ms"""
        
        # 保存指标到专用文件
        with open(f"{output_dir}/results.txt", "w") as f:
            f.write(metrics_output)
        
        print(metrics_output)
        print(f"仿真完成！使用{protocol}路由协议。可视化结果已保存到 {output_dir}。")
        print(f"日志已保存到 log_results/{protocol}/running_log.log")
        print(f"性能指标已保存到 log_results/{protocol}/results.txt")
        
        # 在仿真结束后打印指标
        print("----------- Metrics 类的属性 -----------")
        for attr in dir(sim.metrics):
            if not attr.startswith('__'):
                value = getattr(sim.metrics, attr)
                if not callable(value):  # 只打印非方法属性
                    print(f"{attr}: {value}")
                    
        # 保存输出到文件
        print(f"控制台输出已保存到 {output_dir}/results.log")

        
        # 清理资源，为下一个协议准备
        del sim
        del visualizer
        del env
        del channel_states