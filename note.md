# UavNetSim项目总结：状态输入、动作输出与优化目标

## 一、状态输入 (State)

### 1. 无人机物理状态
- **位置信息**：三维坐标 (x, y, z)
- **速度与方向**：
  - 飞行速度（`drone.speed`）
  - 飞行方向（`drone.direction`）
  - 俯仰角（`drone.pitch`）
  - 速度向量（`drone.velocity`）
- **能量状态**：
  - 剩余能量（`drone.residual_energy`）
  - 初始能量设置为20kJ（`config.INITIAL_ENERGY`）
  - 能量阈值为2000J（`config.ENERGY_THRESHOLD`）

### 2. 网络拓扑状态
- **邻居信息**：通过Hello包交换获取的邻居无人机表
- **链路状态**：
  - 信号强度
  - 信噪比（SINR）
  - 距离信息
- **信道状态**：忙/闲（通过CSMA/CA协议监控）

### 3. 通信队列状态
- **传输队列**：`drone.transmitting_queue`存储待发送的数据包
- **等待列表**：`drone.waiting_list`存储因无可用路由而等待的数据包
- **缓冲资源**：`drone.buffer`控制同一时间只能发送一个数据包

### 4. 路由信息
- **路由表**：根据所使用的路由协议（如DSDV、Greedy、Q-routing等）
- **Q值表**：Q-routing中记录到各目标的估计延迟Q值
- **包状态跟踪**：
  - 重传次数（`packet.number_retransmission_attempt`）
  - 数据包创建时间（`packet.creation_time`）
  - 已传输跳数（通过TTL推断）

## 二、动作输出 (Actions)

### 1. 移动控制动作
- **位置更新**：根据选定的移动模型周期性更新位置
  - Gauss-Markov移动模型：考虑历史移动状态的随机移动
  - 随机游走：纯随机方向变化
  - 随机路点：随机选择目标点并移动

### 2. 数据包生成与管理
- **数据包生成**：按泊松分布生成数据包（`generate_data_packet()`）
- **数据包入队**：将生成的数据包放入传输队列（`transmitting_queue.put()`）
- **数据包读取**：通过`feed_packet()`定期从队列取出数据包处理

### 3. 路由决策
- **下一跳选择**：`routing_protocol.next_hop_selection(packet)`
  - 贪心路由：选择距离目的地最近的邻居
  - Q-routing：选择Q值最小的邻居
  - DSDV：基于距离向量的路由选择
- **路由表更新**：
  - Q-routing的Q值更新（`update_q_table()`）
  - 其他路由协议的表项更新

### 4. MAC层控制
- **信道竞争**：CSMA/CA中的竞争窗口和退避计算
  - 随机退避时间（backoff）
  - 信道监听（`mac_protocol.mac_send()`）
- **数据包发送**：成功获取信道后发送数据包（`phy.unicast()`）
- **重传机制**：碰撞或未收到ACK时进行重传

### 5. 能量管理
- **能量监控**：周期性检查剩余能量（`energy_monitor()`）
- **睡眠机制**：当能量低于阈值时进入睡眠状态（`self.sleep = True`）
- **能耗计算**：
  - 通信能耗：`energy_consumption = (packet.packet_length / config.BIT_RATE) * config.TRANSMITTING_POWER`
  - 飞行能耗：基于速度和物理参数的能量消耗

## 三、优化目标 (Objectives)

### 1. 主要性能指标
- **数据包传递率(PDR)**：成功到达目的地的数据包比例
  - `pdr = len(metrics.datapacket_arrived) / metrics.datapacket_generated_num * 100`
- **端到端延迟**：从创建到接收的平均时间
  - `e2e_delay = np.mean(list(metrics.deliver_time_dict.values())) / 1e3`
- **网络吞吐量**：单位时间成功传输的数据量
  - `throughput = np.mean(list(metrics.throughput_dict.values())) / 1e3`
- **路由负载**：控制包与数据包的比率
  - `rl = metrics.control_packet_num / len(metrics.datapacket_arrived)`

### 2. 次要性能指标
- **平均跳数**：数据包从源到目的地经过的平均节点数
  - `hop_cnt = np.mean(list(metrics.hop_cnt_dict.values()))`
- **碰撞数量**：网络中数据包碰撞的总次数（`metrics.collision_num`）
- **MAC层延迟**：MAC层处理的平均延迟
  - `average_mac_delay = np.mean(metrics.mac_delay)`

### 3. 隐含优化目标
- **能量效率**：虽然未直接计算，但系统建模了能量消耗
  - 飞行能耗与通信能耗的平衡
  - 网络生命周期的最大化（避免节点因能量耗尽而失效）
- **网络连通性**：保持网络的连通性，避免网络分区
- **适应动态拓扑**：能够应对无人机移动导致的拓扑变化

## 四、强化学习实现细节

### 1. Q-routing实现原理
- **基本思想**：使用Q-learning近似端到端数据包传输延迟
- **状态空间**：(当前节点s, 目标节点d)
- **动作空间**：选择邻居节点n作为下一跳
- **Q值含义**：Q(s,d,n)表示从当前节点s通过邻居n到达目标d的估计延迟

### 2. 学习机制
- **在线学习**：每个无人机维护自己的Q表，通过实际交互更新Q值
- **经验分享**：通过ACK包携带邻居节点的状态信息
- **无经验回放**：不使用经验回放缓冲区，直接根据实时交互更新策略

### 3. Q值更新流程
1. 无人机x有发往目的地d的数据包
2. 通过查询Q表，选择具有最小Q值Q_x(d,y)的邻居y作为下一跳
3. 无人机x将数据包传输给y
4. 当y接收到来自x的数据包时，会回复ACK包，携带：
   - t = min Q_y(d,z)：表示y对到达目的地的剩余时间估计
   - q：在无人机x中的排队延迟
5. 当x接收到y的ACK包时，计算实际传输延迟s
6. 使用以下公式更新Q值：
   ```
   Q_x(d,y) = (1-α)·Q_x(d,y) + α·(q + s + t)
   ```
   其中α是学习率，控制Q值更新的程度

### 4. 代码实现核心
```python
# Q值更新函数
def update_q_table(self, packet, next_hop_id):
    data_packet_acked = packet.ack_packet
    dst_drone = data_packet_acked.dst_drone
    
    waiting_time = packet.queuing_delay
    
    transmitting_start_time = data_packet_acked.transmitting_start_time
    transmission_delay = self.simulator.env.now - transmitting_start_time
    
    min_q = packet.min_q
    
    # 目标检测：如果下一跳就是目的地，f=1；否则f=0
    if next_hop_id == dst_drone.identifier:
        f = 1
    else:
        f = 0
    
    # Q值更新公式
    self.table.q_table[next_hop_id][dst_drone.identifier] = \
        (1 - self.learning_rate) * self.table.q_table[next_hop_id][dst_drone.identifier] + \
        self.learning_rate * (waiting_time + transmission_delay + (1 - f) * min_q)
```

### 5. 探索与利用
- **ε-贪心策略**：以一定概率随机选择下一跳，以探索新路径
- **自适应学习率**：根据网络条件动态调整学习率
- **初始Q值设置**：所有Q值初始化为较高值（例如30000），鼓励探索未知路径
- **邻居有效性检查**：使用生存时间（entry_life_time）机制维护有效邻居

### 6. 高级特性
- **处理拓扑变化**：快速适应无人机移动导致的拓扑变化
- **处理链路故障**：当链路状态恶化时能够重新选择路由
- **拥塞感知**：通过排队延迟感知网络拥塞并调整路由
- **跨层优化**：结合MAC层信息（如信道状态）进行决策

### 7. 与传统路由对比
- **自适应性**：相比DSDV等静态路由协议，能更好适应动态网络环境
- **学习能力**：从网络交互中学习，而非依赖预设规则
- **分布式决策**：每个无人机独立做决策，无需中央控制
- **准最优性**：在长期运行后，能逐渐接近最优路由策略

### 8. 挑战与局限性
- **初始学习阶段**：早期性能可能不佳，需要时间学习
- **探索-利用权衡**：如何平衡已知最佳路径与探索新路径
- **状态空间爆炸**：在大规模网络中Q表可能变得庞大
- **非稳态环境**：拓扑频繁变化可能导致学习速度跟不上环境变化

## 五、QGeo路由算法

### 1. 基本思想
- **混合架构**：结合地理位置信息与Q-learning强化学习
- **状态空间**：包含无人机位置、速度等地理信息的增强状态空间
- **动作空间**：选择邻居节点作为下一跳
- **Q值含义**：Q(s,d,n)表示从当前节点s通过邻居n到达目标d的估计价值

### 2. 地理信息整合
- **位置预测**：根据无人机当前位置和速度预测未来位置
- **链接持久性**：评估通信链接在未来时刻的可靠性
- **距离计算**：使用三维欧氏距离计算无人机之间的距离
- **通信范围感知**：根据最大通信范围动态调整决策

### 3. 学习机制
- **基于未来位置的折扣因子**：根据预测的未来位置动态调整折扣因子
- **奖励函数**：结合传输成功率和链路质量设计奖励
- **Hello包增强**：Hello包携带位置和速度信息，用于更新邻居表
- **邻居表结构**：`neighbor_table[drone_id] = [position, velocity, timestamp]`

### 4. Q值更新流程
1. 无人机x有发往目的地d的数据包
2. 查询Q表选择合适的邻居y作为下一跳
3. 将数据包传输给y
4. 当接收到ACK包时，获取以下信息：
   - reward：根据传输质量计算的奖励
   - max_q：从下一跳可获得的最大Q值
5. 预测未来位置以计算动态折扣因子gamma
6. 使用以下公式更新Q值：
   ```
   Q_x(d,y) = (1-α)·Q_x(d,y) + α·(reward + gamma·(1-f)·max_q)
   ```
   其中f表示是否已到达目的地(f=1为已到达)

### 5. 代码实现核心
```python
def update_q_table(self, packet, next_hop_id, next_hop_coords, next_hop_velocity):
    data_packet_acked = packet.acked_packet
    dst_drone = data_packet_acked.dst_drone

    reward = packet.reward
    max_q = packet.max_q

    # 计算奖励函数
    if next_hop_id == dst_drone.identifier:
        f = 1
    else:
        f = 0

    # 计算动态折扣因子
    t = (math.ceil(self.simulator.env.now / self.hello_interval) * 
         self.hello_interval - self.simulator.env.now) / 1e6
    future_pos_myself = self.my_drone.coords + [i * t for i in self.my_drone.velocity]
    future_pos_next_hop = next_hop_coords + [i * t for i in next_hop_velocity]
    future_distance = util_function.euclidean_distance_3d(future_pos_myself, future_pos_next_hop)

    if future_distance < maximum_communication_range():
        gamma = 0.6  # 预计未来仍在通信范围内
    else:
        gamma = 0.4  # 预计未来可能超出通信范围

    # Q值更新公式
    self.table.q_table[next_hop_id][dst_drone.identifier] = \
        (1 - self.learning_rate) * self.table.q_table[next_hop_id][dst_drone.identifier] + \
        self.learning_rate * (reward + gamma * (1 - f) * max_q)
```

### 6. 高级特性
- **位置感知决策**：根据无人机相对位置和移动趋势做出更智能的路由决策
- **预测未来连接**：通过预测未来位置关系，避免选择即将断开连接的路径
- **自适应折扣因子**：根据链路持久性动态调整折扣因子，提高长期决策质量
- **快速适应拓扑变化**：能够迅速调整路由策略应对频繁的拓扑变化

### 7. 与Q-routing比较
- **状态表示**：QGeo包含更丰富的地理位置信息，而Q-routing主要关注节点标识
- **未来预测**：QGeo能预测未来位置关系，Q-routing主要基于当前观测
- **折扣策略**：QGeo使用动态折扣因子，Q-routing通常使用固定折扣
- **适用场景**：QGeo更适合高速移动的无人机网络，Q-routing适合相对稳定的网络
- **收敛速度**：QGeo在高动态环境中通常能更快收敛到可用策略

### 8. 挑战与优化方向
- **计算复杂度**：需要额外计算未来位置，增加了计算开销
- **参数调优**：折扣因子阈值(0.6/0.4)需要根据具体场景调整
- **平衡地理与学习信息**：如何有效整合地理信息和Q学习结果
- **处理预测不准确**：当移动模式突变时，位置预测可能不准确
- **扩展性研究**：与其他移动预测算法的结合潜力
