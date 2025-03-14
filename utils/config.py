import logging
from utils.ieee_802_11 import IeeeStandard

IEEE_802_11 = IeeeStandard().b_802_11  # 使用IEEE 802.11b标准参数

# --------------------- 仿真场景参数 --------------------- #
MAP_LENGTH = 600  # 地图长度，单位：米
MAP_WIDTH = 600   # 地图宽度，单位：米
MAP_HEIGHT = 600  # 地图高度，单位：米
SIM_TIME = 5 * 1e6  # 仿真总时长，单位：微秒 (15秒)
NUMBER_OF_DRONES = 5  # 网络中的无人机数量
STATIC_CASE = 0  # 是否模拟静态网络 (0:动态网络, 1:静态网络)
HETEROGENEOUS = 0  # 异构网络支持 (0:同构速度, 1:异构速度)
LOGGING_LEVEL = logging.INFO  # 日志级别，决定仿真过程中打印的详细信息

# ---------- 无人机硬件参数 (旋翼型) -----------#
PROFILE_DRAG_COEFFICIENT = 0.012  # 剖面阻力系数
AIR_DENSITY = 1.225  # 空气密度，单位：kg/m^3
ROTOR_SOLIDITY = 0.05  # 旋翼实度，定义为总叶片面积与盘面积之比
ROTOR_DISC_AREA = 0.79  # 旋翼盘面积，单位：m^2
BLADE_ANGULAR_VELOCITY = 400  # 叶片角速度，单位：弧度/秒
ROTOR_RADIUS = 0.5  # 旋翼半径，单位：米
INCREMENTAL_CORRECTION_FACTOR = 0.1  # 增量校正因子
AIRCRAFT_WEIGHT = 100  # 无人机重量，单位：牛顿
ROTOR_BLADE_TIP_SPEED = 500  # 旋翼叶片尖端速度
MEAN_ROTOR_VELOCITY = 7.2  # 悬停时的平均旋翼诱导速度
FUSELAGE_DRAG_RATIO = 0.3  # 机身阻力比
INITIAL_ENERGY = 20 * 1e3  # 初始能量，单位：焦耳 (20kJ)
ENERGY_THRESHOLD = 2000  # 能量阈值，单位：焦耳
MAX_QUEUE_SIZE = 200  # 无人机队列的最大容量

# ----------------------- 无线电参数 ----------------------- #
TRANSMITTING_POWER = 0.1  # 传输功率，单位：瓦特
LIGHT_SPEED = 3 * 1e8  # 光速，单位：m/s
CARRIER_FREQUENCY = IEEE_802_11['carrier_frequency']  # 载波频率，单位：赫兹
NOISE_POWER = 4 * 1e-11  # 噪声功率，单位：瓦特
RADIO_SWITCHING_TIME = 100  # 收发模式切换时间，单位：微秒
SNR_THRESHOLD = 6  # 信噪比阈值，单位：dB

# ---------------------- 数据包参数 ----------------------- #
MAX_TTL = NUMBER_OF_DRONES + 1  # 最大生存时间值
PACKET_LIFETIME = 10 * 1e6  # 数据包生命周期，单位：微秒 (10秒)
IP_HEADER_LENGTH = 20 * 8  # 网络层头部长度，20字节
MAC_HEADER_LENGTH = 14 * 8  # MAC层头部长度，14字节

# ---------------------- 物理层参数 -------------------------- #
PATH_LOSS_EXPONENT = 2  # 大尺度衰落的路径损耗指数
PLCP_PREAMBLE = 128 + 16  # 包括同步和SFD(帧开始界定符)
PLCP_HEADER = 8 + 8 + 16 + 16  # 包括信号、服务、长度和HEC(头错误检查)
PHY_HEADER_LENGTH = PLCP_PREAMBLE + PLCP_HEADER  # 物理层头部长度

ACK_HEADER_LENGTH = 16 * 8  # ACK包头部长度，16字节

DATA_PACKET_PAYLOAD_LENGTH = 1024 * 8  # 数据包有效载荷长度，1024字节
DATA_PACKET_LENGTH = IP_HEADER_LENGTH + MAC_HEADER_LENGTH + PHY_HEADER_LENGTH + DATA_PACKET_PAYLOAD_LENGTH  # 完整数据包长度

ACK_PACKET_LENGTH = ACK_HEADER_LENGTH + 14 * 8  # ACK包长度，单位：比特

HELLO_PACKET_PAYLOAD_LENGTH = 256  # Hello包有效载荷长度，单位：比特
HELLO_PACKET_LENGTH = IP_HEADER_LENGTH + MAC_HEADER_LENGTH + PHY_HEADER_LENGTH + HELLO_PACKET_PAYLOAD_LENGTH  # 完整Hello包长度

# 定义不同类型数据包的packet_id范围
GL_ID_HELLO_PACKET = 10000  # Hello包ID起始值
GL_ID_ACK_PACKET = 20000    # ACK包ID起始值
GL_ID_VF_PACKET = 30000     # 虚拟力包ID起始值
GL_ID_GRAD_MESSAGE = 40000  # 梯度路由消息ID起始值
GL_ID_CHIRP_PACKET = 50000  # Chirp包ID起始值

# ------------------ 物理层参数 ------------------- #
BIT_RATE = IEEE_802_11['bit_rate']  # 比特率，来自IEEE 802.11b标准
BIT_TRANSMISSION_TIME = 1/BIT_RATE * 1e6  # 单位比特传输时间，单位：微秒
BANDWIDTH = IEEE_802_11['bandwidth']  # 带宽，来自IEEE 802.11b标准
SENSING_RANGE = 750  # 感知范围，单位：米，定义发送节点可能干扰第三方节点传输的区域

# --------------------- MAC层参数 --------------------- #
SLOT_DURATION = IEEE_802_11['slot_duration']  # 时隙持续时间
SIFS_DURATION = IEEE_802_11['SIFS']  # 短帧间间隔持续时间
DIFS_DURATION = SIFS_DURATION + (2 * SLOT_DURATION)  # 分布式帧间间隔持续时间
CW_MIN = 31  # 初始竞争窗口大小
ACK_TIMEOUT = ACK_PACKET_LENGTH / BIT_RATE * 1e6 + SIFS_DURATION + 50  # ACK最大等待时间，单位：微秒
MAX_RETRANSMISSION_ATTEMPT = 5  # 最大重传尝试次数
