import simpy
import logging
import numpy as np
import random
import math
import queue
from entities.packet import DataPacket
from routing.dsdv.dsdv import Dsdv
from routing.greedy.greedy import Greedy
from routing.grad.grad import Grad
from routing.opar.opar import Opar
from routing.q_routing.q_routing import QRouting
from mac.csma_ca import CsmaCa
from mac.pure_aloha import PureAloha
from mobility.gauss_markov_3d import GaussMarkov3D
from mobility.random_walk_3d import RandomWalk3D
from mobility.random_waypoint_3d import RandomWaypoint3D
from topology.virtual_force.vf_motion_control import VfMotionController
from energy.energy_model import EnergyModel
from utils import config
from utils.util_function import has_intersection
from phy.large_scale_fading import sinr_calculator

# config logging
logging.basicConfig(filename='running_log.log',
                    filemode='w',  # there are two modes: 'a' and 'w'
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    level=config.LOGGING_LEVEL
                    )

GLOBAL_DATA_PACKET_ID = 0


class Drone:
    """
    Drone implementation

    Drones in the simulation are served as routers. Each drone can be selected as a potential source node, destination
    and relaying node. Each drone needs to install the corresponding routing module, MAC module, mobility module and
    energy module, etc. At the same time, each drone also has its own queue and can only send one packet at a time, so
    subsequent data packets need queuing for queue resources, which is used to reflect the queue delay in the drone
    network

    Attributes:
        simulator: the simulation platform that contains everything
        env: simulation environment created by simpy
        identifier: used to uniquely represent a drone
        coords: the 3-D position of the drone
        start_coords: the initial position of drone
        direction: current direction of the drone
        pitch: current pitch of the drone
        speed: current speed of the drone
        velocity: velocity components in three directions
        direction_mean: mean direction
        pitch_mean: mean pitch
        velocity_mean: mean velocity
        inbox: a "Store" in simpy, used to receive the packets from other drones (calculate SINR)
        buffer: used to describe the queuing delay of sending packet
        transmitting_queue: when the next hop node receives the packet, it should first temporarily store the packet in
                    "transmitting_queue" instead of immediately yield "packet_coming" process. It can prevent the buffer
                    resource of the previous hop node from being occupied all the time
        waiting_list: for reactive routing protocol, if there is no available next hop, it will put the data packet into
                      "waiting_list". Once the routing information bound for a destination is obtained, drone will get
                      the data packets related to this destination, and put them into "transmitting_queue"
        mac_protocol: installed mac protocol (CSMA/CA, ALOHA, etc.)
        mac_process_dict: a dictionary, used to store the mac_process that is launched each time
        mac_process_finish: a dictionary, used to indicate the completion of the process
        mac_process_count: used to distinguish between different "mac_send" processes
        enable_blocking: describe whether the process of waiting for an ACK blocks the delivery of subsequent packets
                         1: stop-and-wait protocol; 0: sliding window (need further implemented)
        routing_protocol: routing protocol installed (GPSR, DSDV, etc.)
        mobility_model: mobility model installed (3-D Gauss-markov, 3-D random waypoint, etc.)
        energy_model: energy consumption model installed
        residual_energy: the residual energy of drone in Joule
        sleep: if the drone is in a "sleep" state, it cannot perform packet sending and receiving operations.

    Author: Zihao Zhou, eezihaozhou@gmail.com
    Created at: 2024/1/11
    Updated at: 2025/2/24
    """

    def __init__(self, env, node_id, coords, speed, inbox, simulator, routing_protocol="greedy"):
        """
        Initialize the drone
        :param env: simpy environment
        :param node_id: identifier of the drone
        :param coords: coordinates of the drone (x, y, z)
        :param speed: speed (m/s)
        :param inbox: inbox of the drone
        :param simulator: simulation platform
        :param routing_protocol: routing protocol to use (greedy, q_routing, qgeo, etc.)
        """
        self.simulator = simulator
        self.env = env
        self.identifier = node_id
        self.coords = coords
        self.start_coords = coords

        self.rng_drone = random.Random(self.identifier + self.simulator.seed)

        self.direction = self.rng_drone.uniform(0, 2 * np.pi)
        self.pitch = self.rng_drone.uniform(-0.05, 0.05)
        self.speed = speed  # constant speed throughout the simulation
        self.velocity = [self.speed * math.cos(self.direction) * math.cos(self.pitch),
                         self.speed * math.sin(self.direction) * math.cos(self.pitch),
                         self.speed * math.sin(self.pitch)]

        self.direction_mean = self.direction
        self.pitch_mean = self.pitch
        self.velocity_mean = self.speed

        self.inbox = inbox

        self.buffer = simpy.Resource(env, capacity=1)
        self.max_queue_size = config.MAX_QUEUE_SIZE
        self.transmitting_queue = queue.Queue()  # queue in the real sense
        self.waiting_list = []

        self.mac_protocol = CsmaCa(self)
        self.mac_process_dict = dict()
        self.mac_process_finish = dict()
        self.mac_process_count = 0
        self.enable_blocking = 1  # enable "stop-and-wait" protocol

        # Initialize routing protocol based on provided parameter
        if routing_protocol == "q_routing":
            self.routing_protocol = QRouting(simulator, self)
        elif routing_protocol == "qgeo":
            from routing.qgeo.qgeo import QGeo  # 可能需要导入
            self.routing_protocol = QGeo(simulator, self)
        elif routing_protocol == "greedy":
            self.routing_protocol = Greedy(simulator, self)
        elif routing_protocol == "dsdv":
            self.routing_protocol = Dsdv(simulator, self)
        elif routing_protocol == "grad":
            self.routing_protocol = Grad(simulator, self)
        elif routing_protocol == "opar":
            self.routing_protocol = Opar(simulator, self)
        else:
            # 默认使用Greedy
            self.routing_protocol = Greedy(simulator, self)

        self.mobility_model = GaussMarkov3D(self)
        # self.motion_controller = VfMotionController(self)

        self.energy_model = EnergyModel()
        self.residual_energy = config.INITIAL_ENERGY
        self.sleep = False

        self.env.process(self.generate_data_packet())

        self.env.process(self.feed_packet())
        # self.env.process(self.energy_monitor())
        self.env.process(self.receive())

    def generate_data_packet(self, traffic_pattern='Poisson'):
        """
        Generate one data packet, it should be noted that only when the current packet has been sent can the next
        packet be started. When the drone generates a data packet, it will first put it into the "transmitting_queue",
        the drone reads a data packet from the head of the queue every very short time through "feed_packet()" function.
        :param traffic_pattern: characterize the time interval between generating data packets
        :return: none
        """

        global GLOBAL_DATA_PACKET_ID

        while True:
            if not self.sleep:
                if traffic_pattern == 'Uniform':
                    # the drone generates a data packet every 0.5s with jitter
                    yield self.env.timeout(self.rng_drone.randint(500000, 505000))
                elif traffic_pattern == 'Poisson':
                    """
                    the process of generating data packets by nodes follows Poisson distribution, thus the generation 
                    interval of data packets follows exponential distribution
                    """

                    rate = 5  # on average, how many packets are generated in 1s
                    yield self.env.timeout(round(self.rng_drone.expovariate(rate) * 1e6))

                GLOBAL_DATA_PACKET_ID += 1  # data packet id

                # randomly choose a destination
                all_candidate_list = [i for i in range(config.NUMBER_OF_DRONES)]
                all_candidate_list.remove(self.identifier)
                dst_id = self.rng_drone.choice(all_candidate_list)
                destination = self.simulator.drones[dst_id]  # obtain the destination drone

                pkd = DataPacket(self,
                                 dst_drone=destination,
                                 creation_time=self.env.now,
                                 data_packet_id=GLOBAL_DATA_PACKET_ID,
                                 data_packet_length=config.DATA_PACKET_LENGTH,
                                 simulator=self.simulator)
                pkd.transmission_mode = 0  # the default transmission mode of data packet is "unicast" (0)

                self.simulator.metrics.datapacket_generated_num += 1

                logging.info('------> UAV: %s generates a data packet (id: %s, dst: %s) at: %s, qsize is: %s',
                             self.identifier, pkd.packet_id, destination.identifier, self.env.now,
                             self.transmitting_queue.qsize())

                pkd.waiting_start_time = self.env.now

                if self.transmitting_queue.qsize() < self.max_queue_size:
                    self.transmitting_queue.put(pkd)
                else:
                    # the drone has no more room for new packets
                    pass
            else:  # cannot generate packets if "my_drone" is in sleep state
                break

    def blocking(self):
        """
        The process of waiting for an ACK will block subsequent incoming data packets to simulate the
        "head-of-line blocking problem"
        :return: none
        """

        if self.enable_blocking:
            if not self.mac_protocol.wait_ack_process_finish:
                flag = False  # there is currently no waiting process for ACK
            else:
                # get the latest process status
                final_indicator = list(self.mac_protocol.wait_ack_process_finish.items())[-1]

                if final_indicator[1] == 0:
                    flag = True  # indicates that the drone is still waiting
                else:
                    flag = False  # there is currently no waiting process for ACK
        else:
            flag = False

        return flag

    def feed_packet(self):
        """
        It should be noted that this function is designed for those packets which need to compete for wireless channel

        Firstly, all packets received or generated will be put into the "transmitting_queue", every very short
        time, the drone will read the packet in the head of the "transmitting_queue". Then the drone will check
        if the packet is expired (exceed its maximum lifetime in the network), check the type of packet:
        1) data packet: check if the data packet exceeds its maximum re-transmission attempts. If the above inspection
           passes, routing protocol is executed to determine the next hop drone. If next hop is found, then this data
           packet is ready to transmit, otherwise, it will be put into the "waiting_queue".
        2) control packet: no need to determine next hop, so it will directly start waiting for buffer

        :return: none
        """

        while True:
            if not self.sleep:  # if drone still has enough energy to relay packets
                yield self.env.timeout(10)  # for speed up the simulation

                if not self.blocking():
                    if not self.transmitting_queue.empty():
                        packet = self.transmitting_queue.get()  # get the packet at the head of the queue

                        if self.env.now < packet.creation_time + packet.deadline:  # this packet has not expired
                            if isinstance(packet, DataPacket):
                                if packet.number_retransmission_attempt[self.identifier] < config.MAX_RETRANSMISSION_ATTEMPT:
                                    # it should be noted that "final_packet" may be the data packet itself or a control
                                    # packet, depending on whether the routing protocol can find an appropriate next hop
                                    has_route, final_packet, enquire = self.routing_protocol.next_hop_selection(packet)

                                    if has_route:
                                        logging.info('UAV: %s obtain the next hop: %s of data packet (id: %s)',
                                                     self.identifier, packet.next_hop_id, packet.packet_id)

                                        # in this case, the "final_packet" is actually the data packet
                                        yield self.env.process(self.packet_coming(final_packet))
                                    else:
                                        self.waiting_list.append(packet)
                                        self.remove_from_queue(packet)

                                        if enquire:
                                            # in this case, the "final_packet" is actually the control packet
                                            yield self.env.process(self.packet_coming(final_packet))

                            else:  # control packet but not ack
                                yield self.env.process(self.packet_coming(packet))
                        else:
                            pass  # means dropping this data packet for expiration
            else:  # this drone runs out of energy
                break  # it is important to break the while loop

    def packet_coming(self, pkd):
        """
        When drone has a packet ready to transmit, yield it.

        The requirement of "ready" is:
            1) this packet is a control packet, or
            2) the valid next hop of this data packet is obtained
        :param pkd: packet that waits to enter the buffer of drone
        :return: none
        """

        if not self.sleep:
            arrival_time = self.env.now
            logging.info('Packet: %s waiting for UAV: %s buffer resource at: %s',
                         pkd.packet_id, self.identifier, arrival_time)

            with self.buffer.request() as request:
                yield request  # wait to enter to buffer

                logging.info('Packet: %s has been added to the buffer at: %s of UAV: %s, waiting time is: %s',
                             pkd.packet_id, self.env.now, self.identifier, self.env.now - arrival_time)

                pkd.number_retransmission_attempt[self.identifier] += 1

                if pkd.number_retransmission_attempt[self.identifier] == 1:
                    pkd.time_transmitted_at_last_hop = self.env.now

                logging.info('Re-transmission times of pkd: %s at UAV: %s is: %s',
                             pkd.packet_id, self.identifier, pkd.number_retransmission_attempt[self.identifier])

                # every time the drone initiates a data packet transmission, "mac_process_count" will be increased by 1
                self.mac_process_count += 1

                key=''.join(['mac_send', str(self.identifier), '_', str(pkd.packet_id)])

                mac_process = self.env.process(self.mac_protocol.mac_send(pkd))
                self.mac_process_dict[key] = mac_process
                self.mac_process_finish[key] = 0

                yield mac_process
        else:
            pass

    def energy_monitor(self):
        while True:
            yield self.env.timeout(1 * 1e5)  # report residual energy every 0.1s
            if self.residual_energy <= config.ENERGY_THRESHOLD:
                self.sleep = True
                # print('UAV: ', self.identifier, ' run out of energy at: ', self.env.now)

    def remove_from_queue(self, data_pkd):
        """
        After receiving the ack packet, drone should remove the data packet that has been acked from its queue
        :param data_pkd: the acked data packet
        :return: none
        """
        temp_queue = queue.Queue()

        while not self.transmitting_queue.empty():
            pkd_entry = self.transmitting_queue.get()
            if pkd_entry != data_pkd:
                temp_queue.put(pkd_entry)

        while not temp_queue.empty():
            self.transmitting_queue.put(temp_queue.get())

    def receive(self):
        """
        Core receiving function of drone
        1. the drone checks its "inbox" to see if there is incoming packet every 5 units (in us) from the time it is
           instantiated to the end of the simulation
        2. update the "inbox" by deleting the inconsequential data packet
        3. then the drone will detect if it receives a (or multiple) complete data packet(s)
        4. SINR calculation
        :return: none
        """

        while True:
            if not self.sleep:
                # delete packets that have been processed and do not interfere with
                # the transmission and reception of all current packets
                self.update_inbox()

                flag, all_drones_send_to_me, time_span, potential_packet = self.trigger()

                if flag:
                    # find the transmitters of all packets currently transmitted on the channel
                    transmitting_node_list = []
                    for drone in self.simulator.drones:
                        for item in drone.inbox:
                            packet = item[0]
                            insertion_time = item[1]
                            transmitter = item[2]
                            transmitting_time = packet.packet_length / config.BIT_RATE * 1e6
                            interval = [insertion_time, insertion_time + transmitting_time]

                            for interval2 in time_span:
                                if has_intersection(interval, interval2):
                                    transmitting_node_list.append(transmitter)

                    transmitting_node_list = list(set(transmitting_node_list))  # remove duplicates

                    sinr_list = sinr_calculator(self, all_drones_send_to_me, transmitting_node_list)

                    # receive the packet of the transmitting node corresponding to the maximum SINR
                    max_sinr = max(sinr_list)
                    if max_sinr >= config.SNR_THRESHOLD:
                        which_one = sinr_list.index(max_sinr)

                        pkd = potential_packet[which_one]

                        if pkd.get_current_ttl() < config.MAX_TTL:
                            sender = all_drones_send_to_me[which_one]

                            logging.info('Packet %s from UAV: %s is received by UAV: %s at time: %s, sinr is: %s',
                                         pkd.packet_id, sender, self.identifier, self.simulator.env.now, max_sinr)

                            yield self.env.process(self.routing_protocol.packet_reception(pkd, sender))
                        else:
                            logging.info('Packet %s is dropped due to exceeding max TTL', pkd.packet_id)
                    else:  # sinr is lower than threshold
                        self.simulator.metrics.collision_num += len(sinr_list)
                        pass

                yield self.env.timeout(5)
            else:
                break

    def update_inbox(self):
        """
        Clear the packets that have been processed.
                                           ↓ (current time step)
                              |==========|←- (current incoming packet p1)
                       |==========|←- (packet p2 that has been processed, but also can affect p1, so reserve it)
        |==========|←- (packet p3 that has been processed, no impact on p1, can be deleted)
        --------------------------------------------------------> time
        :return:
        """

        max_transmission_time = (config.DATA_PACKET_LENGTH / config.BIT_RATE) * 1e6  # for a single data packet
        for item in self.inbox:
            insertion_time = item[1]  # the moment that this packet begins to be sent to the channel
            received = item[3]  # used to indicate if this packet has been processed (1: processed, 0: unprocessed)
            if insertion_time + 2 * max_transmission_time < self.env.now:  # no impact on the current packet
                if received:
                    self.inbox.remove(item)

    def trigger(self):
        """
        Detects whether the drone has received a complete data packet
        :return:
        1. flag: bool variable, "1" means a complete data packet has been received by this drone and vice versa
        2. all_drones_send_to_me: a list, including all the sender of the complete data packets received
        3. time_span, a list, the element inside is the time interval in which the received complete data packet is
           transmitted in the channel
        4. potential_packet, a list, including all the instances of the received complete data packet
        """

        flag = 0  # used to indicate if I receive a complete packet
        all_drones_send_to_me = []
        time_span = []
        potential_packet = []

        for item in self.inbox:
            packet = item[0]  # not sure yet whether it has been completely transmitted
            insertion_time = item[1]  # transmission start time
            transmitter = item[2]
            processed = item[3]  # indicate if this packet has been processed
            transmitting_time = packet.packet_length / config.BIT_RATE * 1e6  # expected transmission time

            if not processed:  # this packet has not been processed yet
                if self.env.now >= insertion_time + transmitting_time:  # it has been transmitted completely
                    flag = 1
                    all_drones_send_to_me.append(transmitter)
                    time_span.append([insertion_time, insertion_time + transmitting_time])
                    potential_packet.append(packet)
                    item[3] = 1
                else:
                    pass
            else:
                pass

        return flag, all_drones_send_to_me, time_span, potential_packet
