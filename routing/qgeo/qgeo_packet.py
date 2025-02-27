from entities.packet import Packet


class QGeoHelloPacket(Packet):
    def __init__(self,
                 src_drone,
                 creation_time,
                 id_hello_packet,
                 hello_packet_length,
                 simulator):
        super().__init__(id_hello_packet, hello_packet_length, creation_time, simulator)

        self.src_drone = src_drone
        self.cur_position = src_drone.coords
        self.cur_velocity = src_drone.velocity


class QGeoAckPacket(Packet):
    def __init__(self,
                 src_drone,
                 dst_drone,
                 ack_packet_id,
                 ack_packet_length,
                 acked_packet,
                 void_area_flag,
                 reward,
                 max_q,
                 simulator,
                 creation_time=None
                 ):
        super().__init__(ack_packet_id, ack_packet_length, creation_time, simulator)

        self.src_drone = src_drone
        self.src_coords = src_drone.coords
        self.src_velocity = src_drone.velocity

        self.dst_drone = dst_drone

        self.acked_packet = acked_packet

        self.void_area_flag = void_area_flag
        self.reward = reward
        self.max_q = max_q
