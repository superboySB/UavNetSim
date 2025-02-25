from entities.packet import Packet


class DsdvHelloPacket(Packet):
    def __init__(self,
                 src_drone,
                 creation_time,
                 id_hello_packet,
                 hello_packet_length,
                 packet_type,
                 routing_table,
                 simulator):
        super().__init__(id_hello_packet, hello_packet_length, creation_time, simulator)

        self.type = packet_type
        self.src_drone = src_drone
        self.routing_table = routing_table
