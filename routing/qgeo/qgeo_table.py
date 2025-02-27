import logging
import math
import random
import numpy as np
from collections import defaultdict
from utils.util_function import euclidean_distance_3d


class QGeoTable:
    def __init__(self, env, my_drone, rng_routing):
        self.env = env
        self.my_drone = my_drone
        self.neighbor_table = defaultdict(list)

        # row: action id, column: destination id
        self.q_table = np.zeros((my_drone.simulator.n_drones, my_drone.simulator.n_drones))  # initialization
        self.entry_life_time = 1 * 1e6  # unit: us
        self.rng_routing = rng_routing

    # determine if the neighbor table is empty
    def is_empty(self):
        return not bool(self.neighbor_table)

    # get the updated time of certain item
    def get_updated_time(self, drone_id):
        if drone_id not in self.neighbor_table.keys():
            raise RuntimeError('This item is not in the neighbor table')
        else:
            return self.neighbor_table[drone_id][-1]

    def add_neighbor(self, hello_packet, cur_time):
        """
        Update the neighbor table according to the hello packet
        :param hello_packet: the received hello packet
        :param cur_time: the moment when the packet is received
        :return: none
        """

        if hello_packet.src_drone.identifier != self.my_drone.identifier:
            drone_id = hello_packet.src_drone.identifier
            position = hello_packet.cur_position
            velocity = hello_packet.cur_velocity

            self.neighbor_table[drone_id] = [position, velocity, cur_time]
        else:
            pass

    # delete the specified item
    def remove_neighbor(self, drone_id):
        del self.neighbor_table[drone_id]

    # determine whether a certain drone is one's neighbor
    def is_neighbor(self, drone_id):
        if drone_id in self.neighbor_table.keys():
            if self.get_updated_time(drone_id) + self.entry_life_time > self.env.now:  # valid neighbor
                return True
        else:
            return False

    # remove the expired item
    def purge(self):
        if self.is_empty():
            # it means that the neighbor table is empty
            return

        for key in list(self.neighbor_table):
            updated_time = self.get_updated_time(key)
            if updated_time + self.entry_life_time <= self.env.now:  # expired
                self.remove_neighbor(key)

    # clear neighbor table
    def clear(self):
        self.neighbor_table.clear()

    # determine if the drone is in void area
    def void_area_judgment(self, dst_drone):
        flag = 1  # 1 means there is a void area
        distance_myself = euclidean_distance_3d(self.my_drone.coords, dst_drone.coords)

        for neighbor in self.neighbor_table.keys():
            neighbor_coords = self.neighbor_table[neighbor][0]
            distance_temp = euclidean_distance_3d(neighbor_coords, dst_drone.coords)
            if distance_temp <= distance_myself:
                flag = 0

        return flag

    # get the minimum Q-value of my neighbors
    def get_max_q_value(self, dst_drone_id):
        self.purge()

        max_q = -10000  # initial value
        for neighbor in self.neighbor_table.keys():
            max_q_temp = self.q_table[neighbor][dst_drone_id]
            if max_q_temp > max_q:
                max_q = max_q_temp

        return max_q

    def best_neighbor(self, my_drone, dst_drone):
        """
        Choose the best next hop according to the Q-table
        :param my_drone: the drone that installed the GPSR
        :param dst_drone: the destination of the data packet
        :return: none
        """

        self.purge()

        dst_id = dst_drone.identifier

        if self.is_empty():  # there is no neighbor drone
            best_id = my_drone.identifier
        else:
            if self.rng_routing.random() < 0.9 * math.pow(0.5, self.env.now / 1e6):
                best_id = self.rng_routing.choice(list(self.neighbor_table.keys()))
            else:
                best_q_value = -10000
                best_id = my_drone.identifier

                candidate_of_max_q_list = []

                for neighbor in self.neighbor_table.keys():
                    if neighbor != self.my_drone.identifier:  # cannot forward the packet to myself
                        next_hop_q_value = self.q_table[neighbor][dst_id]
                        if next_hop_q_value > best_q_value:
                            best_q_value = next_hop_q_value

                for neighbor in self.neighbor_table.keys():
                    if neighbor != self.my_drone.identifier:
                        if self.q_table[neighbor][dst_id] == best_q_value:
                            candidate_of_max_q_list.append(neighbor)

                if len(candidate_of_max_q_list) != 0:
                    best_id = self.rng_routing.choice(candidate_of_max_q_list)

        return best_id
