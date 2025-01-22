import math
import logging
from utils import config
from utils.util_function import euclidean_distance_3d, euclidean_distance_2d


# config logging
logging.basicConfig(filename='running_log.log',
                    filemode='w',  # there are two modes: 'a' and 'w'
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.DEBUG
                    )


def sinr_calculator(my_drone, main_drones_list, all_transmitting_drones_list):
    """
    calculate signal to signal-to-interference-plus-noise ratio
    :param my_drone: receiver drone
    :param main_drones_list: list of drones that wants to transmit packet to receiver
    :param all_transmitting_drones_list: list of all drones currently transmitting packet
    :return: list of sinr of each main drone
    """

    simulator = my_drone.simulator
    transmit_power = config.TRANSMITTING_POWER
    noise_power = config.NOISE_POWER

    sinr_list = []  # record the sinr of all transmitter
    receiver = my_drone

    logging.info('Main node list: %s', main_drones_list)
    for transmitter_id in main_drones_list:
        transmitter = simulator.drones[transmitter_id]
        interference_list = all_transmitting_drones_list[:]
        interference_list.remove(transmitter_id)

        main_link_path_loss = general_path_loss(receiver, transmitter)
        receive_power = transmit_power * main_link_path_loss
        interference_power = 0

        if len(interference_list) != 0:
            logging.info('Has interference')
            # my_drone.simulator.metrics.collision_num += 1
            for interference_id in interference_list:
                interference = simulator.drones[interference_id]

                logging.info('Main node is: %s, interference node is: %s, distance between them is: %s, main link' 
                             ' distance is: %s, interference link distance is: %s',
                             transmitter_id, interference_id, euclidean_distance_3d(transmitter.coords, interference.coords),
                             euclidean_distance_3d(transmitter.coords, receiver.coords),
                             euclidean_distance_3d(interference.coords, receiver.coords))

                interference_link_path_loss = general_path_loss(receiver, interference)
                interference_power += transmit_power * interference_link_path_loss
        else:
            logging.info('No interference, main link distance is: %s',
                         euclidean_distance_3d(transmitter.coords, receiver.coords))

        sinr = 10 * math.log10(receive_power / (noise_power + interference_power))
        logging.info('The SINR of main link is: %s', sinr)
        sinr_list.append(sinr)

    return sinr_list


def general_path_loss(receiver, transmitter):
    """
    general path loss model of line-of-sight (LoS) channels without system loss

    References:
        [1] J. Sabzehali, et al., "Optimizing number, placement, and backhaul connectivity of multi-UAV networks," in
            IEEE Internet of Things Journal, vol. 9, no. 21, pp. 21548-21560, 2022.

    :param receiver: the drone that receives the packet
    :param transmitter: the drone that sends the packet
    :return: path loss
    """

    c = config.LIGHT_SPEED
    fc = config.CARRIER_FREQUENCY
    alpha = 2  # path loss exponent

    distance = euclidean_distance_3d(receiver.coords, transmitter.coords)

    if distance != 0:
        path_loss = (c / (4 * math.pi * fc * distance)) ** alpha
    else:
        path_loss = 1

    return path_loss

def probabilistic_los_path_loss(receiver, transmitter):
    """
    probabilistic loss mode

    References:
        [1] A. Al-Hourani, S. Kandeepan and S. Lardner, "Optimal LAP Altitude for Maximum Coverage," in IEEE Wireless
            Communications Letters, vol. 3, no. 6, pp. 569-572, 2014.
        [2] J. Sabzehali, et al., "Optimizing number, placement, and backhaul connectivity of multi-UAV networks," in
            IEEE Internet of Things Journal, vol. 9, no. 21, pp. 21548-21560, 2022.

    :param receiver: the drone that receives the packet
    :param transmitter: the drone that sends the packet
    :return: path loss
    """

    c = config.LIGHT_SPEED
    fc = config.CARRIER_FREQUENCY
    alpha = 2  # path loss exponent
    eta_los = 0.1
    eta_nlos = 21
    a = 4.88
    b = 0.429

    distance = euclidean_distance_3d(receiver.coords, transmitter.coords)
    horizontal_dist = euclidean_distance_2d(receiver, transmitter)
    vertical_dist = max(receiver.coords[2], transmitter.coords[2])

    elevation_angle = math.atan(horizontal_dist / vertical_dist) * 180 / math.pi

    los_prob = 1 / (1 + a * math.exp(-b * (elevation_angle - a)))
    nlos_prob = 1 - los_prob

    if distance != 0:
        path_loss_los = ((c / (4 * math.pi * fc * distance)) ** alpha) * (10 ** (eta_los / 10))
        path_loss_nlos = ((c / (4 * math.pi * fc * distance)) ** alpha) * (10 ** (eta_nlos / 10))
    else:
        path_loss_los = 1
        path_loss_nlos = 1

    path_loss = los_prob * path_loss_los + nlos_prob * path_loss_nlos
    return path_loss


def maximum_communication_range():
    c = config.LIGHT_SPEED
    fc = config.CARRIER_FREQUENCY
    alpha = config.PATH_LOSS_EXPONENT  # path loss exponent
    transmit_power_db = 10 * math.log10(config.TRANSMITTING_POWER)
    noise_power_db = 10 * math.log10(config.NOISE_POWER)
    snr_threshold_db = config.SNR_THRESHOLD

    path_loss_db = transmit_power_db - noise_power_db - snr_threshold_db

    max_comm_range = (c * (10 ** (path_loss_db / (alpha * 10)))) / (4 * math.pi * fc)

    return max_comm_range
