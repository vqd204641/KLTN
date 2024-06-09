import yaml
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from scipy.spatial import distance

from physical_env.mc.MobileCharger import MobileCharger
from physical_env.network.NetworkIO import NetworkIO
from optimizer.q_learning_heuristic import Q_learningv2
import sys
import os
import time
import copy
import networkx as nx
# from networkx.drawing import draw
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def log(net, mcs, q_learning):
    while True:
        yield net.env.timeout(100)
        print_state_net(net, mcs)
        #print(q_learning.list_request)


def print_state_net(net, mcs):
    print("[Network] Simulating time: {}s, lowest energy node is: id={} energy={:.2f} at {}".format(
        net.env.now, net.min_node(), net.listNodes[net.min_node()].energy, net.listNodes[net.min_node()].location))
    for mc in net.mc_list:
        if mc.chargingTime != 0 and mc.get_status() == "charging":
            print("\t\tMC #{} energy:{} is {} at {} state:{}".format(mc.id, mc.energy,
                                                                                         mc.get_status(),
                                                                                         mc.current,
                                                                                         mc.state))
        elif mc.moving_time != 0 and mc.get_status() == "moving":
            print("\t\tMC #{} energy:{} is {} to {} state:{}".format(mc.id, mc.energy,
                                                                                            mc.get_status(), mc.end,
                                                                                            mc.state))
        else:
            print("\t\tMC #{} energy:{} is {} at {} state:{}".format(mc.id, mc.energy, mc.get_status(), mc.current,
                                                                     mc.state))

networkIO = NetworkIO(r"C:/Users/Quoc Dat/Desktop/KLTN/KLTN/physical_env/network/network_scenarios/hanoi1000n50.yaml")
env, net = networkIO.makeNetwork()

with open(r"C:/Users/Quoc Dat/Desktop/KLTN/KLTN/physical_env/mc/mc_types/default.yaml",
          'r') as file:
    mc_argc = yaml.safe_load(file)
mcs = [MobileCharger(copy.deepcopy(net.baseStation.location), mc_phy_spe=mc_argc) for _ in range(3)]
#print(mc for mc in mcs)
q_learning = Q_learningv2(net=net, nb_action=37, lamda=0.5, q_alpha=0.1, q_gamma=0.1)

for id, mc in enumerate(mcs):
    mc.env = env
    mc.net = net
    mc.id = id
    mc.cur_phy_action = [net.baseStation.location[0], net.baseStation.location[1], 0]
    mc.state = q_learning.nb_action - 1
# Node:   50    100     150     200
# Center: 37    57      70      75
# Time:   20.8  7.5     5.4     2.9
print("start program")
net.mc_list = mcs
x = env.process(net.operate(optimizer=q_learning))
env.process(log(net, mcs, q_learning))
env.run(until=x)
