import sys
import math
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from sympy import *
from opt import *

# 737-800 engine CFM56-7B27/3 characteristics
# Temp. at flat rating(F)  86 (30(C)

#----Take-off Conditions (Sea Level)----------
# max takeoff thrust 121436 N (MSL)
# bypass ratio 5.10  

#----- In-Flight Performance (35000ft-Mach=0.80-ISA)-----
# Max climb thrust 26511 (N)
# Overall pressure ratio  32.8
# Max Cruise Thrust  24376 (lbf) 

maxTakeoffThrust_30c = 121436

# Fuel fow 2178kg/h
fuelFlow = 2178

# TAS 450 knot 833.4km/h
TrueAirSpeed = 833.4

#latitude 纬度  longitude 经度  altitude 高度 temperature(C) 温度  air pressure(kpa) fuel weight(kg)   initial gross weight 77000kg cost 7000kg  
# [31.14500, 121.79333, 0, 30, 101.3, 77000], 
trueWayPointList = [[30.30167, 120.16333, 10668, -34, 24.0, 75000], [29.76333, 119.65833, 10668, -34, 24.0, 74400],
             [29.63167, 119.49000, 10668, -34, 24.0, 72900], [28.41667, 117.96667, 10668, -34, 24.0, 72350], [27.52000, 116.92167, 10668, -34, 24.0, 71600],
             [27.21667, 116.56667, 10668, -34, 24.0, 71000], [26.50167, 115.70833, 10668, -34, 24.0, 70450], [25.34833, 115.11167, 10668, -34, 24.0, 69850],
             [24.31333, 114.50667, 10668, -34, 24.0, 69200], [23.86833, 114.14667, 10668, -34, 24.0, 68600], [23.50833, 113.85667, 10668, -34, 24.0, 67800],
             [22.88833, 113.67000, 10668, -34, 24.0, 67500], [22.53167, 113.56333, 10668, -34, 24.0, 66900], [21.85000, 111.93333, 10668, -34, 24.0, 66300],
             [20.95333, 111.05500, 10668, -34, 24.0, 65700], [20.50500, 110.49500, 9540, -27.34, 28.4, 65100],
            #  [20.01333, 110.13667, 7224, -13.344], [19.25000, 109.83333, 4023, 5.862], [18.30295, 109.41227, 0.0, 30]]
             [20.01333, 110.13667, 7224, -13.344, 40.5, 64500], [19.25000, 109.83333, 4023, 5.862, 61.5, 64200], [18.30295, 109.41227, 0, 30, 101.3, 63900]]


trueSpeedList = [700, 770, 800]
# [speed, angel]
truePointWindList = [[144, 150], [144, 150], [144, 150], [144, 150], [144, 150], [144, 150], [144, 150], [144, 150], [144, 150], 
                    [144, 150], [216, 70], [216, 70], [216, 70], [216, 70], [216, 70], [216, 70], [216, 70], [216, 70], [216, 70]]


def find_lowest_cost(costs,to_process):
    lowest_cost_node = to_process[0]
    lowest_cost = costs[lowest_cost_node]
    if len(to_process)>1:
        for node in to_process[1:]:
            new_cost = costs[node]
            if new_cost < lowest_cost:
                lowest_cost = new_cost
                lowest_cost_node = node
    return lowest_cost_node

def initialize_costs_n_fathers(graph):
    costs, fathers ={},{}
    for node in graph:
        
        if node == '1003':
            costs[node] = 0
            fathers[node] = None
        else:
            costs[node] = float('inf')
    return costs,fathers


def get_shortest_path(fathers, fin):
    path = []
    reverPath = []
    father = fin
    reverPath.append(father)
    while father != '1003':
        father = fathers[father]
        reverPath.append(father)
    print ('Shortest path is:')
    for i in reverPath[-1:-len(reverPath):-1]:
        print ('{}-->'.format(i), end=' ')
        path.append(i)
    print (reverPath[0])
    path.append(reverPath[0])
    return path


def main():

    myWayPointGrid = GenerateWayPointGrid(trueWayPointList)
    wayPointGrid = myWayPointGrid.createWayPointGrid()
    # print(wayPointGrid) 
    print('-----------------------------')
    testWind = GenerateWindDataGrid(trueWayPointList, truePointWindList)
    windGrid = testWind.createWindDataGrid()
    # print(windGrid)
    # print('-----------------------------')

    setupGraph = setupFuelCostGraph(wayPointGrid, windGrid)
    initialGraph = setupGraph.initFuelCostGraph()
    initKeys = initialGraph.keys()
    for key in initialGraph.keys():
        print (key, initialGraph[key])
    # print(initKeys)
    fuelCostGraph = setupGraph.makeFuelCostGraph(initialGraph, trueSpeedList)
    # for i in fuelCostGraph.keys():
    #     print(i, fuelCostGraph[i])

    print ('Graph as below:')
    # print(fuelCostGraph)
    # print ('\n')
    costs, fathers = initialize_costs_n_fathers(fuelCostGraph)
    to_process = [i for i in fuelCostGraph.keys()]
    # print (to_process)
    fin = str(1000 + (len(trueWayPointList)-4)*10 +3)
    to_process.remove(fin)
    while to_process:
        node = find_lowest_cost(costs,to_process)
        neighbors = fuelCostGraph[node]
        for neighbor in neighbors:
            new_cost = costs[node] + fuelCostGraph[node][neighbor]
            if new_cost < costs[neighbor]:
                costs[neighbor] = new_cost
                fathers[neighbor] = node
            # print(costs)
            
        to_process.remove(node) # keys donnot share names
    # print(fathers)
    shortest_path = get_shortest_path(fathers, fin)
    print (shortest_path)
    print ('The lowest cost is {}.'.format(costs[fin]))

#--------------No-Fly Zone Avoidance-------------------------
#  [29.000, 118.700, 80]  [27.700, 117.800, 50]
    NFZ = [27.700, 117.800, 50]
    NFZRadiusDe = convertMKtoDegree(NFZ[2])

    avoidNFZ = NFZAvoidance(wayPointGrid, shortest_path, NFZ)
    theFlightPath, wayPointOrder = avoidNFZ.createflightPathPoints()

    print('the original flight path', theFlightPath)

    pointInNFZStatus, pointsInNFZ = avoidNFZ.checkPointInNFZ(theFlightPath)
    if pointInNFZStatus == True:
        print('way point %s in the No-Fly Zone!!! Reset flight path authenticated by ATC!' %(pointsInNFZ))

    crossNFZStatus, NFZPath, pointOrder= avoidNFZ.checkPathCrossNFZ(theFlightPath)
    print(crossNFZStatus, NFZPath, pointOrder)
    if crossNFZStatus == True:
        print('Path between %s crosses the No-Fly Zone' %(NFZPath))

        avoidancePoints = avoidNFZ.createAoidancePoint(NFZPath)
        print(avoidancePoints)

        selectPoint = avoidNFZ.selectOneAoidance(NFZPath, avoidancePoints, pointOrder, truePointWindList, trueSpeedList)
        print (selectPoint)
        theFlightPath = avoidNFZ.insertAvoidanceInPath(pointOrder, theFlightPath, selectPoint)
        print(theFlightPath)

# -------first phase distance-------------------
    # m = int(shortest_path[-1])//10
    # n = int(shortest_path[-1])%10
    # k = int(shortest_path[-2])//10
    # j = int(shortest_path[-2])%10
    

    # print(m, n-1, k, j-1)
    # print(wayPointGrid[m][n-1])
    # print(wayPointGrid[k][j-1])
    
    # first_distance = calculatePointDistance(wayPointGrid[m][n-1], wayPointGrid[k][j-1])
    # first_GSWindAngel = setupGraph.calculateGSWindAngel(wayPointGrid[m][n-1], \
    #                             wayPointGrid[k][j-1], windGrid[m][n-1][1])
    # first_GroundSpeed = setupGraph.calculateGroundSpeed(first_GSWindAngel, windGrid[m][n-1][0],\
    #                             851.92)
    # print(first_distance, first_GSWindAngel, first_GroundSpeed) 
#---------last phase distance------------------
    # q = int(shortest_path[1])//10
    # w = int(shortest_path[1])%10
    # e = int(shortest_path[0])//10
    # r = int(shortest_path[0])%10
    # print(q, w-1, e, r-1)
    # print(wayPointGrid[q][w-1])
    # print(wayPointGrid[e][r-1])
    # last_distance = calculatePointDistance(wayPointGrid[q][w-1], wayPointGrid[e][r-1])
    # last_GSWindAngel = setupGraph.calculateGSWindAngel(wayPointGrid[q][w-1], \
    #                             wayPointGrid[e][r-1], windGrid[q][w-1][1])
    # last_GroundSpeed = setupGraph.calculateGroundSpeed(first_GSWindAngel, windGrid[q][w-1][0],\
    #                             851.92)
    # print(last_distance, last_GSWindAngel, last_GroundSpeed) 
    
#----------------------------------------------------------------

    theta = np.arange(0, 2*np.pi, 0.01)
    NFZ_x = NFZ[1] + NFZRadiusDe * np.cos(theta)
    NFZ_y = NFZ[0] + NFZRadiusDe * np.sin(theta) 
    lats = []
    lons = []
    alts = []
    pathLats = []
    pathLons = []
    pathAlts = []
    origPathLats = []
    origPathLons = []
    origPathAlts = []
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlim3d(108, 123)
    ax.set_ylim3d(18, 32)
    ax.set_zlim3d(0, 12000)

    for way in trueWayPointList:
        origPathLats.append(way[0])
        origPathLons.append(way[1])
        origPathAlts.append(way[2])
    for point in theFlightPath:
        pathLats.append(point[0])
        pathLons.append(point[1])
        pathAlts.append(point[2])
    for layer in range(3):
        for way in range (len(trueWayPointList)):
            for group in range (5):
                lats.append(wayPointGrid[layer][way][group][0])
                lons.append(wayPointGrid[layer][way][group][1])
                alts.append(wayPointGrid[layer][way][group][2])
    if crossNFZStatus == True:
        avoid_x = [0, 0, 0, 0]
        avoid_y = [0, 0, 0, 0]
        avoid_z = [0, 0, 0, 0]
        for i in range(4):
            avoid_x[i] = avoidancePoints[i][1]
            avoid_y[i] = avoidancePoints[i][0]
            avoid_z[i] = selectPoint[2]
        # print(void_x, void_y)
        # ax.scatter(avoid_x, avoid_y, avoid_z, 'g')
        ax.scatter(selectPoint[1], selectPoint[0], selectPoint[2], 'b') 
    ax.scatter(lons, lats, alts, c='orange', marker='o')
    ax.plot3D(pathLons, pathLats, pathAlts, c='blue', linestyle='-')
    ax.plot3D(origPathLons, origPathLats, origPathAlts, c='green', linestyle='--')
    ax.set_xlabel('Longitude (green dash: original path; blue line: Opt path; red circle: NFZ)')
    ax.set_ylabel('Latitude (degree)')
    ax.set_zlabel('Altitude (meter)')
    ax.plot3D(NFZ_x, NFZ_y, selectPoint[2], 'r')
    
    # ax = fig.gca(projection='3d')

    # Make the grid
    x, y, z = np.meshgrid(np.arange(108, 123, 4),
                      np.arange(18, 32, 3),
                      np.arange(11277, 12000, 11277))
    # Make the direction data for the arrows
    u = np.cos((110 + (y - 23.7)/abs(y - 23.7)* 40) * np.pi/180)
    v = np.sin((110 + (y - 23.7)/abs(y - 23.7)* 40) * np.pi/180)
    w = 0
    ax.quiver(x, y, z, u, v, w, length=0.7, normalize=True)
    plt.title('Flight Opt App Simulation(from Pudong to Sanya)')
    plt.show()

if __name__=='__main__':
    main()
