import sys
import math
import os
from sympy import *

EARTH_RADIUS = 6371.0
class GenerateWayPointGrid:

    def __init__(self, wayPointList):
        self.wayPointList = wayPointList

    def createWayPointGroup(self, wayPointFrom, wayPointTo):
        y = wayPointFrom[0] - wayPointTo[0]
        x = wayPointFrom[1] - wayPointTo[1]
        wayPointGroup = [[0 for a in range(6)] for r in range (5)]
        slope = - x/y
        theta = math.atan(y/x)*180/math.pi
        abthe = abs(theta)
        if abthe < 30:
            coe = 0.1
        elif abthe > 60:
            coe = 0.3
        else:
            coe = 0.2

        for i in range (0,2):
            j = i + 1
            if theta >= 0:
                if x >= 0:
                    wayPointGroup[1-i][1] = wayPointTo[1] + coe*j
                    wayPointGroup[1-i][0] = wayPointTo[0] + coe*j*slope  
                    wayPointGroup[i+3][1] = wayPointTo[1] - coe*j  
                    wayPointGroup[i+3][0] = wayPointTo[0] - coe*j*slope
                else:
                    wayPointGroup[1-i][1] = wayPointTo[1] - coe*j
                    wayPointGroup[1-i][0] = wayPointTo[0] - coe*j*slope  
                    wayPointGroup[i+3][1] = wayPointTo[1] + coe*j  
                    wayPointGroup[i+3][0] = wayPointTo[0] + coe*j*slope
            else:
                if x >= 0:
                    wayPointGroup[1-i][1] = wayPointTo[1] - coe*j
                    wayPointGroup[1-i][0] = wayPointTo[0] - coe*j*slope  
                    wayPointGroup[i+3][1] = wayPointTo[1] + coe*j  
                    wayPointGroup[i+3][0] = wayPointTo[0] + coe*j*slope
                else:
                    wayPointGroup[1-i][1] = wayPointTo[1] + coe*j
                    wayPointGroup[1-i][0] = wayPointTo[0] + coe*j*slope  
                    wayPointGroup[i+3][1] = wayPointTo[1] - coe*j  
                    wayPointGroup[i+3][0] = wayPointTo[0] - coe*j*slope
            for k in range (2, 6):
                wayPointGroup[1-i][k] = wayPointTo[k]
                wayPointGroup[i+3][k] = wayPointTo[k]
        for w in range (6):
            wayPointGroup[2][w] = wayPointTo[w]
        return wayPointGroup

#-----------------false code----------------------                    
        # theta = math.atan(y/x)*180/math.pi
        # print (theta)
        # for i in range (0,2):
        #     j = i + 1
        #     if (0 <= theta < 30):
        #         if y > 0:
        #             wayPointGroup[1-i][0] = wayPointTo[0] - 0.3*j  
        #             wayPointGroup[1-i][1] = wayPointTo[1] + 0.1*j  
        #             wayPointGroup[i+3][0] = wayPointTo[0] + j*0.3    
        #             wayPointGroup[i+3][1] = wayPointTo[1] - 0.1*j
        #         else:
        #             wayPointGroup[1-i][0] = wayPointTo[0] - (0.1*j)
        #             wayPointGroup[1-i][1] = wayPointTo[1] + (0.3*j)
        #             wayPointGroup[i+3][0] = wayPointTo[0] + (0.1*j)    
        #             wayPointGroup[i+3][1] = wayPointTo[1] - (0.3*j)
                
        #     elif (30 <= theta < 60):
        #         if y >0:
        #             wayPointGroup[1-i][0] = wayPointTo[0] + (0.2*j)    
        #             wayPointGroup[1-i][1] = wayPointTo[1] - (0.2*j)
        #             wayPointGroup[i+3][0] = wayPointTo[0] - (0.2*j)    
        #             wayPointGroup[i+3][1] = wayPointTo[1] + (0.2*j)
        #         else:
        #             wayPointGroup[1-i][0] = wayPointTo[0] - (0.2*j)    
        #             wayPointGroup[1-i][1] = wayPointTo[1] + (0.2*j)
        #             wayPointGroup[i+3][0] = wayPointTo[0] + (0.2*j)    
        #             wayPointGroup[i+3][1] = wayPointTo[1] - (0.2*j)
        #     elif (60 <= theta < 90):
        #         if y > 0:
        #             wayPointGroup[1-i][0] = wayPointTo[0] + (0.3*j)    
        #             wayPointGroup[1-i][1] = wayPointTo[1] - (0.1*j)
        #             wayPointGroup[i+3][0] = wayPointTo[0] - (0.3*j)    
        #             wayPointGroup[i+3][1] = wayPointTo[1] + (0.1*j)
        #         else:
        #             wayPointGroup[1-i][0] = wayPointTo[0] - (0.3*j)
        #             wayPointGroup[1-i][1] = wayPointTo[1] + (0.1*j)
        #             wayPointGroup[i+3][0] = wayPointTo[0] + (0.3*j)    
        #             wayPointGroup[i+3][1] = wayPointTo[1] - (0.1*j)
        #     elif (-30 <= theta < 0):
        #         if y < 0:
        #             wayPointGroup[1-i][0] = wayPointTo[0] - (0.1*j)
        #             wayPointGroup[1-i][1] = wayPointTo[1] - (0.3*j)
        #             wayPointGroup[i+3][0] = wayPointTo[0] + (0.1*j)    
        #             wayPointGroup[i+3][1] = wayPointTo[1] + (0.3*j)
        #         else:
        #             wayPointGroup[1-i][0] = wayPointTo[0] + (0.1*j)
        #             wayPointGroup[1-i][1] = wayPointTo[1] + (0.3*j)
        #             wayPointGroup[i+3][0] = wayPointTo[0] - (0.1*j)    
        #             wayPointGroup[i+3][1] = wayPointTo[1] - (0.3*j)
        #     elif (-60 <= theta < -30):
        #         if y < 0:
        #             wayPointGroup[1-i][0] = wayPointTo[0] - (0.2*j)
        #             wayPointGroup[1-i][1] = wayPointTo[1] - (0.2*j)
        #             wayPointGroup[i+3][0] = wayPointTo[0] + (0.2*j)    
        #             wayPointGroup[i+3][1] = wayPointTo[1] + (0.2*j)
                    
        #         else:
        #             wayPointGroup[1-i][0] = wayPointTo[0] + (0.2*j)
        #             wayPointGroup[1-i][1] = wayPointTo[1] + (0.2*j)
        #             wayPointGroup[i+3][0] = wayPointTo[0] - (0.2*j)    
        #             wayPointGroup[i+3][1] = wayPointTo[1] - (0.2*j)
        #     else:
        #         if y < 0:
        #             wayPointGroup[1-i][0] = wayPointTo[0] - (0.3*j)
        #             wayPointGroup[1-i][1] = wayPointTo[1] - (0.1*j)
        #             wayPointGroup[i+3][0] = wayPointTo[0] + (0.3*j)    
        #             wayPointGroup[i+3][1] = wayPointTo[1] + (0.1*j)
        #         else:
        #             wayPointGroup[1-i][0] = wayPointTo[0] + (0.3*j)
        #             wayPointGroup[1-i][1] = wayPointTo[1] + (0.1*j)
        #             wayPointGroup[i+3][0] = wayPointTo[0] - (0.3*j)    
        #             wayPointGroup[i+3][1] = wayPointTo[1] - (0.1*j)  
        #     wayPointGroup[1-i][2] = wayPointTo[2]
        #     wayPointGroup[1-i][3] = wayPointTo[3]
        #     wayPointGroup[i+3][2] = wayPointTo[2]
        #     wayPointGroup[i+3][3] = wayPointTo[3]
        
        # wayPointGroup[2] = wayPointTo
        # print('-----------------')
        
#-------------------------------------------------------------------------
    
    def createWayPointGrid(self):
        wayPointCount = len(self.wayPointList)
        wayPointGrid = [[[[0 for a in range(6)] for r in range (5)] for y in range (wayPointCount)] for z in range(3)]
        
        for m in range (wayPointCount):
            if (m == 0) or ((m == wayPointCount - 1)):
                wayPointGrid[1][m][2] = self.wayPointList[m]
            else:
                wayPointGrid[1][m]  = self.createWayPointGroup(self.wayPointList[m-1], self.wayPointList[m])
        
        for m in range (wayPointCount):
            if (m == 0) or (m == (wayPointCount - 3)) or (m == (wayPointCount - 2)) or\
                     (m == (wayPointCount - 1)):
                wayPointGrid[0][m][2] = self.wayPointList[m]
                wayPointGrid[2][m][2] = self.wayPointList[m]
            else:
                wayPointGrid[0][m]  = self.createWayPointGroup(self.wayPointList[m-1], self.wayPointList[m])
                wayPointGrid[2][m]  = self.createWayPointGroup(self.wayPointList[m-1], self.wayPointList[m])
                for k in range (5):
                    wayPointGrid[0][m][k][2] = self.wayPointList[m][2] - 609
                    wayPointGrid[2][m][k][2] = self.wayPointList[m][2] + 609
                    wayPointGrid[0][m][k][3] = self.wayPointList[m][3] - 6 * 0.6
                    wayPointGrid[2][m][k][3] = self.wayPointList[m][3] + 6 * 0.6
        # for m in range (wayPointCount):
        #     if (m > 0) and (m < (wayPointCount - 3)):
                
                    # print(wayPointGrid[2][m][k][2])
        
        for n in range (3):
            for m in range(wayPointCount):
                for k in range(5):
                    print (n, m, k, wayPointGrid[n][m][k])
                print ('\n')
        return wayPointGrid


class GenerateWindDataGrid:
    def __init__(self, wayPointList, wayPointWindList):
        self.wayPointList = wayPointList
        self.wayPointWindList = wayPointWindList
    
    def createWindDataGrid(self):
        wayPointCount = len(self.wayPointList)
        # print(wayPointCount)
        windDataGrid = [[[0 for i in range(2)] for j in range(5)] for k in range (wayPointCount)]
        
        for i in range (wayPointCount):
            for j in range(0,5):
                windDataGrid[i][j][0] = self.wayPointWindList[i][0]
                windDataGrid[i][j][1] = self.wayPointWindList[i][1]
                # windDataGrid[i][j][2] = 0
        return(windDataGrid)


def calculatePointDistance(wayPointFrom, wayPointTo):
    # https://www.cnblogs.com/softfair/p/distance_of_two_latitude_and_longitude_points.html

    # EARTH_RADIUS = 6371.0
    lat1 = wayPointFrom[0] *math.pi/180
    lon1 = wayPointFrom[1] *math.pi/180
    lat2 = wayPointTo[0] *math.pi/180
    lon2 = wayPointTo[1] *math.pi/180
    dlon = abs(lon1 - lon2)
    dlat = abs(lat1 - lat2)
    sin2Dlat = math.sin(dlat/2)*math.sin(dlat/2)
    sin2Dlon = math.sin(dlon/2)*math.sin(dlon/2)
    h = sin2Dlat + math.cos(lat1)*math.cos(lat2)*sin2Dlon
    distance = 2 * (EARTH_RADIUS+wayPointFrom[2]/1000) * math.asin(math.sqrt(h))
    if wayPointFrom[2] != wayPointTo[2]:
        wayDefer = (wayPointFrom[2] - wayPointTo[2])/1000
        distance2 =  distance**2 + wayDefer**2
        distance = math.sqrt(distance2)
    return distance

def convertMKtoDegree(distance):
    degreeD = distance*360/(2*math.pi*EARTH_RADIUS)
    return degreeD

def convertDegreeToRadian(degree):
    return (degree *math.pi/180)

def calculateGroundSpeed(GSWindAngel, windSpeed, trueAirSpeed):
    theta = math.pi - GSWindAngel
    groundSpeed = trueAirSpeed * math.cos(math.asin(math.sin(theta)*windSpeed/trueAirSpeed)) \
                    - windSpeed * math.cos(theta)
    return(groundSpeed)

def calculateGSWindAngel(wayPointFrom, wayPointTo, windAngel):
    y = wayPointTo[0] - wayPointFrom[0]
    x = wayPointTo[1] - wayPointFrom[1]
    if x == 0:
        if y > 0:
            a = 90
        else:
            a = -90
    elif x < 0:
        a = math.atan(y/x)*180/math.pi
        if y >= 0:
            a = 180 + a
        else:
            a = a - 180
    else:
        a = math.atan(y/x)*180/math.pi
        
    beta = abs(a - windAngel)
    if beta > 180:
        beta = 360 - beta
    GSWindAngel = convertDegreeToRadian(beta)
    return (GSWindAngel)

# intial weight 77000kg  TAS = 833.4    
def calculateFuelFlow(TAS, weight, airPressure, temperature):
    cf1 = 0.9468/1000
    cf2 = 1000 *1.852/3.6
    cfcr = 0.9737
    cd0 = 0.020591
    cd2 = 0.051977
    g = 9.8       
    S = 124.58    #Wing reference area
    thou = 1.293*(airPressure/101.325) * 273.15/(temperature + 273.15)
    TAS = TAS/3.6
    eco = thou*S*TAS**2
    # print(eco, thou, airPressure, temperature)

    fcr = 60 * cf1 *(1 + TAS/cf2)* cfcr *(cd0 + cd2*(2*weight*g/eco)**2)/2*eco
    return fcr


class setupFuelCostGraph:
    def __init__(self, wayPointGrid, wayPointWindGrid):
        self.wayPointGrid = wayPointGrid
        self.wayPointWindGrid = wayPointWindGrid    

    def initFuelCostGraph(self):
        graph = {}
        # print(len(self.wayPointGrid[1]))
        for i in range (len(self.wayPointGrid[1])):

            for j in range (1,6):
                # Fuel cost grid 不同于 waypoint Grid 第二个下标从 ‘1’ 开始
                nodeName = [2*1000 + i*10 + j, 2*1000 + (i+1)*10 + j-1, \
                            2*1000 + (i+1)*10 + j, 2*1000 + (i+1)*10 + j+1, \
                            1*1000 + (i+1)*10 + j-1, 1*1000 + (i+1)*10 + j, 1*1000 + (i+1)*10 + j+1,]
                node = str(nodeName[0])
                nodeLeftNeighbor = str(nodeName[1])
                nodeNeighbor = str(nodeName[2])
                nodeRightNeighbor = str(nodeName[3])
                nodeDwLeftNeighbor = str(nodeName[4])
                nodeDwNeighbor = str(nodeName[5])
                nodeDwRightNeighbor = str(nodeName[6])
                if 0 == i:
                    pass 
                #     if 3 == j: 
                #         graph[node] = {nodeLeftNeighbor:0, nodeNeighbor:0, nodeRightNeighbor:0}
                elif 1 == i:
                    if 1 != j and 5 != j:
                        graph[node] = {nodeLeftNeighbor:0, nodeNeighbor:0, nodeRightNeighbor:0}
                elif (len(self.wayPointGrid[1]) - 6) == i:
                    if j == 1:
                        graph[node] = {nodeRightNeighbor:0}
                    elif j == 2:
                        graph[node] = {nodeNeighbor:0, nodeRightNeighbor:0}
                    elif j == 3:
                        graph[node] = {nodeLeftNeighbor:0, nodeNeighbor:0, nodeRightNeighbor:0}      
                    elif j == 4:
                        graph[node] = {nodeLeftNeighbor:0, nodeNeighbor:0}
                    else:
                        graph[node] = {nodeLeftNeighbor:0}
                elif (len(self.wayPointGrid[1]) - 5) == i:
                    if j == 2:
                        # graph[node] = {}
                        graph[node] = {nodeDwRightNeighbor:0}
                    if j == 3:
                        # graph[node] = {}
                        graph[node] = {nodeDwNeighbor:0}
                    if j == 4:
                        # graph[node] = {}
                        graph[node] = {nodeDwLeftNeighbor:0}
                elif ((len(self.wayPointGrid[1]) - 2) == i) or ((len(self.wayPointGrid[1]) - 3) == i) or\
                    ((len(self.wayPointGrid[1]) - 1) == i) or ((len(self.wayPointGrid[1]) - 4) == i):
                    pass
                    # if j == 3:
                    #     graph[node] = {nodeNeighbor:0}
                # elif (len(self.wayPointGrid[1]) - 1) == i:
                #     if j == 3:
                #         graph[node] = {}
                else:
                    if 1 == j:
                        graph[node] = {nodeNeighbor:0, nodeRightNeighbor:0}
                    elif 5 == j:
                        graph[node] = {nodeLeftNeighbor:0, nodeNeighbor:0}
                    else:
                        graph[node] = {nodeLeftNeighbor:0, nodeNeighbor:0, nodeRightNeighbor:0}
                

                for k in range(2):
                    nodeName = [k*1000 + i*10 + j, k*1000 + (i+1)*10 + j-1, k*1000 + (i+1)*10 + j, \
                                k*1000 + (i+1)*10 + j+1, (k+1)*1000 + (i+1)*10 + j-1, \
                                (k+1)*1000 + (i+1)*10 + j, (k+1)*1000 + (i+1)*10 + j+1]
                    node = str(nodeName[0])
                    nodeLeftNeighbor = str(nodeName[1])
                    nodeNeighbor = str(nodeName[2])
                    nodeRightNeighbor = str(nodeName[3])
                    nodeUpLeftNeighbor = str(nodeName[4])
                    nodeUpNeighbor = str(nodeName[5])
                    nodeUpRightNeighbor = str(nodeName[6])
                    if 0 == i: 
                        if 3 == j:
                            if k == 1: 
                                nodeDwLeftNeighbor = str(nodeName[1] - 1000)
                                nodeDwNeighbor = str(nodeName[2] - 1000)
                                nodeDwRightNeighbor = str(nodeName[3] - 1000)
                                graph[node] = {nodeDwLeftNeighbor:0, nodeDwNeighbor:0, nodeDwRightNeighbor:0, \
                                        nodeLeftNeighbor:0, nodeNeighbor:0, nodeRightNeighbor:0, \
                                        nodeUpLeftNeighbor:0, nodeUpNeighbor:0, nodeUpRightNeighbor:0}
                    elif 1 == i:
                        if 1 != j and 5 != j:
                            graph[node] = {nodeLeftNeighbor:0, nodeNeighbor:0, nodeRightNeighbor:0, \
                                        nodeUpLeftNeighbor:0, nodeUpNeighbor:0, nodeUpRightNeighbor:0}
                    elif (len(self.wayPointGrid[1]) - 6) == i:
                        if j == 1:
                            graph[node] = {nodeRightNeighbor:0, nodeUpRightNeighbor:0}
                        elif j == 2:
                            graph[node] = {nodeNeighbor:0, nodeRightNeighbor:0, nodeUpNeighbor:0, \
                                            nodeUpRightNeighbor:0}
                        elif j == 3:
                            graph[node] = {nodeLeftNeighbor:0, nodeNeighbor:0, nodeRightNeighbor:0, \
                                            nodeUpLeftNeighbor:0, nodeUpNeighbor:0, nodeUpRightNeighbor:0}      
                        elif j == 4:
                            graph[node] = {nodeLeftNeighbor:0, nodeNeighbor:0, nodeUpLeftNeighbor:0, \
                                            nodeUpNeighbor:0}
                        else:
                            graph[node] = {nodeLeftNeighbor:0, nodeUpLeftNeighbor:0}
                    elif (len(self.wayPointGrid[1]) - 5) == i:
                        if k == 0:
                            if j == 2:
                                graph[node] = {nodeUpRightNeighbor:0}
                            if j == 3:
                                graph[node] = {nodeUpNeighbor:0}
                            if j == 4:
                                graph[node] = {nodeUpLeftNeighbor:0}
                        else:
                            if j == 2:
                                graph[node] = {nodeRightNeighbor:0}
                            if j == 3:
                                graph[node] = {nodeNeighbor:0}
                            if j == 4:
                                graph[node] = {nodeLeftNeighbor:0}
                    elif (len(self.wayPointGrid[1]) - 4) == i:
                        if k == 1:
                            if j == 3:
                                graph[node] = {}
                    elif (len(self.wayPointGrid[1]) - 2) == i or (len(self.wayPointGrid[1]) - 3) == i:
                        pass
                        # if k == 1:
                        #     if j == 3:
                        #         graph[node] = {nodeNeighbor:0}
                    elif (len(self.wayPointGrid[1]) - 1) == i:
                        pass
                        # if k == 1:
                        #     if j == 3:
                        #         graph[node] = {}
                    else:
                        if 1 ==j:
                            graph[node] = {nodeNeighbor:0, nodeRightNeighbor:0, nodeUpNeighbor:0, \
                                            nodeUpRightNeighbor:0}
                        elif 5 == j:
                            graph[node] = {nodeLeftNeighbor:0, nodeNeighbor:0, nodeUpLeftNeighbor:0, \
                                            nodeUpNeighbor:0}
                        else:
                            graph[node] = {nodeLeftNeighbor:0, nodeNeighbor:0, nodeRightNeighbor:0, \
                                            nodeUpLeftNeighbor:0, nodeUpNeighbor:0, nodeUpRightNeighbor:0}
            # print(i, j)
        return (graph)


    def makeFuelCostGraph(self, FuelCostGraph, TAS):
        # FuelCostGraph = self.initFuelCostGraph()
        print('--------------- each fuel cost ---------------')

        for node in FuelCostGraph.keys():
            l = int(node) // 1000
            m = (int(node) - l*1000) // 10
            n = int(node) % 10
            
            neighbors = FuelCostGraph[node]
            for neighbor in neighbors:
                h = int(neighbor) // 1000
                k = (int(neighbor) - h*1000) // 10
                j = int(neighbor) % 10
                # print(h, k, j, 'hkj')
                # print(l, m, n, 'lmn')
                # print(self.wayPointGrid[h][k][j-1])
                distance = calculatePointDistance(self.wayPointGrid[l][m][n-1], self.wayPointGrid[h][k][j-1])
                GSWindAngel = calculateGSWindAngel(self.wayPointGrid[l][m][n-1], \
                                self.wayPointGrid[h][k][j-1], self.wayPointWindGrid[m][n-1][1])
                groundSpeed = calculateGroundSpeed(GSWindAngel, self.wayPointWindGrid[m][n-1][0],\
                                TAS[l])
                # print(neighbor, TAS, self.wayPointGrid[l][m][n-1][5], \
                        # self.wayPointGrid[l][m][n-1][4], self.wayPointGrid[l][m][n-1][3])
                fuelFlow = calculateFuelFlow(TAS[l], self.wayPointGrid[l][m][n-1][5], \
                        self.wayPointGrid[l][m][n-1][4], self.wayPointGrid[l][m][n-1][3])
                fuelCost = fuelFlow * distance/groundSpeed
                FuelCostGraph[node][neighbor] = fuelCost
                print(node, neighbor, fuelFlow, fuelCost)
                print(TAS[l], self.wayPointGrid[l][m][n-1][5], \
                        self.wayPointGrid[l][m][n-1][4], self.wayPointGrid[l][m][n-1][3])
        return(FuelCostGraph)


class NFZAvoidance:
    def __init__(self, wayPointGrid, flightPath, noFlyZone):
        self.wayPointGrid = wayPointGrid
        self.flightPath = flightPath
        self.noFlyZone = noFlyZone

    def createflightPathPoints(self):
        flightPathPoints = []
        pointOrder = []
        # print(self.flightPath)
        for point in self.flightPath:
            point = int(point)
            # print(point)
            a = point // 1000
            b = (point - a*1000) // 10
            c = point % 10 -1
            print(a, b, c)
            print(self.wayPointGrid[a][b][c])
            pointOrder.append(b)
            flightPathPoints.append(self.wayPointGrid[a][b][c])
        # print(flightPathPoints)
        return flightPathPoints, pointOrder

    def checkPointInNFZ(self, flightPathPoints):
        pathPointsInNFZStatus = False
        pathPointsInNFZ = [] 
        for pathpoint in flightPathPoints:

            distance = calculatePointDistance(pathpoint, self.noFlyZone)
            if distance <= self.noFlyZone[2]:
                # print('{} in the No-Fly Zone'.format(pathpoint))
                pathPointsInNFZ.append(pathpoint)
                pathPointsInNFZStatus = True
        return pathPointsInNFZStatus, pathPointsInNFZ

    def pointsDistance(self, point1, point2):
        s = (point2[0]-point1[0])*(point2[0]-point1[0]) + \
            (point2[1]-point1[1])*(point2[1]-point1[1])
        distance =  math.sqrt(s)
        return distance
    
    def pointLineDistance(self, point, line):
        p1 = [0.0, 0.0] 
        p2 = [0.0, 0.0]
        m = point[1]
        p1[0] = point[1]
        n = point[0]
        p1[1] = point[0]
        a = line[0]
        b = line[1]
        p2[0] = (n + m/a -b)*a/(a*a + 1)
        p2[1] = a*p2[0] + b
        distance = self.pointsDistance(p1, p2)
        return distance

    
    def checkPathCrossNFZ(self, flightPathPoints):
        pathCrossNFZStatus = False
        crossPoint = [0, 0]
        NFZPoint = [[0 for k in range(4)] for j in range (2)]
        m = self.noFlyZone[1]
        l = self.noFlyZone[0]
        # print(len(flightPathPoints))
        # print(flightPathPoints)
        for n in range(len(flightPathPoints)-1):
        # the last one does not need to calculate the slope    
            y = flightPathPoints[n][0] - flightPathPoints[n+1][0]
            x = flightPathPoints[n][1] - flightPathPoints[n+1][1]
            slope = y/x
            slopeNev = -x/y
            b = flightPathPoints[n][0] - slope *flightPathPoints[n][1]
            b1 = flightPathPoints[n][0] - slopeNev *flightPathPoints[n][1]
            b2 = flightPathPoints[n+1][0] - slopeNev *flightPathPoints[n+1][1]
            line = [slope, b]
            line1 = [slopeNev, b1]
            line2 = [slopeNev, b2]
            
            dis1 = self.pointLineDistance(self.noFlyZone, line1)
            dis2 = self.pointLineDistance(self.noFlyZone, line2)
            wayPointsDis = self.pointsDistance(flightPathPoints[n], flightPathPoints[n+1])
            NFZRDegree = convertMKtoDegree(self.noFlyZone[2])
            # print(dis1, dis2, wayPointsDis)
            if (dis1+dis2) <= (wayPointsDis+0.0001):
                print('NFZ in path area')
                a = line[0]
                b = line[1]
                # the way point(y,x) is diffirent from the math point(x,y)
                crossPoint[1] = (l + m/a -b)*a/(a*a + 1)
                crossPoint[0] = a*crossPoint[1] + b
                # print(self.noFlyZone, crossPoint)
                dis = self.pointsDistance(self.noFlyZone, crossPoint)
                # print(dis, self.noFlyZone[2])
                if dis <= NFZRDegree:
                    pathCrossNFZStatus = True
                    NFZPoint[0] = flightPathPoints[n]
                    NFZPoint[1] = flightPathPoints[n+1]
                    break
        return pathCrossNFZStatus, NFZPoint, n

    def createAoidancePoint(self, NFZPoints):
        r = convertMKtoDegree(self.noFlyZone[2]+5)
        a = self.noFlyZone[1]
        b = self.noFlyZone[0]
        avoidancePoint = [[0 for i in range(4)] for j in range(4)]
        solvePoints = [0 for j in range(4)]
        slope = [[0 for k in range(4)] for j in range(2)]
        s = Symbol('s')
        for i in range(2):
            m = NFZPoints[i][1]
            n = NFZPoints[i][0]
            # print(m, n)
            # slope[i] = solve((m-a)*(m-a)*s**4 + 2*(m-a)*(b-n)*s**3 + ((b-n)**2-r**2)*s**2 - r**2, s)
            slope[i] = solve(((m-a)**2-r**2)*s**4 + 2*(m-a)*(b-n)*s**3 + \
                ((b-n)**2 +(m-a)**2 - 2*r**2)*s**2 + 2*(m-a)*(b-n)*s + (b-n)**2 - r**2, s)
        x = Symbol('x')
        y = Symbol('y')
        # print('-----------------------------------')
        # print(slope)
        # print(avoidancePoint)
        for j in range(2):

            solvePoints[2*j] = solve([slope[0][j]*x - slope[0][j]*NFZPoints[0][1] - y + NFZPoints[0][0], \
                slope[1][0]*x - slope[1][0]*NFZPoints[1][1] -y + NFZPoints[1][0]], x, y)
            solvePoints[2*j+1] = solve([slope[0][j]*x - slope[0][j]*NFZPoints[0][1] - y + NFZPoints[0][0], \
                slope[1][1]*x - slope[1][1]*NFZPoints[1][1] -y + NFZPoints[1][0]], x, y)
            avoidancePoint[2*j][0] = float(solvePoints[2*j][y])
            avoidancePoint[2*j][1] = float(solvePoints[2*j][x])
            avoidancePoint[2*j][2] = NFZPoints[i][2]
            avoidancePoint[2*j][3] = NFZPoints[i][3]
            avoidancePoint[2*j+1][0] = float(solvePoints[2*j+1][y])
            avoidancePoint[2*j+1][1] = float(solvePoints[2*j+1][x])
            avoidancePoint[2*j+1][2] = NFZPoints[i][2]
            avoidancePoint[2*j+1][3] = NFZPoints[i][3]
        return avoidancePoint 

    def selectOneAoidance(self, NFZPoints, avoidancePoints, pointOrder, windList, trueAirSpeed):
        time = float('inf')
        layer = int(self.flightPath[pointOrder]) // 1000
        for i in range (len(avoidancePoints)):
            GSWindAngel_first = calculateGSWindAngel(NFZPoints[0], avoidancePoints[i], windList[pointOrder][1])
            # print(' ------------------------------------------------------------------------')
            # print(GSWindAngel_first, NFZPoints[0], avoidancePoints[i], windList[pointOrder][1], windList[pointOrder][0])
            groundSpeed_first = calculateGroundSpeed(GSWindAngel_first, windList[pointOrder][0], trueAirSpeed[layer])
            distance_first = calculatePointDistance(NFZPoints[0], avoidancePoints[i])
            t1 = distance_first/groundSpeed_first
            GSWindAngel_last = calculateGSWindAngel(avoidancePoints[i], NFZPoints[1], windList[pointOrder][1])
            groundSpeed_last = calculateGroundSpeed(GSWindAngel_last, windList[pointOrder][0], trueAirSpeed[layer])
            distance_last = calculatePointDistance(avoidancePoints[i], NFZPoints[1])
            t2 = distance_last/groundSpeed_last
            time_new = t1 + t2
            if time_new < time:
                time = time_new
                position = avoidancePoints[i]
                select = i
        return position

    def insertAvoidanceInPath(self, pointOrder, flightPath, selectedPoint):
        # avoidPathPoint = [0, 0, 0, 0]
        # avoidPathPoint[0] = avoidancePoint[0]
        # avoidPathPoint[1] = avoidancePoint[1]
        # avoidPathPoint[2] = flightPath[pointOrder][2]
        # avoidPathPoint[3] = flightPath[pointOrder][3]
        flightPath.insert((pointOrder+1), selectedPoint)
        return flightPath
