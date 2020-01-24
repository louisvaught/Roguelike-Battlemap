# -*- coding: utf-8 -*-
"""
A simple code to produce randomized map skeletons using rigid body physics

Created on Sun Sep 15 00:21:59 2019

@author: Louis Vaught
"""

#Basic functional libraries. Okay, numpy isn't basic, but whatever.
import math
import random
import numpy as np

#Scipy will let us do simple triagulation and graph manipulation
from scipy.spatial import Delaunay
import scipy.sparse.csgraph as csg

#Matplotlib exposes simple plotting options
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle

####################
#Simple Utility Fxns
####################
def getPointsInCircle (npts,radius):
    #Generate twice the number of random points:
    pt=np.random.random((2,npts*2))
    #Pick out the points that fall inside our circle:
    goodpt=pt[:,np.sum(pt*pt,axis=0)<=1]*radius
    return goodpt[:,0:npts]

def getSizes (nrooms,avg,std):
    return np.random.normal(avg,std,size=(2,nrooms))

####################
#Main Body of Script
####################

#Room Generation Control Parameters

nrooms=30            #Number of starting rooms
startsize=250        #Radius of circle rooms are distributed in
avg=40               #Average x or y dimension of a room
std=5                #Standard deviation on dimensions
pctDead=.4           #Pct of rooms culled to make the map less dense
addBackChance=0.15    #Pct of extra connections that get returned to the map


#Physics Simulation Control Parameters

dampingFactor=0.7 #A heuristic damping factor to prevent instability
nfactor=1.5 #Collision force between particles is depth^n. Make this higher for more chaos but less stability.
moveTol=5e-4  #A stopping tolerance on room movement. This only works because we have damping

#Begin by generating the requested rooms:
roomCenters=getPointsInCircle (nrooms, startsize)
roomSizes=getSizes(nrooms,avg,std)
numRooms=roomCenters.shape[1]
areas=roomSizes[1,:]*roomSizes[0,:]

#Randomize room density to add some variability:
m=(np.random.random((1,numRooms))*0.5+0.5)*areas

#Initialize state variables for physics simulation:
a=np.zeros(roomCenters.shape)
v=np.zeros(roomCenters.shape)
x=roomCenters
minRoom=roomSizes.min()

rMaxMove=10000 #This will hold our max move for convergence checking, it starts arbitrarily large

dt0=0.2 #This is our timestep in the physics simulation
          #The physics formulation is unconditionally stable, so this will change as needed

count=0
t=0

#BEGIN PHYSICS SIMULATION LOOP

while(rMaxMove>moveTol and count<100000):
    xold=x
    #Determine bounding boxes
    bbox=np.concatenate((x-roomSizes/2,x+roomSizes/2))
    #Detect collisions
    collisions=[]
    for i in range(x.shape[1]):
        xoverlap=((bbox[0,i]<bbox[2,:]) & (bbox[0,i]>bbox[0,:])|(bbox[2,i]<bbox[2,:]) & (bbox[2,i]>bbox[0,:]))
        yoverlap=((bbox[1,i]<bbox[3,:]) & (bbox[1,i]>bbox[1,:])|(bbox[3,i]<bbox[3,:]) & (bbox[3,i]>bbox[1,:]))
        collisions.append(np.where(xoverlap & yoverlap))
    #Do (collisionarea)*(collisiondepth)^n to determine corrective force
    force=np.zeros(x.shape)
    for object1 in range(x.shape[1]):
        for object2 in collisions[object1][0]:
            minMax=np.minimum(bbox[2:4,object1],bbox[2:4,object2])
            maxMin=np.maximum(bbox[0:2,object1],bbox[0:2,object2])
            collisionArea=np.product(minMax-maxMin)
            collisionDir=((minMax+maxMin)/2<x[:,object1])*-1+((minMax+maxMin)/2>=x[:,object1])*1
            collisionDepth=(minMax-maxMin)
            currForce=-1*collisionDir*np.power((collisionArea*collisionDepth),nfactor)
            force[:,object1]=force[:,object1]+currForce
    #Calculate acceleration
    a=force/m
    #Determine stable time update based on minimum room size and maximum distance travelled:
    dt=dt0
    vproto=v+a*dt
    dxproto=vproto*dt
    while (np.abs(dxproto)).max()>0.1*minRoom:
        dt=dt/2
        vproto=v+a*dt
        dxproto=vproto*dt
    #Calculate updated velocity and position:
    vold=v
    v=vold+a*dt
    xold=x
    x=xold+v*dt
    t=t+dt
    #Damp velocities:
    v=dampingFactor*v
    
    #Check normalized move criteria:
    rMaxMove=((np.abs(x-xold)).max()/dt)/minRoom
    
    #Print timestep sometimes:
    if (count == 0):
        fig = plt.figure()
        scatterplot=plt.scatter(x[0,:],x[1,:])
        #plt.axis([-100,-100,100,100])
    elif (count%100 == 0):
        print("Completed step "+str(count)+" at time t="+str(t))
        print("Max normalized move is: "+str(rMaxMove))
        plt.scatter(x[0,:],x[1,:])
        fig.canvas.draw()
        fig.canvas.flush_events()
    count=count+1

#END PHYSICS SIMULATION LOOP, BEGIN MAP CLEANUP
    
# Plot all of the rooms as rectangular patches
rooms = []
for i in range(bbox.shape[1]):
    rect = Rectangle((bbox[0,i], bbox[1,i]),roomSizes[0,i],roomSizes[1,i])
    rooms.append(rect)
    
pc = PatchCollection(rooms, facecolor='r', alpha=0.5,edgecolor='r')

fig, ax = plt.subplots(1)
ax.add_collection(pc)
artists = ax.scatter(x[0,:],x[1,:])
plt.show()

#Randomly delete some of the rooms to make the map less dense
roomMask=np.random.rand(nrooms)
nRoomSizes=roomSizes[:,roomMask>pctDead]
nBBox=bbox[:,roomMask>pctDead]
nRoomCenters=x[:,roomMask>pctDead]

nRooms = []

# Loop over data points; create box from errors at each point
for i in range(nBBox.shape[1]):
    rect = Rectangle((nBBox[0,i], nBBox[1,i]),nRoomSizes[0,i],nRoomSizes[1,i])
    nRooms.append(rect)
    
#Make two patch collections so we can graph the output twice (for debugging)
pc2 = PatchCollection(nRooms, facecolor='r', alpha=0.5,edgecolor='r')
pc2b = PatchCollection(nRooms, facecolor='r', alpha=0.5,edgecolor='r')

#Triangulation - look how simple it is!
tri = Delaunay(nRoomCenters.T)

#Now we need to convert triangulation into a dense graph
(indptr,indices)=tri.vertex_neighbor_vertices
graphArray=np.zeros((tri.points.shape[0],tri.points.shape[0]))
for i in range(indptr.size-1):
    pointList=indices[indptr[i]:indptr[i+1]]
    for j in pointList:
        dx=(tri.points[i,0]-tri.points[j,0])
        dy=(tri.points[i,1]-tri.points[j,1])
        graphArray[i,j]=math.sqrt(dx*dx+dy*dy)
        
#Now use csg representation to find a minimum spanning tree
roomGraph=csg.csgraph_from_dense(graphArray)
minSpanTree=csg.minimum_spanning_tree(roomGraph)
nodeList=np.asarray(minSpanTree.nonzero()).T

#Use a LineCollection to visualize the spanning tree:
lineList=[]
for i in range(nodeList.shape[0]):
    pt1=nodeList[i,0]
    pt2=nodeList[i,1]
    lineList.append([tri.points[pt1,:],tri.points[pt2,:]])
lines=LineCollection(lineList)

fig, ax = plt.subplots(1)
ax.add_collection(pc2)
ax.add_collection(lines)
artists = ax.scatter(nRoomCenters[0,:],nRoomCenters[1,:])
plt.show()

#Now randomly add some connections back:
lineAddBack=np.random.rand(tri.points.shape[0],tri.points.shape[0])<addBackChance
addBack=csg.csgraph_to_dense(minSpanTree)+csg.csgraph_to_dense(roomGraph)*lineAddBack
nodeList2=np.asarray(addBack.nonzero()).T

#Use a LineCollection to visualize the spanning tree:
lineList2=[]
for i in range(nodeList2.shape[0]):
    pt1=nodeList2[i,0]
    pt2=nodeList2[i,1]
    lineList2.append([tri.points[pt1,:],tri.points[pt2,:]])
lines2=LineCollection(lineList2)

fig, ax = plt.subplots(1)
ax.add_collection(pc2b)
ax.add_collection(lines2)
artists = ax.scatter(nRoomCenters[0,:],nRoomCenters[1,:])
plt.show()

#Loop through all the lines and deconstruct them into x and y components:
deconLines=[]
for currLine in lineList2:
    xFirst=bool(random.getrandbits(1)) #Determine if you go horizontal or vertical first
    if xFirst:
        midPt=np.asarray((currLine[1][0],currLine[0][1]))
    else:
        midPt=np.asarray((currLine[0][0],currLine[1][1]))
    deconLines.append([currLine[0],midPt])
    deconLines.append([midPt,currLine[1]])
lines3=LineCollection(deconLines)
    
#Make another patchcollection and then replot with the changes
pc2c = PatchCollection(nRooms, facecolor='r', alpha=0.5,edgecolor='r')
fig, ax = plt.subplots(1)
ax.add_collection(pc2c)
ax.add_collection(lines3)
artists = ax.scatter(nRoomCenters[0,:],nRoomCenters[1,:])
plt.show()