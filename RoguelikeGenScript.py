# -*- coding: utf-8 -*-
"""
A simple code to produce randomized map skeletons using rigid body physics

Created on Sun Sep 15 00:21:59 2019

@author: Louis Vaught
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

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

nrooms=30
startsize=250
avg=40
std=5

roomCenters=getPointsInCircle (nrooms, startsize)
roomSizes=getSizes(nrooms,avg,std)
numRooms=roomCenters.shape[1]
areas=roomSizes[1,:]*roomSizes[0,:]
#Randomize room mass:
m=(np.random.random((1,numRooms))*0.5+0.5)*areas
a=np.zeros(roomCenters.shape)
v=np.zeros(roomCenters.shape)
x=roomCenters

dampingFactor=0.7 #A heuristic damping factor to prevent instability
nfactor=1.5 #For calculation of force
moveTol=5e-4
minRoom=roomSizes.min()

rMaxMove=10000

dt0=0.2

count=0
t=0

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
    #Do (collisionarea)*(collisiondepth)^n to determine force
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
    #Determine stable time update:
    dt=dt0
    vproto=v+a*dt
    dxproto=vproto*dt
    while (np.abs(dxproto)).max()>0.1*minRoom:
        dt=dt/2
        vproto=v+a*dt
        dxproto=vproto*dt
    #Calculate velocity and position:
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

    
# Create list for all the patches
rooms = []

# Loop over data points; create box from errors at each point
for i in range(bbox.shape[1]):
    rect = Rectangle((bbox[0,i], bbox[1,i]),roomSizes[0,i],roomSizes[1,i])
    rooms.append(rect)
    
pc = PatchCollection(rooms, facecolor='r', alpha=0.5,edgecolor='r')

fig, ax = plt.subplots(1)
ax.add_collection(pc)
artists = ax.scatter(x[0,:],x[1,:])
plt.show()

#Randomly delete some of the rooms
pctDead=.4
roomMask=np.random.rand(nrooms)
nRoomSizes=roomSizes[:,roomMask>pctDead]
nBBox=bbox[:,roomMask>pctDead]
nRoomCenters=x[:,roomMask>pctDead]

nRooms = []

# Loop over data points; create box from errors at each point
for i in range(nBBox.shape[1]):
    rect = Rectangle((nBBox[0,i], nBBox[1,i]),nRoomSizes[0,i],nRoomSizes[1,i])
    nRooms.append(rect)
    
pc2 = PatchCollection(nRooms, facecolor='r', alpha=0.5,edgecolor='r')

fig, ax = plt.subplots(1)
ax.add_collection(pc2)
artists = ax.scatter(nRoomCenters[0,:],nRoomCenters[1,:])
plt.show()

#Triangulation:

from scipy.spatial import Delaunay
tri = Delaunay(nRoomCenters.T)
plt.triplot(nRoomCenters[0,:], nRoomCenters[1,:], tri.simplices.copy())
plt.plot(nRoomCenters[0,:], nRoomCenters[1,:], 'o')
plt.show()