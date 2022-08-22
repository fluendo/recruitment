# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 12:41:45 2022

@author: Júlia Alberic i Torrent
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import cv2

#This program follows proposal 1 except for the automatic person classification,
#which was done manually by selecting each player's coordinates, and it's also
#done for a single image.

#1-We'll select colour boundaries for the jerseys and extract a binary mask
#for each to see if we chose properly and therefore could recognise a player
#based on pixel count

im=plt.imread("im1.jpg")

#colour ranges:we set the lower and upper RGB colour of each team, where lower
#is darker
bar=[(180,160,82),(209,211,128)]
rso=[(183,206,207),(227,253,245)]
ref=[(103,9,0),(214,50,57)]

#we create a an array that will have three channels, one for each team 
#(barça, real sociedad and referee)
mask=np.zeros((im.shape[0],im.shape[1],3), dtype=bool)
mask[:,:,0] = cv2.inRange(im, bar[0],  bar[1])
mask[:,:,1] = cv2.inRange(im, rso[0],  rso[1])
mask[:,:,2] = cv2.inRange(im, ref[0],  ref[1])

plt.figure(1)
plt.subplot(221)
plt.title("Original image")
plt.imshow(im)
plt.subplot(222)
plt.title("Mask for Barça")
plt.imshow(mask[:,:,0],cmap='gray')
plt.subplot(223)
plt.title("Mask for Real Sociedad")
plt.imshow(mask[:,:,1],cmap='gray')
plt.subplot(224)
plt.title("Mask for the referee")
plt.imshow(mask[:,:,2],cmap='gray')

#2-Now that we saw we chose an adequate range for the colours we proceed with
#identification of each player given that we know where there's people.
#We manually read the extreme upper left and bottom right coordinates that
#limit each player's position in our image. This would be done by a Deep
#Learning algorithm that finds people.
num_people=16
people=np.zeros((num_people,4), dtype=int)
people[0,:]=[1242,270,1302,398]
people[1,:]=[1339,389,1398,494]
people[2,:]=[1072,446,1140,582]
people[3,:]=[0,471,45,587]
people[4,:]=[419,454,505,590]
people[5,:]=[610,264,692,375]
people[6,:]=[757,129,816,223]
people[7,:]=[811,34,871,129]
people[8,:]=[846,226,900,351]
people[9,:]=[879,322,922,426]
people[10,:]=[932,459,973,539]
people[11,:]=[918,512,986,628]
people[12,:]=[1229,105,1287,213]
people[13,:]=[1572,359,1640,468]
people[14,:]=[1801,178,1869,259]

labels=[None]*num_people
colours=[None]*num_people

#for each person 'p' the loop will create a temporary smaller image 'crop'
#containing only the player and will assign each channel the values that
#we had in 'mask', so nonzero pixel count for each channel will give us how
#many pixels of that colour range we have for the player. Depending on which
#is bigger we assign the label to the player that will be used for the plot.
for p in range(0,num_people-1):
    crop=np.zeros((people[p,3]-people[p,1],people[p,2]-people[p,0],
                   3),dtype=bool)
    crop[:,:,:]=mask[people[p,1]:people[p,3],people[p,0]:people[p,2],:]
    
    pixel_num=[np.count_nonzero(crop[:,:,0]),
               np.count_nonzero(crop[:,:,1]),
               np.count_nonzero(crop[:,:,2])]
    if np.max(pixel_num)==pixel_num[0]:
        labels[p]='bar'
        colours[p]='y'
    if np.max(pixel_num)==pixel_num[1]:
        labels[p]='rso'
        colours[p]='b'
    if np.max(pixel_num)==pixel_num[2]:
        labels[p]='ref'
        colours[p]='r'

#we plot the image and use the coordinates info and colour the algorithm
#assigned to each team to plot corresponding rectangles on top of the players.
plt.figure(2)
plt.title("Team classification")
plt.imshow(im)
for p in range(0,num_people-1):
    plt.gca().add_patch(Rectangle((people[p,0],people[p,1]),
                              people[p,2]-people[p,0],people[p,3]-people[p,1]
                              ,linewidth=1,edgecolor=colours[p],
                              facecolor='none'))