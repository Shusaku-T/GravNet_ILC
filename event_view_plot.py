import evaluation_noNoise as ev
import matplotlib.pyplot as plt
import numpy as np
import os
import math
import sys

def colorlabel(y,unique_arr):
    if y == unique_arr[0]:
        return "b"
    elif y == unique_arr[1] : return "g"
    if len(unique_arr)==3 :
        if y == unique_arr[2] : return "y"
    if len(unique_arr)==4 :
        if y == unique_arr[2] : return "y"
        elif y == unique_arr[3] : return "r"

def event_display(event,clustering):
    x_point=event.x[:,1]
    y_point=event.x[:,2]
    z_point=event.x[:,3]
    fig = plt.figure(figsize = (8, 8))
    ax1 = fig.add_subplot(1,2,1, projection='3d')
    ax2 = fig.add_subplot(1,2,2, projection='3d')
    label=clustering
    print(len(clustering))
    unique_label=np.unique(label)
    #print(f"predict unique_label : {len(unique_label)}")                                                
    l = 0
    cmap = plt.get_cmap('tab10')
    #for x1,y1,z1,label1 in zip(x_point,y_point,z_point,label):
        #if colorlabel(label1) == "b":                                                         
        #print(f"label : {colorlabel(label1)}")                                                        
    sc1=ax1.scatter(x_point, y_point, z_point,vmin=1,vmax=np.max(unique_label),c=label,cmap=cmap)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")
    ax1.text2D(0.65,0.65,f"Cluster : {len(unique_label)}",transform=ax1.transAxes)
    ax1.set_title("predicted label")
    
    label=event.y
    unique_label=np.unique(label)
    l = 0
    #for x1,y1,z1,label1 in zip(x_point,y_point,z_point,label):
        #if colorlabel(label1) == "b":                                                                   
        #print(f"label : {colorlabel(label1)}") 
    sc2 = ax2.scatter(x_point, y_point, z_point,vmin=1,vmax=np.max(unique_label),c=label,cmap=cmap)
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("z")
    ax2.text2D(0.65,0.65,f"Cluster : {len(unique_label)}",transform=ax2.transAxes)
    ax2.set_title("true label")
    plt.show()
