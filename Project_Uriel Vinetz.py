# -*- coding: utf-8 -*-
"""
Created on Fri May 13 17:23:21 2016

@author: Uriel
"""

#In this script we build a network in which the nodes are the set of trips
#The edges mean that the same driver can do both connected trips.
#The we will formulate the problem as a minimum cover of trips and use the
#Column generation technique where in each iteration, to solve the dual problem
#Of the linear relaxation we will solve a constrained shortest path in the
#Graph where each edge receives the dual value of the target node. 

#We first import some useful modules
import pandas as pd
#import numpy as np
import networkx as nx

#Importing the input into a dataset
data = pd.read_csv("input.csv")

#Give an integer format to the Deaparture and Arrival time and sort the trips by departure time
for index, obs in data.iterrows():
    text = obs["Departure Time"]
    pos = text.find(':')
    hours = int(float(text[0:pos]))
    minutes = int(float(text[pos+1:pos+3]))    
    x = hours*60 + minutes
    data.set_value(index,"Departure Time",x)
    text = obs["Arrival Time"]
    pos = text.find(':')
    hours = int(float(text[0:pos]))
    minutes = int(float(text[pos+1:pos+3]))    
    x = hours*60 + minutes
    data.set_value(index,"Arrival Time",x)
    
data = data.sort_values(by=["Departure Time"])

#--------------------------------Graph creation-----------------------------------------------

#Add a source and sink node and one node for each trip
G = nx.DiGraph()
G.add_nodes_from(['source','sink'])
for trip, obs in data.iterrows():
    G.add_node(trip)
    
#Add edges to the graph. Each edge adds time to the total duration and to the
#Cumulative time without break. Also, we check if there is a bus change in the edge
for trip, obs in data.iterrows():
    print('Scanning trip #: ', trip)
    dur = obs["Arrival Time"] - obs["Departure Time"]    
    #Adds an edge from the source to each trip and from each trip to the sink    
    G.add_edge('source',trip,duration=dur,weight=0,changeBus=0,cumulative=dur)
    G.add_edge(trip,'sink',duration=0,weight=0,changeBus=0,cumulative=0)
    #For each trip we will look the trips that come after it and start in the stop it ended    
    destination = obs["Destination Stop Id"]
    a_time = obs["Arrival Time"]
    temp = data[data["Origin Stop Id"] == destination]
    temp = temp[temp["Departure Time"] >= a_time]
    temp = temp.index.tolist()
    for target in temp:       
        dur = data["Arrival Time"][target] - obs["Arrival Time"]
        G.add_edge(trip,target,duration=dur,weight=0,changeBus=0,cumulative=dur)
        #If the trips on the edge are done by different vehicles or they are not subsequent trips of the same vehicle, we set changeBus to 1.
        vehicleCondition = (data["Vehicle Id"][target] == obs["Vehicle Id"])
        if vehicleCondition or ((target-trip)>1):
            G[trip][target]['changeBus']=1    
    
#Since we will have to solve the constrained SP problem several times, we will
#use the Three Stages Algorithm (Documentation attached) so we can transform the
#Problem into a simple SP in a DAG to be solved in each iteration of the CG.

#----------------------------------1) Preprocessing stage ---------------------
#The objective of this stage is to eliminate bottleneck arcs. That is, eliminate
#arcs that cannot belong to a feasible path from the source to the sink because
#of resource limitations. Moreover, as a result of this stage we can transform
#the resource usage in the arcs into resource windows in each node. 

list_of_nodes = data.index.tolist()
list_of_nodes.append('source')
list_of_nodes.append('sink')
resources = ['duration','changeBus','cumulative']
#Maximum duration and bus changes (The cumulative time without a break is done separately)
limits = [540,1]
#Initialize upper and lower bounds for the resource window at each node
t_upper = [[0,0,0] for i in list_of_nodes]
t_lower = [[0,0,0] for i in list_of_nodes]

#Every time an edge is deleted we perform another iteration
deleted = True
while deleted:
    deleted = False    
    for r in resources[:2]:
        print('Iterationg through resource:', r)
        #Maximum and minimum amount of resource left to travel from the source to each node after finding minimum and maximum resource required to travel from each node to the sink. 
        b_upper = [0 for i in list_of_nodes]
        b_upper[len(list_of_nodes)-1] = limits[resources.index(r)]
        b_lower = [limits[resources.index(r)] for i in list_of_nodes]
        #Minimum and maximum resource required to travel from the source to each node
        f_lower = [limits[resources.index(r)] for i in list_of_nodes]
        f_lower[len(list_of_nodes)-2] = 0
        f_upper = [0 for i in list_of_nodes]
        print('Starting backwards pass')
        count = 0
        #First we do a backward and the a forward scan to compute b and f for each node.        
        for current_node in list_of_nodes[len(list_of_nodes)-3::-1]:
            count = count + 1            
            print('Passing through node: ', count)
            b_up_temp = []
            b_low_temp = []
            for fs in G.out_edges(current_node, data=r):
                b_up_temp.append(b_upper[list_of_nodes.index(fs[1])] - fs[2])
                b_low_temp.append(b_lower[list_of_nodes.index(fs[1])] - fs[2])
                if (b_upper[list_of_nodes.index(fs[1])] - fs[2])<0:
                    G.remove_edge(fs[0],fs[1])
                    deleted = True
            b_upper[list_of_nodes.index(current_node)] = max(b_up_temp)
            b_lower[list_of_nodes.index(current_node)] = min(b_low_temp)
        current_node = 'source'    
        b_up_temp = []
        b_low_temp = []
        for fs in G.out_edges(current_node, data=r):
            b_up_temp.append(b_upper[list_of_nodes.index(fs[1])] - fs[2])
            b_low_temp.append(b_lower[list_of_nodes.index(fs[1])] - fs[2])
            if (b_upper[list_of_nodes.index(fs[1])] - fs[2])<0:
                G.remove_edge(fs[0],fs[1])
                deleted = True
        b_upper[list_of_nodes.index(current_node)] = max(b_up_temp)
        b_lower[list_of_nodes.index(current_node)] = min(b_low_temp)
        print('Starting forward pass')
        count = 0
        for current_node in list_of_nodes[:len(list_of_nodes)-2]:
            count = count + 1            
            print('Passing through node: ', count)
            f_low_temp = []
            f_up_temp = []
            for bs in G.in_edges(current_node, data=r):
                f_low_temp.append(f_lower[list_of_nodes.index(bs[0])] + bs[2][r])
                f_up_temp.append(f_upper[list_of_nodes.index(bs[0])] + bs[2][r])
                if (f_lower[list_of_nodes.index(bs[0])] + bs[2][r])>b_upper[current_node]:
                    G.remove_edge(bs[0],bs[1])
                    deleted = True
            f_lower[list_of_nodes.index(current_node)] = min(f_low_temp)
            f_upper[list_of_nodes.index(current_node)] = max(f_up_temp)
            if f_upper[list_of_nodes.index(current_node)] > b_lower[list_of_nodes.index(current_node)]:
                t_lower[list_of_nodes.index(current_node)][resources.index(r)] = max(f_lower[list_of_nodes.index(current_node)],b_lower[list_of_nodes.index(current_node)])
                t_upper[list_of_nodes.index(current_node)][resources.index(r)] = min(f_upper[list_of_nodes.index(current_node)],b_upper[list_of_nodes.index(current_node)])
            else:
                t_lower[list_of_nodes.index(current_node)][resources.index(r)] = f_upper[list_of_nodes.index(current_node)]
                t_upper[list_of_nodes.index(current_node)][resources.index(r)] = f_upper[list_of_nodes.index(current_node)]
        current_node = 'sink'
        f_low_temp = []
        f_up_temp = []
        for bs in G.in_edges(current_node, data=r):
            f_low_temp.append(f_lower[list_of_nodes.index(bs[0])] + bs[2][r])
            f_up_temp.append(f_upper[list_of_nodes.index(bs[0])] + bs[2][r])
            if (f_lower[list_of_nodes.index(bs[0])] + bs[2][r])>b_upper[list_of_nodes.index(current_node)]:
                G.remove_edge(bs[0],bs[1])
                deleted = True
        f_lower[list_of_nodes.index(current_node)] = min(f_low_temp)
        f_upper[list_of_nodes.index(current_node)] = max(f_up_temp)
        #Computes the resource window at each node
        if f_upper[list_of_nodes.index(current_node)] > b_lower[list_of_nodes.index(current_node)]:
            t_lower[list_of_nodes.index(current_node)][resources.index(r)] = max(f_lower[list_of_nodes.index(current_node)],b_lower[list_of_nodes.index(current_node)])
            t_upper[list_of_nodes.index(current_node)][resources.index(r)] = min(f_upper[list_of_nodes.index(current_node)],b_upper[list_of_nodes.index(current_node)])
        else:
            t_lower[list_of_nodes.index(current_node)][resources.index(r)] = f_upper[list_of_nodes.index(current_node)]
            t_upper[list_of_nodes.index(current_node)][resources.index(r)] = f_upper[list_of_nodes.index(current_node)]

current_node = 'source'
if f_upper[list_of_nodes.index(current_node)] > b_lower[list_of_nodes.index(current_node)]:
    t_lower[list_of_nodes.index(current_node)][resources.index(r)] = max(f_lower[list_of_nodes.index(current_node)],b_lower[list_of_nodes.index(current_node)])
    t_upper[list_of_nodes.index(current_node)][resources.index(r)] = min(f_upper[list_of_nodes.index(current_node)],b_upper[list_of_nodes.index(current_node)])
else:
    t_lower[list_of_nodes.index(current_node)][resources.index(r)] = f_upper[list_of_nodes.index(current_node)]
    t_upper[list_of_nodes.index(current_node)][resources.index(r)] = f_upper[list_of_nodes.index(current_node)]
    
    
#Now, let's deal with breaks. The idea is the same, but if the time between trips is more than 30 minutes
#the cumulative time avaible to reach the node is still 240 minutes in the backwards scan and the cumulative time used is 0 in the forward scan.
b_upper = [0 for i in list_of_nodes]
b_upper[len(list_of_nodes)-1] = 240
b_lower = [240 for i in list_of_nodes]
f_lower = [240 for i in list_of_nodes]
f_lower[len(list_of_nodes)-2] = 0
f_upper = [0 for i in list_of_nodes]
print('Starting backwards pass')
count = 0
for current_node in list_of_nodes[len(list_of_nodes)-3::-1]:
    count = count + 1            
    print('Passing through node: ', count)
    b_up_temp = []
    b_low_temp = []
    for fs in G.out_edges(current_node, data='cumulative'):
        if (fs[1]=='sink'):        
            b_up_temp.append(b_upper[list_of_nodes.index(fs[1])] - fs[2])
            b_low_temp.append(b_lower[list_of_nodes.index(fs[1])] - fs[2])
        else:
            if (data["Departure Time"][fs[1]] - data["Arrival Time"][fs[0]]) < 30:
                b_up_temp.append(b_upper[list_of_nodes.index(fs[1])] - fs[2])
                b_low_temp.append(b_lower[list_of_nodes.index(fs[1])] - fs[2])
            else:
                b_up_temp.append(240)
                b_low_temp.append(240)
        if (fs[1]!='sink'):
            if (b_upper[list_of_nodes.index(fs[1])] - fs[2])<0 and (data["Departure Time"][fs[1]] - data["Arrival Time"][fs[0]]) < 30:
                G.remove_edge(fs[0],fs[1])
    b_upper[list_of_nodes.index(current_node)] = max(b_up_temp)
    b_lower[list_of_nodes.index(current_node)] = min(b_low_temp)
current_node = 'source'    
b_up_temp = []
b_low_temp = []
for fs in G.out_edges(current_node, data='cumulative'):
    b_up_temp.append(b_upper[list_of_nodes.index(fs[1])] - fs[2])
    b_low_temp.append(b_lower[list_of_nodes.index(fs[1])] - fs[2])
    if (b_upper[list_of_nodes.index(fs[1])] - fs[2])<0:
        G.remove_edge(fs[0],fs[1])
b_upper[list_of_nodes.index(current_node)] = max(b_up_temp)
b_lower[list_of_nodes.index(current_node)] = min(b_low_temp)
print('Starting forward pass')
count = 0
for current_node in list_of_nodes[:len(list_of_nodes)-2]:
    count = count + 1            
    print('Passing through node: ', count)
    f_low_temp = []
    f_up_temp = []
    for bs in G.in_edges(current_node, data='cumulative'):
        if bs[0]=='source':
            f_low_temp.append(f_lower[list_of_nodes.index(bs[0])] + bs[2]['cumulative'])
            f_up_temp.append(f_upper[list_of_nodes.index(bs[0])] + bs[2]['cumulative'])
        else:
            if (data["Departure Time"][bs[1]] - data["Arrival Time"][bs[0]]) < 30:
                f_low_temp.append(f_lower[list_of_nodes.index(bs[0])] + bs[2]['cumulative'])
                f_up_temp.append(f_upper[list_of_nodes.index(bs[0])] + bs[2]['cumulative'])
            else:
                f_low_temp.append(0)
                f_low_temp.append(0)
        if (bs[0]!='source'):
            if (f_lower[list_of_nodes.index(bs[0])] + bs[2][r])>b_upper[list_of_nodes.index(current_node)]:
                G.remove_edge(bs[0],bs[1])
    f_lower[list_of_nodes.index(current_node)] = min(f_low_temp)
    f_upper[list_of_nodes.index(current_node)] = max(f_up_temp)
    if f_upper[list_of_nodes.index(current_node)] > b_lower[list_of_nodes.index(current_node)]:
        t_lower[list_of_nodes.index(current_node)][2] = max(f_lower[list_of_nodes.index(current_node)],b_lower[list_of_nodes.index(current_node)])
        t_upper[list_of_nodes.index(current_node)][2] = min(f_upper[list_of_nodes.index(current_node)],b_upper[list_of_nodes.index(current_node)])
    else:
        t_lower[list_of_nodes.index(current_node)][2] = f_upper[list_of_nodes.index(current_node)]
        t_upper[list_of_nodes.index(current_node)][2] = f_upper[list_of_nodes.index(current_node)]
current_node = 'sink'
f_low_temp = []
f_up_temp = []
for bs in G.in_edges(current_node, data=r):
    f_low_temp.append(f_lower[list_of_nodes.index(bs[0])] + bs[2][r])
    f_up_temp.append(f_upper[list_of_nodes.index(bs[0])] + bs[2][r])
    if (f_lower[list_of_nodes.index(bs[0])] + bs[2][r])>b_upper[list_of_nodes.index(current_node)]:
        G.remove_edge(bs[0],bs[1])
f_lower[list_of_nodes.index(current_node)] = min(f_low_temp)
f_upper[list_of_nodes.index(current_node)] = max(f_up_temp)
if f_upper[list_of_nodes.index(current_node)] > b_lower[list_of_nodes.index(current_node)]:
    t_lower[list_of_nodes.index(current_node)][2] = max(f_lower[list_of_nodes.index(current_node)],b_lower[list_of_nodes.index(current_node)])
    t_upper[list_of_nodes.index(current_node)][2] = min(f_upper[list_of_nodes.index(current_node)],b_upper[list_of_nodes.index(current_node)])
else:
    t_lower[list_of_nodes.index(current_node)][2] = f_upper[list_of_nodes.index(current_node)]
    t_upper[list_of_nodes.index(current_node)][2] = f_upper[list_of_nodes.index(current_node)]

current_node = 'source'
if f_upper[list_of_nodes.index(current_node)] > b_lower[list_of_nodes.index(current_node)]:
    t_lower[list_of_nodes.index(current_node)][2] = max(f_lower[list_of_nodes.index(current_node)],b_lower[list_of_nodes.index(current_node)])
    t_upper[list_of_nodes.index(current_node)][2] = min(f_upper[list_of_nodes.index(current_node)],b_upper[list_of_nodes.index(current_node)])
else:
    t_lower[list_of_nodes.index(current_node)][2] = f_upper[list_of_nodes.index(current_node)]
    t_upper[list_of_nodes.index(current_node)][2] = f_upper[list_of_nodes.index(current_node)]
  
#-------------------------------------Expand the graph-----------------------------------
#The idea of this stage is to split each node into multiple nodes that represent
#each possible usage of every resource according to its window from the previous stage.
#Then it sets an edge between two new nodes if there was an edge between their related
#original nodes and they meet the sequence of resource usage.

RL_dur = [[] for i in list_of_nodes]
RL_bus = [[] for i in list_of_nodes]
RL_cum = [[] for i in list_of_nodes]

RL_dur[list_of_nodes.index('sink')].append(limits[0])
RL_bus[list_of_nodes.index('sink')].append(limits[1])
RL_cum[list_of_nodes.index('sink')].append(240)
    
count = 0
for current_node in list_of_nodes[len(list_of_nodes)-3::-1]:
    count = count + 1
    print('Passing through node: ', count)
    RL_dur[list_of_nodes.index(current_node)].append(t_lower[list_of_nodes.index(current_node)][0])
    RL_bus[list_of_nodes.index(current_node)].append(t_lower[list_of_nodes.index(current_node)][1])
    RL_cum[list_of_nodes.index(current_node)].append(t_lower[list_of_nodes.index(current_node)][2])
    if t_upper[list_of_nodes.index(current_node)][0]!=t_lower[list_of_nodes.index(current_node)][0]:    
        RL_dur[list_of_nodes.index(current_node)].append(t_upper[list_of_nodes.index(current_node)][0])
    if t_upper[list_of_nodes.index(current_node)][1]!=t_lower[list_of_nodes.index(current_node)][1]:
        RL_bus[list_of_nodes.index(current_node)].append(t_upper[list_of_nodes.index(current_node)][1])
    if t_upper[list_of_nodes.index(current_node)][2]!=t_lower[list_of_nodes.index(current_node)][2]:
        RL_cum[list_of_nodes.index(current_node)].append(t_upper[list_of_nodes.index(current_node)][2])
    for bs in G.out_edges(current_node, data=True):
        i = bs[1]
        if (len(RL_dur[list_of_nodes.index(current_node)])>1 and len(RL_dur[list_of_nodes.index(current_node)])<(t_upper[list_of_nodes.index(current_node)][0]-t_lower[list_of_nodes.index(current_node)][0]+1)):        
            for d_i in filter(lambda x: (x>(t_lower[list_of_nodes.index(current_node)][0]+bs[2]['duration']) and x<(t_upper[list_of_nodes.index(current_node)][0]+bs[2]['duration'])),RL_dur[list_of_nodes.index(i)]):
                d_j = d_i - bs[2]['duration']
                if d_j not in RL_dur[list_of_nodes.index(current_node)]:
                    RL_dur[list_of_nodes.index(current_node)].append(d_j)   
        if (len(RL_bus[list_of_nodes.index(current_node)])>1 and len(RL_bus[list_of_nodes.index(current_node)])<(t_upper[list_of_nodes.index(current_node)][1]-t_lower[list_of_nodes.index(current_node)][1]+1)):  
            for d_i in filter(lambda x: (x>(t_lower[list_of_nodes.index(current_node)][1]+bs[2]['changeBus']) and x<(t_upper[list_of_nodes.index(current_node)][1]+bs[2]['changeBus'])),RL_bus[list_of_nodes.index(i)]):
                d_j = d_i - bs[2]['changeBus']
                if d_j not in RL_bus[list_of_nodes.index(current_node)]:
                    RL_bus[list_of_nodes.index(current_node)].append(d_j)
        if (len(RL_cum[list_of_nodes.index(current_node)])>1 and len(RL_cum[list_of_nodes.index(current_node)])<(t_upper[list_of_nodes.index(current_node)][2]-t_lower[list_of_nodes.index(current_node)][2]+1)):
            for d_i in filter(lambda x: (x>(t_lower[list_of_nodes.index(current_node)][2]+bs[2]['cumulative']) and x<(t_upper[list_of_nodes.index(current_node)][2]+bs[2]['cumulative'])),RL_cum[list_of_nodes.index(i)]):
                if (bs[1]=='sink'):            
                    d_j = d_i - bs[2]['cumulative']
                else:
                    if (data["Departure Time"][bs[1]] - data["Arrival Time"][bs[0]]) < 30:
                        d_j = d_i - bs[2]['cumulative']
                    else:
                        d_j = 240
                if d_j not in RL_cum[list_of_nodes.index(current_node)]:
                    RL_cum[list_of_nodes.index(current_node)].append(d_j)
    RL_dur[list_of_nodes.index(current_node)].sort()            
    RL_bus[list_of_nodes.index(current_node)].sort()
    RL_cum[list_of_nodes.index(current_node)].sort()

current_node = 'source'
RL_dur[list_of_nodes.index(current_node)].append(t_lower[list_of_nodes.index(current_node)][0])
RL_bus[list_of_nodes.index(current_node)].append(t_lower[list_of_nodes.index(current_node)][1])
RL_cum[list_of_nodes.index(current_node)].append(t_lower[list_of_nodes.index(current_node)][2])
if t_upper[list_of_nodes.index(current_node)][0]!=t_lower[list_of_nodes.index(current_node)][0]:    
    RL_dur[list_of_nodes.index(current_node)].append(t_upper[list_of_nodes.index(current_node)][0])
if t_upper[list_of_nodes.index(current_node)][1]!=t_lower[list_of_nodes.index(current_node)][1]:
    RL_bus[list_of_nodes.index(current_node)].append(t_upper[list_of_nodes.index(current_node)][1])
if t_upper[list_of_nodes.index(current_node)][1]!=t_lower[list_of_nodes.index(current_node)][2]:
    RL_cum[list_of_nodes.index(current_node)].append(t_upper[list_of_nodes.index(current_node)][2])
for bs in G.out_edges(current_node, data=True):
    i = bs[1]
    if (len(RL_dur[list_of_nodes.index(current_node)])>1 and len(RL_dur[list_of_nodes.index(current_node)])<(t_upper[list_of_nodes.index(current_node)][0]-t_lower[list_of_nodes.index(current_node)][0]+1)):        
        for d_i in filter(lambda x: (x>(t_lower[list_of_nodes.index(current_node)][0]+bs[2]['duration']) and x<(t_upper[list_of_nodes.index(current_node)][0]+bs[2]['duration'])),RL_dur[list_of_nodes.index(i)]):
            d_j = d_i - bs[2]['duration']
            if d_j not in RL_dur[list_of_nodes.index(current_node)]:
                RL_dur[list_of_nodes.index(current_node)].append(d_j)   
    if (len(RL_bus[list_of_nodes.index(current_node)])>1 and len(RL_bus[list_of_nodes.index(current_node)])<(t_upper[list_of_nodes.index(current_node)][1]-t_lower[list_of_nodes.index(current_node)][1]+1)):  
        for d_i in filter(lambda x: (x>(t_lower[list_of_nodes.index(current_node)][1]+bs[2]['changeBus']) and x<(t_upper[list_of_nodes.index(current_node)][1]+bs[2]['changeBus'])),RL_bus[list_of_nodes.index(i)]):
            d_j = d_i - bs[2]['changeBus']
            if d_j not in RL_bus[list_of_nodes.index(current_node)]:
                RL_bus[list_of_nodes.index(current_node)].append(d_j)
    if (len(RL_cum[list_of_nodes.index(current_node)])>1 and len(RL_cum[list_of_nodes.index(current_node)])<(t_upper[list_of_nodes.index(current_node)][2]-t_lower[list_of_nodes.index(current_node)][2]+1)):
        for d_i in filter(lambda x: (x>(t_lower[list_of_nodes.index(current_node)][2]+bs[2]['cumulative']) and x<(t_upper[list_of_nodes.index(current_node)][2]+bs[2]['cumulative'])),RL_cum[list_of_nodes.index(i)]):
            if (bs[1]=='sink'):            
                d_j = d_i - bs[2]['cumulative']
            else:
                if (data["Departure Time"][bs[1]] - data["Arrival Time"][bs[0]]) < 30:
                    d_j = d_i - bs[2]['cumulative']
                else:
                    d_j = 240
            if d_j not in RL_cum[list_of_nodes.index(current_node)]:
                RL_cum[list_of_nodes.index(current_node)].append(d_j)
RL_dur[list_of_nodes.index(current_node)].sort()            
RL_bus[list_of_nodes.index(current_node)].sort()
RL_cum[list_of_nodes.index(current_node)].sort()    
    
Ge = nx.DiGraph()
S = [[] for node in list_of_nodes]
Y = [[] for node in list_of_nodes]

num = 1
S[list_of_nodes.index('source')].append(num)
Y[list_of_nodes.index('source')].append([0,0,0])
Ge.add_node(num)
num = num + 1
count = 1

for current_node in list_of_nodes[:len(list_of_nodes)-2]:
    print("Scanning node: ", count)
    count = count + 1
    for bs in G.in_edges(current_node, data=True):
        for y in Y[list_of_nodes.index(bs[0])]:
            vec=[]
            for r in resources:                
                if r == 'duration':
                    d = RL_dur[list_of_nodes.index(current_node)]
                if r == 'changeBus':
                    d = RL_bus[list_of_nodes.index(current_node)]
                if r == 'cumulative':
                    d = RL_cum[list_of_nodes.index(current_node)]
                g = y[resources.index(r)] + bs[2][r]
                if g<=t_lower[list_of_nodes.index(current_node)][resources.index(r)]:
                    y_j = d[0]
                    vec.append(y_j)
                else:
                    if g>t_upper[list_of_nodes.index(current_node)][resources.index(r)]:
                        y_j = float('inf')
                        break
                    else:
                        y_j = filter(lambda x: x>=g, d)[0]
                        vec.append(y_j)
            if len(vec)==3:                
                if vec not in Y[list_of_nodes.index(current_node)]:
                    Y[list_of_nodes.index(current_node)].append(vec)
                    S[list_of_nodes.index(current_node)].append(num)
                    Ge.add_node(num)
                    h = S[list_of_nodes.index(bs[0])][Y[list_of_nodes.index(bs[0])].index(y)]
                    Ge.add_edge(h,num)                    
                    num = num + 1
                else:
                    h = S[list_of_nodes.index(bs[0])][Y[list_of_nodes.index(bs[0])].index(y)]
                    k = S[list_of_nodes.index(current_node)][Y[list_of_nodes.index(current_node)].index(vec)]
                    Ge.add_edge(h,k)
                    
current_node = 'sink'
for bs in G.in_edges(current_node, data=True):
    for y in Y[list_of_nodes.index(bs[0])]:
        vec=[]
        for r in resources:                
            if r == 'duration':
                d = RL_dur[list_of_nodes.index(current_node)]
            if r == 'changeBus':
                d = RL_bus[list_of_nodes.index(current_node)]
            if r == 'cumulative':
                d = RL_cum[list_of_nodes.index(current_node)]
            g = y[resources.index(r)] + bs[2][r]
            if g<=t_lower[list_of_nodes.index(current_node)][resources.index(r)]:
                y_j = d[0]
                vec.append(y_j)
            else:
                if g>t_upper[list_of_nodes.index(current_node)][resources.index(r)]:
                    y_j = float('inf')
                    break
                else:
                    y_j = filter(lambda x: x>=g, d)[0]
                    vec.append(y_j)
        if len(vec)==3:                
            if vec not in Y[list_of_nodes.index(current_node)]:
                Y[list_of_nodes.index(current_node)].append(vec)
                S[list_of_nodes.index(current_node)].append(num)
                Ge.add_node(num)
                h = S[list_of_nodes.index(bs[0])][Y[list_of_nodes.index(bs[0])].index(y)]
                Ge.add_edge(h,num)                    
                num = num + 1
            else:
                h = S[list_of_nodes.index(bs[0])][Y[list_of_nodes.index(bs[0])].index(y)]
                k = S[list_of_nodes.index(current_node)][Y[list_of_nodes.index(current_node)].index(vec)]
                Ge.add_edge(h,k)


#---------------------------------------Column Generation----------------------------

import cplex   # get Cplex API

J = len(data)   # number of journeys 

#We start with the formulation that refers to each journey being done by a 
#different driver. This means using J drivers.

main = cplex.Cplex()
main.parameters.simplex.display.set(0)
main.parameters.read.datacheck.set(1)

main.linear_constraints.add(senses = "G"*J, rhs = [1]*J)

A = []
f = [1]*J
for j in range(J):
    col = [[j],[1]]    
    A.append(col)
    
main.variables.add(obj = f, columns = A)
main.solve()
print "Optimal solution value of initial main problem is:",  main.solution.get_objective_value()

y = main.solution.get_dual_values()
count = 0
while True:
    count +=1
    print "Iteration #", count
    #At each iteration we set the edge weights in the graph to be minus the dual value of the target node 
    for i in list_of_nodes[:len(list_of_nodes)-2]:
        
        for node in S[list_of_nodes.index(i)]:
            
            for bs in Ge.in_edges(node):
                
                Ge[bs[0]][bs[1]]['weight'] = -y[i]
    #Solve the SP with the Bellman-Ford algorithm                    
    z = nx.bellman_ford(Ge,S[list_of_nodes.index('source')][0]) 

    if z[1][S[list_of_nodes.index('sink')][0]] < -1.0:

        new_col = []
        node = S[list_of_nodes.index('sink')][0]

        while node is not 1:

            ind = 947
            node = z[0][node]
            for journey in S[ind::-1]:
                if node in journey and node is not 1:
                    ind = S.index(journey)
                    new_col.append(list_of_nodes[ind].astype(int))
        
        new_col.sort()        
        main.variables.add(obj = [1], columns = [[new_col, [1]*len(new_col)]])
        
        # re-solve main problem
        main.solve()
        
        print "Optimal solution value ",  main.solution.get_objective_value()
        y = main.solution.get_dual_values()   # update dual solution

    else:  # if not - we are done
        break
        
# print final fractional solution
x = main.solution.get_values()
for i in range(main.variables.get_num()):
    if x[i]>1e-6:
        print i, x[i]

valid_lb = main.solution.get_objective_value()

# *** resolve the model as an integer programming model ***

# change the type of all the variables from contineous (defualt) to integer
main.variables.set_types(zip(range(main.variables.get_num()),"I"*main.variables.get_num()))


# set a reasonable time limit for the solution time of the integer model (in seconds)
main.parameters.timelimit.set(3600)
main.solve()

# print final integer results and set of journeys each duty is assigned to
x = main.solution.get_values()
count_duties = 0
duty_id = [0]*J
for i in range(main.variables.get_num()):
    if x[i]>1e-6:
        count_duties += 1         
        for j in range(J):
            if i in main.linear_constraints.get_rows()[j].ind: duty_id[j] = count_duties
        print ("Scanned duties:", count_duties)      


print "_______________________________________________________________________"
print "Best integer solution found", main.solution.get_objective_value(),"  Lower bound from LP relaxation ", valid_lb
print "number of duties: ", count_duties

main.linear_constraints.get_num_nonzeros() 
                           
submission = pd.DataFrame(duty_id) #Converting the matrix to a Data Frame
submission.columns=['Duty id']
submission.to_csv(path_or_buf='submission.csv',index=False) #Exporting to a CSV file