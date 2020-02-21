
# coding: utf-8

# In[33]:


import numpy as np
import scipy as sp 
import matplotlib.pyplot as plt 
from numpy import ndarray
import random


# In[153]:


def experiment( device_num = 40):
    test_second = 30    
    hopping_rate = 1600 # 1600 hops per second
    hopping_times = test_second * hopping_rate
    No_noises_channels_num = 39
   
    device= [[0] * (hopping_times*2)] * device_num
    total_channel_index_collision = [[0] * 80] * 10
    channel_index_collision = [0] * 80
    pattern = [[0] * (hopping_times*2) ] * 40
    threshold = [0] * 10
    for i in range(9):
        threshold[i+1] = (i+1) / 10
    del(threshold[0])
    num_time = -1
    collision = [0]*9
    channel = [0] * 80
    for thr in (threshold):
        num_time += 1
        print(thr)
        #init
        collision_num = 0
        for i in range(79) :
            channel_index_collision[i+1] = 0
            channel[i] = 0
        #possion fetch badchannel 
        noises_channels_num = 40
        probability = 0.5
        badchannel_num = np.random.poisson(noises_channels_num * probability)

        for _ in range(badchannel_num):
            channel[random.randrange(1, noises_channels_num)] = 2   # assume 2 represent for bad channel
        
        #fetch pattern
        for i in range(40):
            random.seed()
            seed = random.randrange(0,79)
            random.seed(seed)
            for j in range(hopping_times*2):
                pattern[i][j] = random.randrange(0,79)
        
        record_badchannel= [0] * 80
        for i in range(80):
                record_badchannel[i] = channel[i]
        for i in range(5*1600):
            for device_index in range(device_num) :
                # fetch channel
                fetch_channel = pattern[device_index][i] + 1
                device[device_index][i] = fetch_channel
                # detect collision or bad channel
                if channel[fetch_channel] != 0:
                    collision_num = collision_num + 1
                    total_channel_index_collision[num_time][fetch_channel] += 1
                    if channel[fetch_channel] !=2 : #collision
                        channel_index_collision[fetch_channel] += 1
                if channel[fetch_channel] == 0:
                    channel[fetch_channel] = 1
                elif channel[fetch_channel] == 1:
                    channel[fetch_channel] = 3
                elif channel[fetch_channel] == 2:
                    channel[fetch_channel] = 4
            for i in range(80):
                channel[i] = record_badchannel[i]
        tem = 0
        for i in range(79) :
            total_channel_index_collision[num_time][i+1] = channel_index_collision[i+1] / (5*1600)
            channel_index_collision[i+1] = channel_index_collision[i+1] / (5*1600)
            tem += channel_index_collision[i+1]
        print( tem/79 )
            
        #update badchannel
        for i in range(79) :
            if channel_index_collision[i+1] > thr:
                channel[i] = 2   # assume 2 represent for bad channel
        for i in range(79) :
            total_channel_index_collision[num_time][i+1] = channel_index_collision[i+1] * (5*1600)
            channel_index_collision[i+1] = channel_index_collision[i+1] * (5*1600)
        for i in range(80):
            record_badchannel[i] = channel[i]
            
        # 30 sec    
        for i in range(5*1600,hopping_times) :
            for device_index in range(device_num) :
                # fetch channel
                fetch_channel = random.randrange(0, 79) + 1
                device[device_index][i] = fetch_channel
                # detect collision or bad channel
                if channel[fetch_channel] != 0:
                    collision_num = collision_num + 1
                    total_channel_index_collision[num_time][fetch_channel] += 1
                    if channel[fetch_channel] !=2 : #collision
                        channel_index_collision[fetch_channel] += 1
                if channel[fetch_channel] == 0:
                    channel[fetch_channel] = 1
                elif channel[fetch_channel] == 1:
                    channel[fetch_channel] = 3
                elif channel[fetch_channel] == 2:
                    channel[fetch_channel] = 4
            for i in range(80):
                channel[i] = record_badchannel[i]
        # another 30 sec
        for i in range(hopping_times,hopping_times*2) :
            for device_index in range(device_num) :
                # fetch channel
                fetch_channel = pattern[device_index][i] + 1
                device[device_index][i] = fetch_channel
                # detect collision or bad channel
                if channel[fetch_channel] != 0 :
                    if channel[fetch_channel-1] == 0:
                        device[device_index][i] = fetch_channel-1
                        fetch_channel = (fetch_channel-1)%80
                    elif channel[(fetch_channel+1) % 80] == 0:
                        device[device_index][i] = fetch_channel+1
                        fetch_channel = (fetch_channel+1)%80
                    else:
                        if channel[fetch_channel] !=2 : #collision
                            channel_index_collision[fetch_channel] += 1
                        collision_num = collision_num + 1
                        total_channel_index_collision[num_time][fetch_channel] += 1
                if channel[fetch_channel] == 0:
                    channel[fetch_channel] = 1
                elif channel[fetch_channel] == 1:
                    channel[fetch_channel] = 3
                elif channel[fetch_channel] == 2:
                    channel[fetch_channel] = 4
            for i in range(80):
                channel[i] = record_badchannel[i]
        for i in range(79) :
                total_channel_index_collision[num_time][i+1] = channel_index_collision[i+1] / (hopping_times*2)
                channel_index_collision[i+1] = channel_index_collision[i+1] / (hopping_times*2)
        collision[num_time] = collision_num/(hopping_times*2*device_num)
        print(collision[num_time])
    #plot graph~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    get_ipython().magic('matplotlib inline')
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt

    pt = pd.DataFrame(
        {"threshold": threshold,
         "probability of collision": collision
        }

    )
    sns.factorplot(data = pt, x="threshold", y="probability of collision", ci = None, size=10 ,kind="bar",aspect=2 )


# In[154]:


experiment( device_num = 40)


# In[205]:


import numpy as np
#fetch pattern
t = [0]*40
for i in range(40):
    random.seed()
    seed = random.randrange(0,79)
    random.seed(seed)
    t[i] = seed
print (t)
counter = [0]*80
for i in range(40):
    counter[t[i]] += 1
for i in range(80):
    if counter[i] == 1:
        counter[i]=0

print(len(np.array(counter).nonzero()[0]))


# In[3]:


s

