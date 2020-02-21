
# coding: utf-8

# In[22]:


import random


# In[23]:


device1 = []
device2 = []
collision_num = 0
for i in range(30*1600):
    device1.append(random.randrange(1, 79))
    device2.append(random.randrange(1, 79))
    print()
    if device1[i] == device2[i]:
        collision_num = collision_num + 1


# In[24]:


collision_num/(30*1600)

