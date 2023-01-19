
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plot
import numpy as np
from scipy import misc, ndimage
import keras
from keras import backend as k
from keras.preprocessing.image import ImageDataGenerator


# In[22]:


gen = ImageDataGenerator(
    rotation_range =10,
    width_shift_range = 0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range = 0.1,
    channel_shift_range = 10,
    horizontal_flip=True)


# In[23]:


image_path = "IMG_3490.jpg"


# In[24]:


image = np.expand_dims(ndimage.imread(image_path),0)
plot.imshow(image[0])


# In[25]:


aug_iter = gen.flow(image)


# In[26]:


aug_images = [next(aug_iter)[0].astype(np.uint8)for i in range (10)]


# In[41]:


plot.imshow(aug_images[3])

