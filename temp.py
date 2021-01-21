from cnn_layers import *
import numpy as np

# b = convLayer3D(6, (6,5,5), (-0.1,0.1))

# a = np.ones((6,12,12))



# b.forwardProp(a)



# import skimage.data
# from matplotlib import pyplot as plt
# # Reading the image
# img = skimage.data.chelsea()
# # Converting the image into gray.
# img = skimage.color.rgb2gray(img)
# plt.figure(0)
# plt.imshow(img, cmap = 'gray')


a = np.array([[i for i in range(k,k+10)] for k in range(0,100,10)])
b = np.array([[i for i in range(k,k+10)] for k in range(100,200,10)])


ab = np.zeros((2,10,10))
ab[0,:,:] = a
ab[1,:,:] = b


img2 = np.array([i for i in range(64)])
img2 = img2.reshape((8,8))
img1 = np.array([i for i in range(64,64+64)])
img1 = img1.reshape((8,8))
img = np.array([img2,img1])

# res = ab.flatten()
# print("the 3D matrix", ab)
# print("flatten() result", res)
m = maxPool()
a,b = m.forwardProp(ab)


























