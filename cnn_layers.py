from scipy.signal import correlate2d
import numpy as np
from utility_functions import *



class convLayer2D:
    # an object of this class is a convolution layer which recieves a single channel image (grayscale for example)
    # and performs a convolution on it, with m, different kernels so that its output is m different grayscale results
    def __init__(self, m = 6, size = (5,5), init_bounds = (-0.1,0.1)):
        # input - m: number of kernels to be convolved with thi input image
        #         size: a tuple containing the size of the kernels e.g. (5,5)
        #         init_bounds: a tuple containing the bounds of the uniform sample interval with which we initialize the weights
        #   *there are no "stride" or "padding" vars, we always assume a stride of 1 and no padding.
        self.m = m
        self.kernel_size = size
        
        self.b = np.random.uniform(init_bounds[0], init_bounds[1], m) # b is the biases vector which contains m biases
        self.kernels = np.zeros((m,size[0],size[1])) # its more readable to hold the kernels in a dictionary, but its more efficient to hold them in a 3D np.array
        for i in range(m):
            self.kernels[i] = np.random.uniform(init_bounds[0], init_bounds[1], size)
    def forwardProp(self, img):
        # input - img: a grayscale image 
        # output - res: a 3D np array with each res[m] being the convolution result + bias of the m'th kernel with img
        out_height = img.shape[0]-self.kernel_size[0]+1
        out_len = img.shape[1]-self.kernel_size[1]+1
        out_size = (self.m, out_height, out_len)
        S = np.zeros(out_size)
        for i in range(self.m):
            S[i] = correlate2d(img, self.kernels[i], 'valid') + self.b[i] # b[i]: is a scalar and thus is broadcasted by nump to be the same size of out_size so that it can perform matrix addition
                                                                          # 'valid': means that the output of the correlation is of the size out_size and not the size of image (which is the default) which is achieved by padding the img  
        dCdS = np.zeros(S.shape)
        idxs = np.nonzero(S > 0)
        dCdS[idxs] = 1 
        C = relu(S)
        return (C, dCdS)


class convLayer3D:
    # an object of this class is a convolution layer which recieves a single channel image (grayscale for example)
    # and performs a convolution on it, with m, different kernels so that its output is m different grayscale results
    def __init__(self, m = 12, size = (6,5,5), init_bounds = (-0.1,0.1)):
        # input - m: number of kernels to be convolved with thi input image
        #         size: a tuple containing the size of the kernels e.g. (5,5)
        #         init_bounds: a tuple containing the bounds of the uniform sample interval with which we initialize the weights
        #   *there are no "stride" or "padding" vars, we always assume a stride of 1 and no padding.
        self.m = m
        self.kernel_size = size
        
        self.b = np.random.uniform(init_bounds[0], init_bounds[1], m) # b is the biases vector which contains m biases
        self.kernels = np.zeros((m,size[0],size[1],size[2])) # its more readable to hold the kernels in a dictionary, but its more efficient to hold them in a 3D np.array
        for i in range(m):
            self.kernels[i] = np.random.uniform(init_bounds[0], init_bounds[1], size)
    def forwardProp(self, P):
        # input - P: is a result from the last layer and should contain self.size[0] results i.e. we should have self.size[0] == P.shape[0]
        # output - res: a 3D np array with each res[m] being the convolution result + bias of the m'th kernel with img
        out_height = P.shape[1]-self.kernel_size[1]+1
        out_len = P.shape[2]-self.kernel_size[2]+1
        out_size = (self.m, out_height, out_len)
        S = np.zeros(out_size)
        for m in range(self.m):
            for n in range(self.kernel_size[0]): # n goes through all the max-pooled results from the last layer i.e. self.size[0]
                S[m] += correlate2d(P[n], self.kernels[m,n], 'valid') # b[m]: is a scalar and thus is broadcasted by nump to be the same size of out_size so that it can perform matrix addition
                                                                   # 'valid': means that the output of the correlation is of the size out_size and not the size of image (which is the default) which is achieved by padding the img  
            S[m] += self.b[m]
        
        dCdS = np.zeros(S.shape)
        idxs = np.nonzero(S > 0)
        dCdS[idxs] = 1 
        C = relu(S)
        return (C, dCdS)
    def backProp(self, dCdS, dS_next_dP, dLdP, P_max_idxs, P_in):
        # remember:S is the result after convolution + bias and C = relu(S) is the result after relu on S.
        # we need to convert dLdP we got from the next (pooling) layer to dLdC, which is easily done with the P_max_idxs
        # we get from the pooling layer. this matrix holds in the (m,i,j) coordinates, a tuple (u,v) which corresponds
        # to the (i,j) pixel in P[m] (which is C[m] after pooling). this (u,v) is the original coordinates of P[m,i,j] in image C[m]
        dLdC = np.zeros(self.kernel_size)
        for m in range(P_max_idxs.shape[0]):
            for i in range(P_max_idxs.shape[1]):
                for j in range(P_max_idxs.shape[2]):
                    u,v = P_max_idxs[m,i,j]
                    dLdC[m,u,v] = dLdP[m,i,j]
        dLdS = dLdC * dCdS
        
        dLdB = np.zeros(self.m)
        for m in range(self.m):
            dLdB[m] += np.sum(dLdS[m])
        
        dLdK = np.zeros((m,self.kernel_size[0],self.kernel_size[1],self.kernel_size[2]))
        for m in range(self.m):
            for n in range(self.kernel_size[0]):
                dLdK[m,n] = correlate2d(dLdS ,P_in[n] ,'valid')
        return dLdK, dLdB, dLdS
        
        
        
            
        
        
        

class FClayer: #fully connected layer
    def __init__(self, m = 16*12, n = 10, init_bounds = (-0.1,0.1)):
        self.input_size = m
        self.output_size = n
        self.weights =  np.random.uniform(init_bounds[0], init_bounds[1], (n,m+1) )
    def forwardProp(self, img):
        f = img.flatten()
        f = np.hstack((f,[1]))
        f = self.weights @ f
        # f = softmax(f)
        return f


class maxPool:
    def __init__(self, window_size = (2,2)):
        self.window_size = window_size
    
    def forwardProp(self, img): # img is a 3D array consisting of n layered images  
        # initialize the output matrices
        out_shape = (img.shape[0], int(img.shape[1]/self.window_size[0]), int(img.shape[2]/self.window_size[1]) )
        res = np.zeros( out_shape, dtype = np.float64 )
        idxs = np.zeros( out_shape, dtype = (np.int64,2) )
        # loop through the image with window_size stride and place the max and argmax values in res and idx respectively
        u=0
        v=0
        for n in range(img.shape[0]):
            for i in range(0, img.shape[1]-self.window_size[0]+1, self.window_size[0]):
                for j in range(0, img.shape[2]-self.window_size[1]+1, self.window_size[1]):
                    roi = img[n,i:i+self.window_size[0], j:j+self.window_size[1]] # roi = region of interest
                    res[n,u,v] = np.max(roi)
                    maxIdx_in_roi = np.unravel_index(np.argmax(roi), self.window_size) # turn the argmax output which is a scalar of the maximum values 
                                                                                       # index in roi when the roi is flattened, to an index (x,y) tuple
                                                                                       # corresponding to the index places in a matrix of shape self.window_size 
                    idxs[n,u,v] = [i + maxIdx_in_roi[0], j + maxIdx_in_roi[1]]        
                    v += 1
                u += 1
                v = 0
            u = 0
        return res, idxs
                
                
        
        































