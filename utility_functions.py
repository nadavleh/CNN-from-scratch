import numpy as np

def relu(x):
    # input -  x: np.array of float64 values
    # output - c: np.array of float64 values after element wise Relu on x    
    b = np.nonzero(x>0) # b holds the indecies of num in x that are bigger than 0
    c = np.zeros(x.shape) # c will be the return array
    c[b] = x[b] 
    return c
    
    
 def softmax(x):
     # input -  x: np.array of float64 values
     # output - np.array of float64 values after softmaxing x i.e. x_i = e^x_i/sum_over_j(e^x_j)
     return np.exp(x) / np.sum(np.exp(x))


def loss(y_pred):
    return -np.log(y_pred)