import numpy as np
from cnn_layers import *
from utility_functions import *
from scipy.signal import correlate2d
from time import time

class covNet:
    # this calss is an implementation of the network architecture presented in https://medium.com/@ngocson2vn/a-gentle-explanation-of-backpropagation-in-convolutional-neural-network-cnn-1a70abff508b
    # and thus ecxept for initializing the layer objects (which are oly caple of forward propagaion) we will write the methods for all network opperations i.e. test and train.
    # as this network is of a specific architecture, we will have to implement the backprop that was derived for this specific architecture, any other architecture using the layes we wrote, would have a different
    # backprop scheme as the network's connectivity effects what the layes need to pass to each other in the forward and backwards feed. luclily pytorch\tensorflow are able to generalize this concept
    # with symbolic deriviatives, and thus are able to get the propper gradients of each layer, in any architecture, without manually doing so for each different architecture. 
    # As this is just an educational example, our scheme is os only valid for this architecture (presented in the link above), but can be generalized like pytorch\tensorflow one however this is not an easy task,
    # making the layer classes we wrote to be so flexible that we can concatinate as many different layers as we want, as long as the dimensions work out. However it is possible.
    def __init__(self, m1 = 6, m2 = 12):
        # initialize layers
        self.C1 = convLayer2D( m = m1, size = (5,5), init_bounds = (-0.1,0.1))
        self.C2 = convLayer3D(m = m2, size = (6,5,5), init_bounds = (-0.1,0.1))
        self.maxPool = maxPool((2,2)) # the maxPool stage doesnt have weights, and this object is just a shell that preforms maxpooling
                                      # on an input and return some stuff, hence we'll use just this object for both maxpooling opperations 
        self.FClayer = FClayer( m = 16*12, n = 10, init_bounds = (-0.1,0.1) )
    def forwardFeed(self, img):
        C1, _ = self.C1.forwardProp(img)
        P1,_ = self.maxPool.forwardProp(C1)
        
        C2, _ = self.C2.forwardProp(P1)
        P2,_ = self.maxPool.forwardProp(C2)
        
        S = self.FClayer.forwardProp(P2)
        
        return softmax(S)
        
    def train(self, train_set, train_labels, validation_set, validation_lables, epoch = 10, lr = 0.01):
        epochs = []
        avg_losses = []
        accuracies = []
        for ep in range(epoch):
            print("epoch {}".format(ep+1))  
            # Shuffle the training data
            # permutation = np.random.permutation(len(train_set))
            # train_set = train_set[permutation]
            # train_labels = train_labels[permutation]
            for img, lable in zip(train_set, train_labels):
                # t0 = time()
                ## feed forward
                C1, dC1dS1 = self.C1.forwardProp(img)
                P1,P1_idxs = self.maxPool.forwardProp(C1)
                
                C2, dC2dS2 = self.C2.forwardProp(P1)
                P2,P2_idxs = self.maxPool.forwardProp(C2)
                
                S = self.FClayer.forwardProp(P2)
                
                res = softmax(S)
                # t2 = time()
                # print("feed forward time:",t2-t0)
                ####################### backpropagation ###############################
                
                # FC layer:
                # lable = np.argmax(res)
                dLdS = np.copy(res)
                dLdS[lable] -= 1
                dLdB = dLdS
                
                dLdW = np.outer(dLdS, P2.flatten())
                dLdW = np.hstack((dLdW,dLdB.reshape((dLdW.shape[0],1))))
                # t3 = time()
                # print("FC time", t3-t2)
                
                
                
                
                # 2nd Conc Layer:
                W = self.FClayer.weights[:,:-1] # discard the last col of the weights matrix of the last layer because they correspond to the biases 
                dLdP2 = W.T @ dLdS
                dLdP2 = dLdP2.reshape(P2.shape)
                
                dLdC2 = np.zeros(C2.shape)
                for m in range(P2_idxs.shape[0]):
                    for u in range(P2_idxs.shape[1]):
                        for v in range(P2_idxs.shape[2]):
                            u_max, v_max = P2_idxs[m,u,v]
                            dLdC2[m, u_max, v_max] = dLdP2[m,u,v]
                
                dLdS2 = dLdC2 * dC2dS2
                dLdB2 = np.zeros(self.C2.b.shape)
                for m in range(dLdB2.shape[0]):
                    dLdB2[m] += np.sum(dLdS2)
                dLdK2 = np.zeros((self.C2.m,self.C2.kernel_size[0],self.C2.kernel_size[1],self.C2.kernel_size[2]))
                for m in range(self.C2.m):
                    for n in range(self.C2.kernel_size[0]):
                        dLdK2[m,n] = correlate2d(dLdS2[m] ,P1[n] ,'valid')
                
                # t4 = time()
                # print("C2 time", t4-t3)
                
                
                # 1st Conv Layer
                dS2dP1 = np.zeros((P1.shape[0],P1.shape[1],P1.shape[2],C2.shape[0],C2.shape[1],C2.shape[2]))
                for m in range(C2.shape[0]):
                    for u in range(C2.shape[1]):
                        for v in range(C2.shape[2]):
                            dS2dP1[0:P1.shape[0], u:u+self.C1.kernel_size[0], v:v+self.C1.kernel_size[1],m ,u, v] = self.C2.kernels[m]
                ###################################### this was the slowdown
                t4 = time()
                dLdP1 = np.zeros(P1.shape)
                for n in range(P1.shape[0]):
                    for r in range(P1.shape[1]):
                        for s in range(P1.shape[2]):
                            dLdP1[n,r,s] = np.sum(dS2dP1[n,r,s] * dLdS2)

                # t7 = time()
                # print("dLdP1 time", t7-t4)
                ######################################

                
                dLdC1 = np.zeros(C1.shape)
                for m in range(P1_idxs.shape[0]):
                    for u in range(P1_idxs.shape[1]):
                        for v in range(P1_idxs.shape[2]):
                            u_max, v_max = P1_idxs[m,u,v]
                            dLdC1[m, u_max, v_max] = dLdP1[m,u,v] 

                
                
                dLdS1 = dLdC1 * dC1dS1
                
                
                dLdB1 = np.zeros(self.C1.b.shape[0])
                dLdK1 = np.zeros( (self.C1.m, self.C1.kernel_size[0],self.C1.kernel_size[1]) )
                for n in range(self.C1.b.shape[0]):
                   dLdB1[n] = np.sum(dLdS1[n]) 
                   dLdK1[n] = correlate2d( img, dLdS1[n], 'valid')

                
                # t5 = time()
                # print("C1 time", t5-t4)
                
                # update parameters
                self.C1.kernels -= lr * dLdK1
                self.C1.b -= lr * dLdB1
                self.C2.kernels -= lr * dLdK2
                self.C2.b -= lr * dLdB2
                self.FClayer.weights -= lr * dLdW
                t6 = time() ##########################################################################################################################
                # print("update time:",t6-t5)
                
            losses = []
            acc = 0
            for img, label in zip(validation_set, validation_lables):
                O = self.forwardFeed(img)
                losses.append(-np.log(O[label]))
                acc += 1 if np.argmax(O) == label else 0
            losses = np.array(losses)
              
            epochs.append(ep + 1)
            avg_losses.append(losses.mean())
            accuracy = 100 * acc / len(validation_lables)
            accuracies.append(accuracy)
            print("Epoch: {}, validate_average_loss: {}, validate_accuracy: {:02.2f}%".format(ep + 1, losses.mean(), accuracy))
            
        return (epochs, avg_losses, accuracies)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        