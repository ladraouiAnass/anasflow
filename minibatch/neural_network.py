import pandas as pd
import numpy as np
import pickle

class NN:
    def __init__(self,sizes_layers=[784,20,20,10]):
        self.layers = []
        for i in range(len(sizes_layers)):
            size=sizes_layers[i]
            self.layers.append(np.zeros((size), dtype=float))
            self.layers[i].shape+=(1,)
        self.weights = {}
        for i in range(len(self.layers)-1):
            self.weights["w"+str(i)] = np.random.uniform(-0.5,0.5,(sizes_layers[i+1],sizes_layers[i]))
        self.biases = []
        for i in range(1,len(self.layers)):
            self.biases.append(np.random.uniform(-0.5,0.5,(sizes_layers[i])))
            self.biases[i-1].shape+=(1,)
    
    def forepropagation(self,data):
        self.layers[0] = data
        for i in range(len(self.layers) - 1):
            temp_layer =     self.weights["w"+str(i)] @ self.layers[i]  + self.biases[i]
            if i==len(self.layers) - 2: #softmax
                temp_layer = np.exp(temp_layer - np.max(temp_layer))/np.exp(temp_layer - np.max(temp_layer)).sum(axis=0, keepdims=True)
            else:  #sigmoid
              temp_layer = 1 / (1 + np.exp(-temp_layer))
            self.layers[i+1] = temp_layer
        return self.layers[-1]
    
    def error(self, output_layer, label):
        return -np.sum(label * np.log(output_layer)) / output_layer.shape[0]

    def softmax(self, x):
        exp_values = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp_values / np.sum(exp_values, axis=0, keepdims=True)

    def correct_or_not(self,output_layer,lable):
        return int(np.argmax(output_layer)==np.argmax(lable))
    
    def backpropagation(self,errorsForWeights,errorsForBiases,batch,lr=0.01):
        self.weights["w"+str(len(self.layers)-2)] += - lr * errorsForWeights["w"+str(len(self.layers)-2)]/batch
        self.biases[-1] += -lr * errorsForBiases[-1]/batch
        for i in range(len(self.layers)-3,0,-1):
           self.weights["w"+str(i)] += -lr * errorsForWeights["w"+str(i)]/batch
           self.biases[i] += -lr * errorsForBiases[i]/batch
    
    def train(self,lr,epoches,images,labels,minibatch):
        acc=0
        for j in range(epoches):
            print(f"epoche {j} ...")
            accuracies = 0
            lr-=lr/epoches
            for k in range(0,len(images),minibatch):
                    imagesMinibatch = images[k:k+minibatch]
                    labelsMiniBatch = labels[k:k+minibatch]
                    
                    errorsForWeights = {}               
                    for key,val in self.weights.items():
                        errorsForWeights[key] = np.zeros_like(val)
                    errorsForBiases = []                
                    for i in range(len(self.biases)):
                        errorsForBiases.append(np.zeros_like(self.biases[i]))
                        
                    for img,lab in zip(imagesMinibatch,labelsMiniBatch):
                        img = np.array([int(value) for value in img])/255.0
                        img.shape+=(1,)
                        lab.shape+=(1,)
                        output=self.forepropagation(img)
                        dE = 2*(output-lab)/output.shape[0]
                        errorsForWeights["w"+str(len(self.layers)-2)] += dE @ np.transpose(self.layers[-2])
                        errorsForBiases[-1] += dE
                        for i in range(len(self.layers)-3,0,-1):
                            h=self.layers[i+1]
                            dE = self.weights["w"+str(i+1)].T @ dE * (h * (1-h))
                            errorsForWeights["w"+str(i)] += dE @ self.layers[i].T
                            errorsForBiases[i] += dE
                        accuracies+=self.correct_or_not(output,lab)
                    self.backpropagation(errorsForWeights,errorsForBiases,batch=minibatch,lr=lr)
            accuracies = 100 * accuracies /len(labels)
            print(f"correctivitie: {accuracies}%")
            acc=accuracies
        return acc
        


def convert_to_one_hot(labels):
    labels  = [int(value) for value in labels]
    num_classes = np.max(labels) + 1
    return np.eye(num_classes)[labels]

def getData(data):
    labels = data.iloc[1:, 0].values
    images = data.iloc[1:, 1:].values
    labels = convert_to_one_hot(labels)
    return images, labels

if __name__=="__main__":
    data = pd.read_csv("mnist_train.csv")
    images,labels = getData(data) 
    nn = NN([784, 50, 31, 46, 11, 35, 10])
    nn.train(lr=0.9,epoches=70,images=images,labels=labels,minibatch=10)
    with open('parameters.pkl', 'wb') as f:
        pickle.dump({'biases': nn.biases, 'weights': nn.weights}, f)
    
    
