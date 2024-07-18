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
            if i==len(self.layers) - 2:
                temp_layer = np.exp(temp_layer - np.max(temp_layer))/np.exp(temp_layer - np.max(temp_layer)).sum(axis=0, keepdims=True)
            else:
              temp_layer = 1 / (1 + np.exp(-temp_layer))
            self.layers[i+1] = temp_layer
    
    def error(self, output_layer, label):
        return -np.sum(label * np.log(output_layer)) / output_layer.shape[0]

    def softmax(self, x):
        exp_values = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp_values / np.sum(exp_values, axis=0, keepdims=True)

    def correct_or_not(self,output_layer,lable):
        return int(np.argmax(output_layer)==np.argmax(lable))
    
    def backpropagation(self,label,lr=0.01):
        output = self.layers[-1]
        dE = 2*(output-label)/output.shape[0]
        self.weights["w"+str(len(self.layers)-2)] += - lr * dE @ np.transpose(self.layers[-2])
        self.biases[-1] += -lr * dE

        for i in range(len(self.layers)-3,0,-1):
           h=self.layers[i+1]
           dE = self.weights["w"+str(i+1)].T @ dE * (h * (1-h))
           self.weights["w"+str(i)] += -lr * dE @ self.layers[i].T
           self.biases[i] += -lr * dE
    
    def train(self,lr,epoches,images,labels):
        
        for i in range(epoches):
            lr-=lr/epoches
            accuracies = 0
            print(f"epoche {i} ...")
            for img,lab in zip(images,labels):
                img = np.array([int(value) for value in img])/255.0
                img.shape+=(1,)
                lab.shape+=(1,)
                self.forepropagation(img)
                self.backpropagation(lab,lr)
                accuracies+=self.correct_or_not(self.layers[-1],lab)
            accuracies = 100 * accuracies /len(labels)
            print(f"correctivitie: {accuracies}%")
        


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
    nn = NN()
    nn.train(lr=0.1,epoches=70,images=images,labels=labels)
    with open('parameters.pkl', 'wb') as f:
        pickle.dump({'biases': nn.biases, 'weights': nn.weights}, f)
    
    
