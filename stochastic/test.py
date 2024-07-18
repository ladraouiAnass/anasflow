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
    
    def test(self,image):
        image = np.array([int(value) for value in image])/255.0
        image.shape+=(1,)
        with open('stochastic\\parameters.pkl', 'rb') as f:
            parameters = pickle.load(f)
        self.biases = parameters["biases"]
        self.weights = parameters["weights"]
        

        self.forepropagation(image)
        output = self.layers[-1]
        print(f"the prediction is : {np.argmax(output)}")
        return np.argmax(output)


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
    data = pd.read_csv("mnist_test.csv", header=None)
    images,labels = getData(data) 
    nn = NN([784,200,100,50,10])
    accuracy = 0
    for i in range(len(images)):
        predict = nn.test(images[i])
        print(f"la valeur precise de l'image est :{np.argmax(labels[i])}")
        print("================================================") 
        if predict==np.argmax(labels[i]):
             accuracy+=1
    accuracy = accuracy*100/images.shape[0]
    print(f"accuracy = {accuracy}%")    