import pandas as pd
import numpy as np
import pickle
import copy

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
        self.derivateweights = copy.deepcopy(self.weights)
        self.momentumweights  = {}
        for k,v in self.weights.items():
             self.momentumweights[k] = np.zeros_like(v)

        self.biases = []
        for i in range(1,len(self.layers)):
            self.biases.append(np.random.uniform(-0.5,0.5,(sizes_layers[i])))
            self.biases[i-1].shape+=(1,)
        self.derivatebiases = self.biases[:]
        self.momentumbiases = []
        for i in range(len(self.biases)):
            self.momentumbiases.append(np.zeros_like(self.biases[i]))
   
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

    def momentum(self,alpha,lamda,param,deriv,mn_1):
        m = alpha * mn_1 + (1-alpha) * deriv
        param = param - lamda*m
        return param,m

    def backpropagation(self,label,lr=0.01,alpha=0.9):
        output = self.layers[-1]
        dE = 2*(output-label)/output.shape[0]
        self.derivateweights["w"+str(len(self.layers)-2)] = dE @ np.transpose(self.layers[-2])
        w,mw = self.momentum(alpha,lr,self.weights["w"+str(len(self.layers)-2)],self.derivateweights["w"+str(len(self.layers)-2)],0)
        self.weights["w"+str(len(self.layers)-2)] = w[:]
        self.momentumweights["w"+str(len(self.layers)-2)] = mw[:]

        self.derivatebiases[-1] = dE
        b,mb = self.momentum(alpha,lr,self.biases[-1],self.derivatebiases[-1],0)
        self.biases[-1] = b[:]
        self.momentumbiases[-1] = mb[:]
        
        for i in range(len(self.layers)-3,0,-1):
            h=self.layers[i+1]
            dE = self.weights["w"+str(i+1)].T @ dE * (h * (1-h))

            self.derivateweights["w"+str(i)] = dE @ self.layers[i].T
            w,mw = self.momentum(alpha,lr,self.weights["w"+str(i)],self.derivateweights["w"+str(i)],self.momentumweights["w"+str(i)])
            self.weights["w"+str(i)] = w[:]
            self.momentumweights["w"+str(i)] = mw[:]

            self.derivatebiases[i] = dE
            b,mb = self.momentum(alpha,lr,self.biases[i],self.derivatebiases[i],self.momentumbiases[i])
            self.biases[i] = b[:]
            self.momentumbiases[i] = mb[:]
    
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
    data = pd.read_csv("..\\version basic\\mnist_train.csv")
    images,labels = getData(data) 
    nn = NN([784,200,100,50,10])
    nn.train(lr=0.1,epoches=70,images=images,labels=labels)
    with open('..\\version momentum\\parameters.pkl', 'wb') as f:
        pickle.dump({'biases': nn.biases, 'weights': nn.weights}, f)
    
    
