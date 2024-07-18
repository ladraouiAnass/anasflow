import pandas as pd
import numpy as np
import pickle
import random

class NN:
    def __init__(self,lr=0.01,activation="sigmouid",activatioOut="softmax",errorFunc="mse",sizes_layers=[784,20,20,10]):
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
        self.lr = lr
        self.activatioOutput = activatioOut
        self.activation = activation
        self.errorFunction = errorFunc
    
    
    def mse(self, output_layer, label):
        return -np.sum(label * np.log(output_layer)) / output_layer.shape[0]
    
    def crossetropy(self,output_layer,label):
        epsilon = 1e-15  
        output_layer = np.clip(output_layer, epsilon, 1 - epsilon)
        loss = -np.sum(label * np.log(output_layer + 1e-9)) / output_layer.shape[0]
        return loss
    
    def derivative_mse(self,output_layer,label):
           return -(2/output_layer.shape[0])*(label-output_layer)
        

    def derivative_crossentropy(self,output_layer,label):
        return -(1/output_layer.shape[0])*(output_layer/(label+0.00000000000001) - (1-output_layer)/((1-label)+0.00000000000001))
        
    
    def activationChoice(self,layer,func):
        if hasattr(self,func) and callable(getattr(self,func)):
            method = getattr(self,func)
            return method(layer)
        else:
            print("the activation function does not found")
            exit(code=-1)
    
    def errorChoice(self,func,output_layer,label):
        if hasattr(self,func) and callable(getattr(self,func)):
            method = getattr(self,func)
            return method(output_layer,label)
        else:
            print("the error function does not found")
            exit(code=-1)
    
    def derivateErrorchoice(self,func,output,label):
        func = "derivative_"+func
        if hasattr(self,func) and callable(getattr(self,func)):
            method = getattr(self,func)
            return method(output,label)
        else:
            print("derivate does not exist")
            exit(code=-1)
    

  
    def sigmouid(self,x):
        return 1 / (1 + np.exp(-x)+0.0000000000001)
        
    def softmax(self, x):
        exp_values = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp_values / np.sum(exp_values, axis=0, keepdims=True)
    
    def tanh(self, x):
        return np.tanh(x)

    def relu(self, x):
        return np.maximum(0, x)
    
    
    def correct_or_not(self,output_layer,lable):
        return int(np.argmax(output_layer)==np.argmax(lable))
    
    def forepropagation(self,data):
        self.layers[0] = data
        for i in range(len(self.layers) - 1):
            temp_layer =     self.weights["w"+str(i)] @ self.layers[i]  + self.biases[i]
            if i==len(self.layers) - 2:
                temp_layer = self.activationChoice(temp_layer,self.activatioOutput)
            else:
              temp_layer = self.activationChoice(temp_layer,self.activation)
            self.layers[i+1] = temp_layer
        return self.layers[-1]


    def backpropagation(self,errorsForWeights,errorsForBiases,batch):
        self.weights["w"+str(len(self.layers)-2)] += -self.lr * errorsForWeights["w"+str(len(self.layers)-2)]/batch
        self.biases[-1] += -self.lr * errorsForBiases[-1]/batch
        for i in range(len(self.layers)-3,0,-1):
           self.weights["w"+str(i)] += -self.lr * errorsForWeights["w"+str(i)]/batch
           self.biases[i] += -self.lr * errorsForBiases[i]/batch
           
    def train(self,epoches,images,labels,minibatch):
        acc=0
        for j in range(epoches):
            print(f"epoche {j} ...")
            accuracies = 0
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
                        img = np.array([int(value) for value in img])
                        if self.activatioOutput=="softmax" or self.activation=="softmax":
                            img=img/255.0
                        img.shape+=(1,)
                        lab.shape+=(1,)
                        print(f"{i}==>{img}")
                        print(f"{i}==>{lab}")
                        output=self.forepropagation(img)
                        dE = self.derivateErrorchoice(self.errorFunction,output,lab)
                        errorsForWeights["w"+str(len(self.layers)-2)] += dE @ np.transpose(self.layers[-2])
                        errorsForBiases[-1] += dE
                        for i in range(len(self.layers)-3,0,-1):
                            h=self.layers[i+1]
                            dE = self.weights["w"+str(i+1)].T @ dE * (h * (1-h))
                            errorsForWeights["w"+str(i)] += dE @ self.layers[i].T
                            errorsForBiases[i] += dE
                        accuracies+=self.correct_or_not(output,lab)
                    self.backpropagation(errorsForWeights,errorsForBiases,batch=minibatch)
            accuracies = 100 * accuracies /len(labels)
            print(f"correctivitie: {accuracies}%")
            acc=accuracies
        return acc
    

    def test(self,image):
        image = np.array([int(value) for value in image])
        if self.activatioOutput=="softmax" or self.activation=="softmax":
            image=image/255.0
        image.shape+=(1,)
        with open('parameters.pkl', 'rb') as f:
            parameters = pickle.load(f)
        self.biases = parameters["biases"]
        self.weights = parameters["weights"]
        self.forepropagation(image)
        output = self.layers[-1]
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

def one_hot_encoding_words(sentence):
    lista = []
    for i,w in enumerate(list(sentence.split())):
        l = [1 if j==i else 0 for j in range(len(sentence.split()))]
        lista.append(l)
    return np.array(lista)

if  __name__=="__main__":
 #   data = pd.read_csv("mnist_train.csv")
  #  images,labels = getData(data) 
    nn = NN(lr=0.08329064495947662,sizes_layers=[5,9, 2, 2, 3, 9,5],activation="sigmouid",activatioOut="sigmouid",errorFunc="crossentropy")
  #  subset=20000
  #  subSetStart = random.randint(0,(60000/2)-subset)
  #  trainAccuracy=nn.train(epoches=20,images=images[subSetStart:subSetStart+subset],labels=labels[subSetStart:subSetStart+subset],minibatch=1)
  #  print(f"train_acc = {trainAccuracy}%")
  #  with open('parameters.pkl', 'wb') as f:
   #    pickle.dump({'biases': nn.biases, 'weights': nn.weights}, f)
   # subSetStart = random.randint(60000/2,60000-subset)
   # testAccuracy=0
   # for img,lab in zip(images[subSetStart:subSetStart+subset],labels[subSetStart:subSetStart+subset]):
    #    predict = nn.test(img    print(data1)
    print(eticket)
)
     #   if predict==np.argmax(lab):
      #       testAccuracy+=1
   # testAccuracy=100*testAccuracy/subset
   # print(f"test_acc = {testAccuracy}%")
    #print(trainAccuracy-testAccuracy)

    data = one_hot_encoding_words("i like natural language processing")
    data1 = np.array([data[0]])
    eticket = np.array([data[1]])
    trainAccuracy=nn.train(epoches=10,images=data1,labels=eticket,minibatch=1)
    print(f"train_acc = {trainAccuracy}%")
   
