#import library
import numpy as np
#nnfs library for work with dataset
import nnfs
from nnfs.datasets import spiral_data

#starting the dataset
nnfs.init()

#making group of neuron
class LayerDense:
    def __init__(self , n_inputs , n_neurons):
        self.weights = np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))

    def forward(self,inputs):
        self.inputs = inputs
        self.output = np.dot(inputs,self.weights) + self.biases

    def backward(self,dvalues):
        self.dweights = np.dot(self.inputs.T,dvalues)
        self.dbiases = np.sum(dvalues,axis = 0,keepdims=True)
        self.dinputs = np.dot(dvalues,self.weights.T)

#ReLu activation function
class Activation_Relu:
    def forward(self,inputs):
        self.inputs = inputs
        self.output = np.maximum(inputs,0)

    def backward(self,dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs<=0] = 0

#Softmax activation function
class Activation_Softmax:
    def forward(self,inputs):
        exp_value = np.exp(inputs-np.max(inputs,axis=1,keepdims=True))
        probability = exp_value/np.sum(exp_value,axis=1,keepdims=True)
        self.output = probability

    def backward(self,dvalues):
        self.dinputs = np.empty_like(dvalues)

        for index,(single_output,single_dvalues) in enumerate(zip(self.output,dvalues)):
            single_output = single_output.reshape(-1,1)
            jabocian_matrix = np.diagflat(single_output) - np.dot(single_output,single_output.T)
            self.dinputs[index] = np.dot(jabocian_matrix,single_dvalues)

#making loss class
class Loss:
    def calculate(self,output,y):
        sample_loss = self.forward(output,y)
        data_loss = np.mean(sample_loss)
        return data_loss

#use categorical cross entropy as loss function
class CategoricalCrossEntropy(Loss):
    def forward(self,y_pred,y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred,1e-7,1-1e-7)

        if len(y_true.shape) == 1:
            correct_confidence = y_pred_clipped[range(samples),y_true]

        if len(y_true.shape) == 2:
            correct_confidence = np.sum(y_pred_clipped*y_true,axis=1)

        negative_log_likelihhod = -np.log(correct_confidence)
        return negative_log_likelihhod

    def backward(self,dvalues,y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs/samples

#combine softmax and loss function
class Activation_Softmax_Categorical_Crossentropy:
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = CategoricalCrossEntropy()

    def forward(self,inputs,y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output,y_true)

    def backward(self,dvalues,y_true):
        samples = len(dvalues)

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true,axis = 1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(samples),y_true] -= 1
        self.dinputs = self.dinputs/samples

#using SGD optimizer
class optimizer_SGD:
  def __init__(self,learning_rate = 1.0):
    self.learning_rate = learning_rate

  def update_params(self,layer):
    layer.weights -= self.learning_rate*layer.dweights
    layer.biases -= self.learning_rate*layer.dbiases

#making dataset
X,y = spiral_data(samples = 100, classes= 3)

#making object from layerdense and activation function
dense1 = LayerDense(2,3)
activation1 = Activation_Relu()
dense2 = LayerDense(3,3)
activation2 = Activation_Softmax()
loss_activation = Activation_Softmax_Categorical_Crossentropy()
optimizer = optimizer_SGD()

#looping 
for epoch in range(1000):
    #forward
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output,y)

    #calculating accuracy
    predictions = np.argmax(loss_activation.output,axis = 1)
    if len(y.shape) == 2:
        y = np.argmax(y,axis = 1)
    accuracy = np.mean(y == predictions)

    if not epoch % 100:
        print(f"epoch:{epoch}"+
            f"accuracy:{accuracy:.3f}"+
            f"loss:{loss:.3f}")

    #backward
    loss_activation.backward(loss_activation.output,y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
