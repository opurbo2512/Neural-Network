#import librarys
from engine import *
import numpy as np
import math
import random

class RNNCell:

  def __init__(self,nin):
    self.w = [Value(random.uniform(-0.1,0.1)) for _ in range(nin)]
    self.wh = [Value(random.uniform(-0.1,0.1)) for _ in range(nin)]
    self.bh = Value(random.uniform(-0.1,0.1))
    self.size = nin

  def __call__(self,x,h):
    out = [(w_i*x_i + wh_i*h_i + self.bh).tanh() for w_i,x_i,wh_i,h_i in zip(self.w,x,self.wh,h)]
    return out

  def parameters(self):
    return self.w + self.wh + [self.bh]

  def __repr__(self):
    return f"RNNCell hidden state size : {self.size}"

#Neuron class
class Neuron:

  def __init__(self,nin,nonlinear=True):
    self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
    self.b = Value(random.uniform(-1,1))
    self.nonlinear = nonlinear

  def __call__(self,x):
    act = sum([wi * xi for wi,xi in zip(self.w,x)],self.b)
    return act.relu() if self.nonlinear else act

  def parameters(self):
    return self.w + [self.b]

  def __repr__(self):
    return f"{'ReLU' if self.nonlinear else 'Linear'}Neuron({len(self.w)})"

#Layer class
class Layer:

  def __init__(self,nin,nout,nonlinear=True):
    self.neurons = [Neuron(nin, nonlinear=nonlinear) for _ in range(nout)]

  def __call__(self,x):
    out = [n(x) for n in self.neurons]
    return out[0] if len(out)==1 else out

  def parameters(self):
    params = []
    for pm in self.neurons:
      params.extend(pm.parameters())
    return params

  def __repr__(self):
    return f"Layer of :\n[{', \n'.join(str(n) for n in self.neurons)}]"

#RNN class
class RNN:

  def __init__(self,nin,nout,lr=0.01):
    ls = [nin] + nout
    self.lr = lr
    self.nin = nin
    self.hidden_state = [Value(0.0) for _ in range(nin)]
    self.rnn_cell = RNNCell(nin)
    self.layers = []
    for i in range(len(nout)):
        is_last_layer = (i == len(nout) - 1)
        self.layers.append(Layer(ls[i], ls[i+1], nonlinear=not is_last_layer))
    

  def __call__(self,x):
    self.hidden_state = self.rnn_cell(x,self.hidden_state)
    out = self.hidden_state

    for layer in self.layers:
      out = layer(out)
    return out

  def parameters(self):
    params = list(self.rnn_cell.parameters())
    for pm in self.layers:
      params.extend(pm.parameters())
    return params

  def __repr__(self):
    return f"RNN of :\n\n[{' ,\n\n'.join(str(l) for l in self.layers)}]"

