
#import libray
from engine import *
import numpy as np
import math
import random

  
#Module(parents of neuron,layer,mlp) class
class Module:

  def zero_grad(self):
    for p in self.parameters():
      p.grad = 0.0

  def step(self):
    for p in self.parameters():
      p.data -= p.grad * self.lr

  def l2_regu(self):
    out = [p*p for p in self.parameters()]
    return sum(out)

  def l1_regu(self):
    out = [abs(p.data) for p in self.parameters()]
    return sum(out)


  def parameters(self):
    return []

#Neuron class
class Neuron(Module):

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
class Layer(Module):

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

#MLP(Multi Laper Perceptron) class
class MLP(Module):

  def __init__(self,nin,nout,lr=0.01):
    ls = [nin] + nout
    self.lr = lr
    self.layers = []
    for i in range(len(nout)):
        is_last_layer = (i == len(nout) - 1)
        self.layers.append(Layer(ls[i], ls[i+1], nonlinear=not is_last_layer))

  def __call__(self,x):
    for layer in self.layers:
      x = layer(x)
    return x

  def parameters(self):
    params = []
    for pm in self.layers:
      params.extend(pm.parameters())
    return params

  def __repr__(self):
    return f"Multi Layer Perceptron of :\n\n[{' ,\n\n'.join(str(l) for l in self.layers)}]"

