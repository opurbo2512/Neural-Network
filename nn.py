#import libray
import numpy as np
import math
import random

#Value class
class Value:
  
  def __init__(self,data,_children=(),_op='',label=''):
    self.data = data
    self._prev = set(_children)
    self.grad = 0.0
    self._backward = lambda : None
    self._op = _op
    self.label = label

  def __repr__(self):
    return f"Value(data={self.data})"

  def __add__(self,other):
    other = other if isinstance(other,Value) else Value(other)
    out = Value(self.data + other.data,(self,other),'+')

    def _backward():
      self.grad += 1.0 * out.grad
      other.grad += 1.0 * out.grad
    out._backward = _backward
    return out

  def __mul__(self,other):
    other = other if isinstance(other,Value) else Value(other)
    out = Value(self.data * other.data,(self,other),'*')

    def _backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad
    out._backward = _backward
    return out

  def __pow__(self,n):
    assert isinstance(n,(int,float))
    out = Value(self.data ** n,(self,),f"**{n}")
    
    def _backward():
      self.grad += n * (self.data ** (n-1)) * out.grad
    out._backward = _backward
    return out

  def __neg__(self):
    return self * (-1)

  def __truediv__(self,other):
    return self * (other**-1)

  def __sub__(self,other):
    return self + (-other)

  def __rmul__(self,other):
    return self * other

  def __radd__(self,other):
    return self + other

  def tanh(self):
    x = self.data
    t = (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))
    out = Value(t,(self,),'tanh')

    def _backward():
      self.grad += out.grad * (1-t**2)
    out._backward = _backward
    return out

  def relu(self):
    t = max(0,self.data)
    out = Value(t,(self,),'relu')
    
    def _backward():
      self.grad += (self.data > 0) * out.grad
    out._backward = _backward
    return out

  def sigmoid(self):
    t = 1 / (1+math.exp(-self.data))
    out = Value(t,(self,),'sigmoid')

    def _backward():
      self.grad += out.grad * (t * (1-t))
    out._backward = _backward
    return out

  def exp(self):
    out = Value(math.exp(self.data),(self,),'exp')

    def _backward():
      self.grad += out.grad * math.exp(self.data)
    out._backward = _backward
    return out

  def log(self):
    assert (self.data > 0)
    out = Value(math.log(self.data),(self,),'log')

    def _backward():
      self.grad += out.grad * (1/self.data)
    out._backward = _backward
    return out

  def backward(self):
    topo = []
    visited = set()

    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topo(child)
        topo.append(v)
    self.grad = 1.0
    build_topo(self)
    for node in reversed(topo):
      node._backward()
  
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

class Softmax:

  def __call__(self,x):
    exp = [i.exp() for i in x]
    total = sum(exp)
    prob = [i/total for i in exp]
    return prob
      
class MSELoss:

  def __call__(self,pred,y):
    out = [(pred_i - y_i)**2 for pred_i,y_i in zip(pred,y)]
    return sum(out) / len(out)

class BCELoss:

  def __call__(self, pred, y):
    def loss_t(pred_t, y_t):
      first_term = pred_t.log() * y_t
      second_term = (-pred_t + 1).log() * (1 - y_t)
      
      return -(first_term + second_term)
      
    out = [loss_t(pred_t, y_t) for pred_t, y_t in zip(pred, y)]
    return sum(out) / len(out)


class CrossEntropyLoss:

    def __call__(self, pred, y):
        losses = [(-y_i * pred_i.log()) for y_i, pred_i in zip(y, pred)]
        return sum(losses) / len(losses)

class Accuracy:
    def __call__(self, pred, y):
        correct = 0
        for pred_i, y_i in zip(pred, y):
            pred_class = 1 if pred_i.data > 0.5 else 0
            if pred_class == y_i:
                correct += 1
        return correct / len(y)
