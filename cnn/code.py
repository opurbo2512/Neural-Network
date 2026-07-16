#import librarys
import numpy as np
import math
import random

#Value Class
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

  def __lt__(self,other):
    other_data = other.data if isinstance(other,Value) else other
    return self.data > other.data

  def __gt__(self,other):
    other_data = other.data if isinstance(other,Value) else other
    return self.data < other.data

  def __ge__(self,other):
    other_data = other.data if isinstance(other,Value) else other
    return self.data >= other.data

  def __le__(self,other):
    other_data = other.data if isinstance(other,Value) else other
    return self.data <= other.data

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

#InputLayer Class
class InputLayer:
  def __call__(self,images):
    out = []
    for image in images:
      t = [list(map(Value,im)) for im in image]
      out.append(t)
    return out

  def __repr__(self):
    return f"Input Layer "

#One layer conv Class
class ConvNeuron:

  def __init__(self,filter_number):
    self.f_n = filter_number
    self.w = np.array([[Value(random.uniform(-1,1)) for _ in range(self.f_n)] for _ in range(self.f_n)])

  def __call__(self,images):

    output_rank = len(images[0])-len(self.w)+1
    final_out = np.zeros((output_rank,output_rank))

    for x in images:
      out = []
      x = np.array(x)

      for i in range(0,len(x)-self.f_n+1):
        row = []
        for j in range(0,len(x)-self.f_n+1):
          t = np.array((x[i:self.f_n+i,j:self.f_n+j]))
          t1 = np.sum(t * self.w)
          row.append(t1)
        out.append(row)
      final_out = final_out + np.array(out)

    return np.array(final_out)

  def parameters(self):
    out = []
    for row in self.w:
      out.extend(row)
    return out

  def __repr__(self):
    return f"Convolution layer({self.f_n} * {self.f_n})"

#Convolution Layer
class Conv:

  def __init__(self,filter_size,nout):
    self.conv_neurons = [ConvNeuron(filter_size) for _ in range(nout)]

  def __call__(self,x):
    out = [c(x) for c in self.conv_neurons]
    return out

  def parameters(self):
    params = []
    for pm in self.conv_neurons:
      params.extend(pm.parameters())
    return params

  def __repr__(self):
    return f"Convolution Layer of : [{', '.join(str(c) for c in self.conv_neurons)}]"

#Average Pooling Class
class AvgPool:

  def __init__(self,index):
    self.id = index

  def __call__(self,images):
    final_out = []
    images = np.array(images)

    for img in images:
      out = []
      for i in range(0,len(img)-self.id+1,self.id):
        row = []
        for j in range(0,len(img)-self.id+1,self.id):
          t = img[i:self.id+i,j:self.id+j]
          flat = t.flatten()
          pooling_value = sum(flat, Value(0.0)) / float(len(flat))
          row.append(pooling_value)
        out.append(row)
      final_out.append(out)
    return np.array(final_out)

  def __repr__(self):
    return f"Average Pooling layer of {self.id} * {self.id}"

#Max Pooling Class
class MaxPool:

  def __init__(self,index):
    self.id = index

  def __call__(self,images):
    final_out = []
    images = np.array(images)

    for img in images:
      out = []
      for i in range(0,len(img)-self.id+1,self.id):
        row = []
        for j in range(0,len(img)-self.id+1,self.id):
          t = img[i:self.id+i,j:self.id+j]
          pooling_value = np.max(t)
          row.append(pooling_value)
        out.append(row)
      final_out.append(out)
    return np.array(final_out)

  def __repr__(self):
    return f"Max Pooling layer of {self.id} * {self.id}"

#Flatten Class
class Flatten:

  def __call__(self,x):
    if isinstance(x,np.ndarray):
      return x.reshape(-1)
    out = []
    for mat in x:
      for row in mat:
        out.extend(row)
    return out

  def __repr__(self):
    return f"Flatten Layer."

#Neuron Class
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

#Layer Class
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
    return f"Layer of : [{', '.join(str(n) for n in self.neurons)}]"

#Full CNN class
class CNN:

  def __init__(self,conv_index,pooling_index,ann_input,ann_layers):
    conv_layers = [Conv(t[0],t[1]) for t in conv_index] #info is is tuple
    sz = [ann_input] + ann_layers

    layers = []
    for i in range(len(ann_layers)):
        is_last_layer = (i == len(ann_layers) - 1)
        layers.append(Layer(sz[i], sz[i+1], nonlinear=not is_last_layer))

    self.graph = [InputLayer()]
    self.pm_layers = []

    for conv_t in conv_layers:
      self.graph.append(conv_t)
      self.graph.append(AvgPool(pooling_index))
      self.pm_layers.append(conv_t)

    self.graph.append(Flatten())
    self.graph.extend(layers)

    for layer in layers:
      self.pm_layers.append(layer)

  def __call__(self,x):
    for graph_t in self.graph:
      x = graph_t(x)
    return x

  def parameters(self):
    out = []
    for pm in self.pm_layers:
      out.extend(pm.parameters())
    return out

  def __repr__(self):
    return f"Convolution Neural Network of :\n\n[{' ,\n\n'.join(str(l) for l in self.graph)}]"

#Sofmax Activation
class Softmax:

  def __call__(self,x):
    exp = [i.exp() for i in x]
    total = sum(exp)
    prob = [i/total for i in exp]
    return prob

#Loss Function
class CrossEntropyLoss:

    def __call__(self, pred, y):
        losses = [(-y_i * pred_i.log()) for y_i, pred_i in zip(y, pred)]
        return sum(losses) / len(losses)

