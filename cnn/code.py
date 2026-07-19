#import librarys
from engine import *
import numpy as np
import math
import random


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

#Padding Class
class Padding:
  
  def __init__(self,padding_index):
    self.pi = padding_index

  def __call__(self,img):
    r,c=len(img),len(img[0])
    new = []
    padding = [0 for _ in range(self.pi)]
    up_down = [0 for _ in range(self.pi*2+c)]
    for _ in range(self.pi):
      new.append(up_down)
    for row in img:
      t = padding + row + padding
      new.append(t)
    for _ in range(self.pi):
      new.append(up_down)
    return new

  def __repr__(self):
    return f"Padding layer {self.pi} * {self.pi}"

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

