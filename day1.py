import torch
import numpy

x=torch.arange(4.0,requires_grad=True)
print(x)

y=2*torch.dot(x,x)

print(y)

y.backward()

print(x.grad==4*x)

x.grad.zero_()

y=x.sum()

print("y2= ",y)
y.backward()
print(x.grad)