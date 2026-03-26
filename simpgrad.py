import math
import numpy as np

class Value:
    def __init__(self, data):
        self.data = data
        self.grad = 0
        self._backward = lambda : None
        self._prev = []
    
    def __repr__(self):
        return f"Value(data={self.data!r}, grad={self.grad!r})"
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data + other.data)
        
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        out._prev.append(self)
        out._prev.append(other)
        
        return out
    
    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data - other.data)

        def _backward():
            self.grad += out.grad
            other.grad -= out.grad
        out._backward = _backward
        out._prev.append(self)
        out._prev.append(other)

        return out

    def __rsub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return other - self

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data * other.data)
        
        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        out._backward = _backward
        out._prev.append(self)
        out._prev.append(other)
        
        return out
    
    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data / other.data)

        def _backward():
            self.grad += out.grad * out.data / self.data
            other.grad += out.grad * (- out.data) / other.data

        out._backward = _backward
        out._prev.append(self)
        out._prev.append(other)

        return out

    def __rtruediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return other / self

    def __matmul__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data @ other.data)

        def _backward():
            self.grad += np.array(out.grad) @ other.data.T
            other.grad += self.grad.T @ np.array(out.grad)

        out._backward = _backward
        out._prev.append(self)
        out._prev.append(other)

        return out


    def __pow__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        
        out = Value(self.data ** other.data)

        def _backward():
            self.grad = out.grad * other.data * out.data / self.data
            other.grad = out.grad * out.data * math.log(self.data)
        
        out._backward = _backward
        out._prev.append(self)
        out._prev.append(other)

        return out
    
    def __rpow__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        
        return other ** self

    def __neg__(self):
        return 0 - self

    def __pos__(self):
        return self

    def zero_grad(self):
        visited = set()
        def dfs(val):
            if val in visited:
                return
            visited.add(val)
            val.grad = 0
            for v in val._prev:
                dfs(v)
        dfs(self)
    
    def backward(self):
        self.grad = np.ones(self.data.shape) if isinstance(self.data, np.ndarray) else 1.0

        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
            for child in v._prev:
                build_topo(child)
            if v not in set(topo):
                topo.append(v)

        build_topo(self)

        for val in topo[::-1]:
            val._backward()
