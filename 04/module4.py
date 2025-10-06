import numpy as np
from three_layer_mlp import ThreeLayerMLP

mlp = ThreeLayerMLP()
x = np.array([1.0, 0.5])
print(f'Input to the MLP = {x}')
y = mlp.forward(x)
print(f'Output to MLP = {y}')