#!/usr/bin/env python3
"""
module3.py
Tests the LogicGate class.
"""

import numpy as np
from logic_gate import LogicGate

g = LogicGate()

a = np.array([1, 0, 1, 0])
b = np.array([1, 1, 0, 0])

print("a:", a)
print("b:", b)
print("AND :", g.and_gate(a, b))
print("NAND:", g.nand_gate(a, b))
print("OR  :", g.or_gate(a, b))
print("NOR :", g.nor_gate(a, b))
print("XOR :", g.xor_gate(a, b))
