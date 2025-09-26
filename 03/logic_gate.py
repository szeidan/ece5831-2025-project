#!/usr/bin/env python3
"""
logic_gate.py
Implements a LogicGate class using NumPy.
Run with `python logic_gate.py --demo` to see a quick demo.
"""

import argparse
import numpy as np

class LogicGate:
    """Basic logic gates implemented with NumPy."""

    def and_gate(self, a, b):
        return (np.asarray(a).astype(bool) & np.asarray(b).astype(bool)).astype(int)

    def nand_gate(self, a, b):
        return (~(np.asarray(a).astype(bool) & np.asarray(b).astype(bool))).astype(int)

    def or_gate(self, a, b):
        return (np.asarray(a).astype(bool) | np.asarray(b).astype(bool)).astype(int)

    def nor_gate(self, a, b):
        return (~(np.asarray(a).astype(bool) | np.asarray(b).astype(bool))).astype(int)

    def xor_gate(self, a, b):
        return (np.asarray(a).astype(bool) ^ np.asarray(b).astype(bool)).astype(int)

def main():
    parser = argparse.ArgumentParser(description="LogicGate demo using NumPy")
    parser.add_argument("--demo", action="store_true", help="Run demo")
    args = parser.parse_args()

    if args.demo:
        g = LogicGate()
        a = [1, 0, 1, 0]
        b = [1, 1, 0, 0]
        print("a:", a)
        print("b:", b)
        print("AND :", g.and_gate(a, b))
        print("NAND:", g.nand_gate(a, b))
        print("OR  :", g.or_gate(a, b))
        print("NOR :", g.nor_gate(a, b))
        print("XOR :", g.xor_gate(a, b))
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
