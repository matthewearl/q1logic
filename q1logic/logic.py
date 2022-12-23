# Copyright (c) 2022 Matthew Earl
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
#     The above copyright notice and this permission notice shall be included
#     in all copies or substantial portions of the Software.
# 
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#     OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
#     NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#     OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
#     USE OR OTHER DEALINGS IN THE SOFTWARE.

from __future__ import annotations

import collections
import dataclasses
import itertools
from typing import Sequence

import graphviz


@dataclasses.dataclass
class GateInput:
    gate: Gate
    input_num: int
    state: bool = False


@dataclasses.dataclass(init=False, eq=False)
class Gate:
    label: str
    outputs: List[GateInput]    # Inputs connected to this gate's output
    inputs: List[GateInput]     # This gates inputs
    coords: Tuple[int, int]

    def __init__(self, label, coords, num_inputs=2):
        self.label = label
        self.outputs = []
        self.inputs = [GateInput(self, input_num) for input_num in range(num_inputs)]
        self.coords = coords

    def get_output_state(self):
        raise NotImplementedError

    def get_name(self):
        raise NotImplementedError

    def propagate(self):
        changed = False
        output_state = self.get_output_state()
        for input_ in self.outputs:
            if input_.state != output_state:
                changed = True
                input_.state = output_state
        return changed

    def __str__(self):
        output_state = self.get_output_state()
        return (
            "01"[output_state]
            + "="
            + self.get_name()
            + "(" + "".join("01"[input_.state] for input_ in self.inputs) + ")"
        )

    def add_output(self, input_: GateInput):
        self.outputs.append(input_)

    def __eq__(self, other):
        return id(self) == id(other)

    def __hash__(self):
        return id(self)


class NandGate(Gate):
    def __init__(self, label, coords, num_inputs=2, *, inverted_inputs=()):
        super().__init__(label, coords, num_inputs)
        self.inverted_inputs = inverted_inputs
        self._inverted = [input_num in inverted_inputs
                          for input_num in range(num_inputs)]

    def get_name(self):
        return "nand"

    def get_output_state(self):
        return not all(input_.state != inv
                       for input_, inv in zip(self.inputs, self._inverted))


@dataclasses.dataclass(init=False, eq=False)
class ConstantGate(Gate):
    output_state: bool = dataclasses.field(default=True, init=False)

    def get_output_state(self):
        return self.output_state

    def get_name(self):
        return "in"


class DummyGate:
    """A gate that will be later replaced with a real gate.

    Use for implementing non-DAG circuits with the functional interface

    """
    outputs: List[GateInput]    # Inputs connected to this gate's output

    def __init__(self):
        self.outputs = []

    def add_output(self, input_: GateInput):
        self.outputs.append(input_)

    def replace(self, gate: Gate):
        for input_ in self.outputs:
            gate.add_output(input_)


def nand(*input_gates, label='nand', coords=None, inverted_inputs=()):
    gate = NandGate(label, coords, len(input_gates),
                    inverted_inputs=inverted_inputs)
    for from_gate, gate_input in zip(input_gates, gate.inputs):
        connect(from_gate, gate_input)
    return gate


def constant(*, label='constant', coords=None):
    return ConstantGate(label, coords, 0)


Circuit = Sequence[Gate]


def converge(circuit: Circuit):
    changed = True
    while changed:
        changed = False
        for gate in circuit:
            if gate.propagate():
                changed = True


def print_circuit(circuit: Circuit):
    min_coords = [min(gate.coords[i] for gate in circuit) for i in range(2)]
    max_coords = [max(gate.coords[i] for gate in circuit) for i in range(2)]

    y_to_gates = collections.defaultdict(list)
    for gate in circuit:
        y_to_gates[gate.coords[1]].append(gate)

    for gates in y_to_gates.values():
        gates.sort(key=lambda gate: gate.coords[0])

    last_y = 0
    for y in reversed(range(min_coords[1], max_coords[1] + 1)):
        print('\n' * (y - last_y))
        last_x = 0
        for gate in y_to_gates[y]:
            x = gate.coords[0] * 15
            gate_str = str(gate)
            print(' ' * (x - last_x) + gate_str, end='')
            last_x = x + len(gate_str)
        last_y = y
    print()


def connect(gate: Gate, input_: GateInput):
    gate.add_output(input_)


def _flood_fill(start_gates, edges):
    prev_out = set()
    out = set(start_gates)

    while prev_out != out:
        next_out = set(out)
        for gate, gate2 in edges:
            if gate in out and gate2 not in out:
                next_out.add(gate2)
        prev_out, out = out, next_out

    return out


def get_circuit(in_gates, out_gates):
    # Find gates that are influenced by input gates.

    prev_circuit = set()
    circuit = set(in_gates)
    while prev_circuit != circuit:
        next_circuit = set(circuit)
        for gate in circuit:
            for input_ in gate.outputs:
                if input_.gate not in circuit:
                    next_circuit.add(input_.gate)
        prev_circuit, circuit = circuit, next_circuit

    # Restrict to gates that influence output gates.
    reversed_edges = []
    for gate in circuit:
        for input_ in gate.outputs:
            reversed_edges.append((input_.gate, gate))
    circuit &= _flood_fill(out_gates, reversed_edges)

    return circuit


def half_adder(c1, c2):
    n1 = nand(c1, c2)
    n4 = nand(nand(c1, n1), nand(n1, c2), label='out1')
    n5 = nand(n1, label='out2')
    return [n4, n5]


def full_adder(c1, c2, c3):
    n1 = nand(c1, c2)
    n4 = nand(nand(c1, n1), nand(n1, c2))
    n5 = nand(n4, c3)
    n8 = nand(nand(n4, n5), nand(n5, c3))
    n9 = nand(n1, n5)

    return [n8, n9]


def ripple_carry_add(num1, num2, carry=None):
    assert len(num1) > 0 and len(num2) > 0, "not supported"

    out = []
    min_len = min(len(num1), len(num2))
    for bit1, bit2 in zip(num1[:min_len], num2[:min_len]):
        if carry is None:
            out_bit, carry = half_adder(bit1, bit2)
        else:
            out_bit, carry = full_adder(bit1, bit2, carry)
        out.append(out_bit)

    num = num2 if len(num1) < len(num2) else num1
    assert carry is not None
    for bit in num[min_len:]:
        out_bit, carry = half_adder(carry, bit)
        out.append(out_bit)

    out.append(carry)
    return out


def mux(select, a, b):
    """`a` when `select` is zero, otherwise `b`"""
    return nand(
        nand(select, a, inverted_inputs=(0,)),
        nand(select, b),
    )


def circuit_to_dot(circuit):
    dot = graphviz.Digraph('circuit')
    ids = {gate: f'g{i}' for i, gate in enumerate(circuit)}
    for gate in circuit:
        dot.node(ids[gate], gate.label)
        for input_ in gate.outputs:
            dot.edge(ids[gate], ids[input_.gate], str(input_.input_num))
    print(dot.source)
    dot.render()


def decode_number(num):
    return sum(gate.get_output_state() << i for i, gate in enumerate(num))
