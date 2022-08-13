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
                #print(f"{self.label} -> {input_.gate.label} = {output_state}")
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

    def __eq__(self, other):
        return id(self) == id(other)

    def __hash__(self):
        return id(self)


class NandGate(Gate):
    def get_name(self):
        return "nand"

    def get_output_state(self):
        return not all(input_.state for input_ in self.inputs)


@dataclasses.dataclass(init=False, eq=False)
class ConstantGate(Gate):
    output_state: bool = dataclasses.field(default=True, init=False)

    def get_output_state(self):
        return self.output_state

    def get_name(self):
        return "in"


def nand(*input_gates, label='nand', coords=None):
    gate = NandGate(label, coords, len(input_gates))
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
    gate.outputs.append(input_)


def get_circuit(start_gates):
    prev_out = set()
    out = set(start_gates)

    while prev_out != out:
        next_out = set(out)
        for gate in out:
            for input_ in gate.outputs:
                if input_.gate not in out:
                    next_out.add(input_.gate)
        prev_out, out = out, next_out

    return out


def half_adder():
    c1 = constant(label='in1')
    c2 = constant(label='in2')
    n1 = nand(c1, c2)
    n4 = nand(nand(c1, n1), nand(n1, c2), label='out1')
    n5 = nand(n1, label='out2')
    return [c1, c2], [n4, n5], get_circuit([c1, c2])


def full_adder():
    c1 = constant(label='in1')
    c2 = constant(label='in2')
    c3 = constant(label='in3')

    n1 = nand(c1, c2)
    n4 = nand(nand(c1, n1), nand(n1, c2))
    n5 = nand(n4, c3)
    n8 = nand(nand(n4, n5), nand(n5, c3))
    n9 = nand(n1, n5)

    return [c1, c2, c3], [n8, n9], get_circuit([c1, c2, c3])


def circuit_to_dot(circuit):
    dot = graphviz.Digraph('circuit')
    ids = {gate: f'g{i}' for i, gate in enumerate(circuit)}
    for gate in circuit:
        dot.node(ids[gate], gate.label)
        for input_ in gate.outputs:
            dot.edge(ids[gate], ids[input_.gate], str(input_.input_num))
    print(dot.source)
    dot.render()


if __name__ == "__main__":
    in_gates, out_gates, circuit = full_adder()

    circuit_to_dot(circuit)

    # Print a truth table
    for in_values in itertools.product([False, True], repeat=len(in_gates)):
        for in_gate, in_value in zip(in_gates, in_values):
            in_gate.output_state = in_value
        converge(circuit)

        print(','.join(
            f'{gate.label}={"01"[gate.get_output_state()]}'
                for gates in [in_gates, out_gates]
                for gate in gates
        ))

