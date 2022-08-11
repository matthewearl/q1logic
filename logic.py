from __future__ import annotations

import collections
import dataclasses
import itertools
from typing import Sequence


@dataclasses.dataclass
class GateInput:
    gate: Gate
    state: bool = False


@dataclasses.dataclass(init=False)
class Gate:
    label: str
    outputs: List[GateInput]    # Inputs connected to this gate's output
    inputs: List[GateInput]     # This gates inputs
    coords: Tuple[int, int]

    def __init__(self, label, coords, num_inputs=2):
        self.label = label
        self.outputs = []
        self.inputs = [GateInput(self) for _ in range(num_inputs)]
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


class NandGate(Gate):
    def get_name(self):
        return "nand"

    def get_output_state(self):
        return not all(input_.state for input_ in self.inputs)


@dataclasses.dataclass(init=False)
class ConstantGate(Gate):
    output_state: bool = dataclasses.field(default=True, init=False)

    def get_output_state(self):
        return self.output_state

    def get_name(self):
        return "in"


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
            x = gate.coords[0] * 10
            gate_str = str(gate)
            print(' ' * (x - last_x) + gate_str, end='')
            last_x = x + len(gate_str)
        last_y = y
    print()


def half_adder():
    in_gate1 = ConstantGate('in1', (0, 0), 0)
    in_gate2 = ConstantGate('in2', (2, 0), 0)
    gate1 = NandGate('n1', (1, 1))
    gate2 = NandGate('n2', (0, 2))
    gate3 = NandGate('n3', (2, 2))
    gate4 = NandGate('n4', (1, 3))
    gate5 = NandGate('n5', (3, 3))

    in_gate1.outputs.extend([gate1.inputs[0], gate2.inputs[0]])
    in_gate2.outputs.extend([gate1.inputs[1], gate3.inputs[1]])
    gate1.outputs.extend([gate2.inputs[1], gate3.inputs[0], gate5.inputs[0], gate5.inputs[1]])
    gate2.outputs.append(gate4.inputs[0])
    gate3.outputs.append(gate4.inputs[1])

    circuit = [
        in_gate1, in_gate2, gate1, gate2, gate3, gate4, gate5
    ]

    return [in_gate1, in_gate2], [gate4, gate5], circuit


if __name__ == "__main__":
    in_gates, out_gates, circuit = half_adder()

    # Print a truth table
    for in_values in itertools.product([False, True], repeat=len(in_gates)):
        for in_gate, in_value in zip(in_gates, in_values):
            in_gate.output_state = in_value
        converge(circuit)
        print_circuit(circuit)
