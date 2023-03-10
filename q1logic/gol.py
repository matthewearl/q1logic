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

from . import logic
from .logic import nand, constant, half_adder, full_adder, ripple_carry_add


def convolve1(inputs):
    """Use a full adder to add the bits in the Y axis"""
    outputs = []
    n = len(inputs)
    for i in range(n):
        output_row = []
        for in1, in2, in3 in zip(inputs[(i - 1) % n], inputs[i], inputs[(i + 1) % n]):
            output_row.append(full_adder(in1, in2, in3))
        outputs.append(output_row)

    return outputs


def convolve2(inputs):
    """Use a ripple-carry adder to add the counts in the X axis"""
    outputs = []

    for input_row in inputs:
        n = len(input_row)
        output_row = []
        for i in range(n):
            num1 = input_row[(i - 1) % n]
            num2 = input_row[i % n]
            num3 = input_row[(i + 1) % n]
            output_row.append(ripple_carry_add(ripple_carry_add(num1, num2), num3))
        outputs.append(output_row)
    return outputs


def gol_step(inputs):
    """Apply a game-of-life step to an input grid"""
    counts = convolve2(convolve1(inputs))
    outputs = []
    for count_row, input_row in zip(counts, inputs):
        output_row = []
        for count, live in zip(count_row, input_row):
            count = count[:3]   # Do not need the most significant bit
            not_three = nand(*count, inverted_inputs=(2,))
            not_four = nand(*count, inverted_inputs=(0, 1))

            next_live = nand(not_three,
                             nand(not_four, live, inverted_inputs=(0,)))
            output_row.append(next_live)
        outputs.append(output_row)
    return outputs


if __name__ == "__main__":
    grid_size = 4
    inputs = [
        [
            constant(label=f"in_{i},{j}")
            for j in range(grid_size)
        ]
        for i in range(grid_size)
    ]
    input_values = [
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
    ]
    for input_row, value_row in zip(inputs, input_values):
        for input_, value in zip(input_row, value_row):
            input_.output_state = bool(value)
    in_gates = [gate for input_row in inputs for gate in input_row]

    outputs = gol_step(inputs)
    out_gates = [gate for output_row in outputs for gate in output_row]
    circuit = logic.get_circuit(in_gates, out_gates)
    logic.converge(circuit)

    output_values = [
        [int(gate.get_output_state()) for gate in output_row]
        for output_row in outputs
    ]
    print('input')
    print('\n'.join(''.join('.@'[v] for v in row) for row in input_values))
    print('output')
    print('\n'.join(''.join('.@'[v] for v in row) for row in output_values))
