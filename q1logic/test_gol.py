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

import itertools

from . import logic
from .gol import convolve1, convolve2, gol_step
from .logic import constant, decode_number


def test_convolve():
    inputs = [
        [
            constant(label=f"in_{i},{j}")
            for j in range(4)
        ]
        for i in range(4)
    ]
    input_values = [
        [1, 0, 0, 0],
        [1, 0, 1, 1],
        [1, 1, 1, 0],
        [0, 0, 0, 1]
    ]
    for input_row, value_row in zip(inputs, input_values):
        for input_, value in zip(input_row, value_row):
            input_.output_state = bool(value)
    in_gates = [gate for input_row in inputs for gate in input_row]

    # 1-D convolve
    outputs = convolve1(inputs)
    out_gates = [gate for output_row in outputs for num in output_row
                 for gate in num]
    circuit = logic.get_circuit(in_gates, out_gates)
    logic.converge(circuit)

    output_values = [
        [decode_number(num) for num in output_row]
        for output_row in outputs
    ]

    expected = [
	[2, 0, 1, 2],
	[3, 1, 2, 1],
	[2, 1, 2, 2],
	[2, 1, 1, 1],
    ]

    assert output_values == expected

    # 2-D convolve
    outputs = convolve2(outputs)
    out_gates = [gate for output_row in outputs for num in output_row
                 for gate in num[:3]]
    circuit = logic.get_circuit(in_gates, out_gates)
    logic.converge(circuit)

    expected = [
        [4, 3, 3, 5],
        [5, 6, 4, 6],
        [5, 5, 5, 6],
        [4, 4, 3, 4],
    ]
    output_values = [
        [decode_number(num) for num in output_row]
        for output_row in outputs
    ]

    assert output_values == expected


def test_gol_step():
    inputs = [
        [
            constant(label=f"in_{i},{j}")
            for j in range(5)
        ]
        for i in range(5)
    ]
    input_values = [
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ]
    for input_row, value_row in zip(inputs, input_values):
        for input_, value in zip(input_row, value_row):
            input_.output_state = bool(value)
    in_gates = [gate for input_row in inputs for gate in input_row]

    outputs = gol_step(inputs)
    out_gates = [gate for output_row in outputs for gate in output_row]
    circuit = logic.get_circuit(in_gates, out_gates)
    logic.converge(circuit)

    expected = [
        [0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]
    output_values = [
        [int(gate.get_output_state()) for gate in output_row]
        for output_row in outputs
    ]
    assert output_values == expected
