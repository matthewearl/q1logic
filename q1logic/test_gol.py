import itertools

from . import logic
from .gol import ripple_carry_add, convolve1, convolve2
from .logic import constant


def _decode_number(num):
    return sum(gate.get_output_state() << i for i, gate in enumerate(num))


def test_ripple_carry_adder():
    num1 = [constant(label=f"num1_{i}") for i in range(2)]
    num2 = [constant(label=f"num2_{i}") for i in range(3)]
    sum_ = ripple_carry_add(num1, num2)
    in_gates = num1 + num2
    circuit = logic.get_circuit(in_gates, sum_)

    for in_values in itertools.product([False, True], repeat=len(in_gates)):
        for in_gate, in_value in zip(in_gates, in_values):
            in_gate.output_state = in_value
        logic.converge(circuit)

        in_num1 = _decode_number(num1)
        in_num2 = _decode_number(num2)
        out_num = _decode_number(sum_)

        assert in_num1 + in_num2 == out_num


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
        [_decode_number(num) for num in output_row]
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
        [_decode_number(num) for num in output_row]
        for output_row in outputs
    ]

    assert output_values == expected

