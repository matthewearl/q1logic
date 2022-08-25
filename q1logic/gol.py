from . import logic
from .logic import nand, constant


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


def ripple_carry_add(num1, num2):
    assert len(num1) > 0 and len(num2) > 0, "not supported"

    carry = None
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
