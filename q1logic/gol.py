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


if __name__ == "__main__":
    num1 = [constant(label=f"num1_{i}") for i in range(2)]
    num2 = [constant(label=f"num2_{i}") for i in range(3)]

    sum_ = ripple_carry_add(num1, num2)
    circuit = logic.get_circuit(num1 + num2)

    logic.print_truth_table(num1 + num2, sum_, circuit)
