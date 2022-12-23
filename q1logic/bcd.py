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

from textwrap import dedent

from . import logic
from .logic import nand, ripple_carry_add, mux, half_adder


def _full_increment_adder(a, carry_in):
    """Like a full adder, but one of the inputs is fixed to 1"""
    carry_out = nand(a, carry_in, inverted_inputs=(0, 1))
    out = nand(carry_out, nand(a, carry_in))
    return out, carry_out


def bcd_add_digit(digit1, digit2, carry_in=None):
    binary_sum = ripple_carry_add(digit1, digit2, carry_in)

    carry_out = nand(
        nand(binary_sum[3], binary_sum[2]),
        nand(binary_sum[3], binary_sum[1]),
        binary_sum[4],
        inverted_inputs=(2,)
    )

    # `incremented_bits` is the middle 3 bits of `binary_sum`, incremented by 3.
    incremented_bits = []

    incremented_bits.append(nand(binary_sum[1]))
    bit, binary_carry = _full_increment_adder(binary_sum[2], binary_sum[1])
    incremented_bits.append(bit)
    bit, _ = half_adder(binary_sum[3], binary_carry)
    incremented_bits.append(bit)

    return [
        binary_sum[0],
        mux(carry_out, binary_sum[1], incremented_bits[0]),
        mux(carry_out, binary_sum[2], incremented_bits[1]),
        mux(carry_out, binary_sum[3], incremented_bits[2]),
    ], carry_out


def bcd_ripple_carry_adder(digits1, digits2):
    if len(digits1) != len(digits2):
        # We could handle this if needed.
        raise ValueError("Digits must be same length")

    sum_digits = []
    carry =  None
    for digit1, digit2 in zip(digits1, digits2):
        sum_digit, carry = bcd_add_digit(digit1, digit2, carry)
        sum_digits.append(sum_digit)

    return sum_digits, carry


def format_7_segment(digit_vals):
    chars = dedent("""\
     ---
    |   |
     ---
    |   |
     ---
    """)
    segs = dedent("""\
     000
    5   1
     666
    4   2
     333
    """)
    assert len(chars) == len(segs)

    out = ''
    for c1, c2 in zip(chars, segs):
        if c2.isdigit():
            if digit_vals[int(c2)]:
                out = out + c1
            else:
                out = out + ' '
        else:
            out = out + c1
    return out


def decode_7_segment(digit):
    D, C, B, A = digit

    t1 = nand(B, D, inverted_inputs=(0, 1))
    a = nand(nand(B, D), t1, A, C, inverted_inputs=(2, 3))
    t2 = nand(C, D, inverted_inputs=(0, 1))
    b = nand(nand(C, D), B, t2)
    c = nand(B, C, D, inverted_inputs=(0, 2))
    t3 = nand(C, D, inverted_inputs=(1,))
    t4 = nand(B, C, inverted_inputs=(0,))
    t5 = nand(B, C, inverted_inputs=(1,))
    d = nand(
        A, t1, t3, t4,
        nand(D, t5, inverted_inputs=(1,)),
        inverted_inputs=(0,)
    )
    e = nand(t3, t1)
    f = nand(t2, nand(B, D, inverted_inputs=(1,)), t5, A, inverted_inputs=(3,))
    g = nand(t3, t5, A, t4, inverted_inputs=(2,))

    return [a, b, c, d, e, f, g]


def _set_digit_constants(digit, val):
    for idx, gate in enumerate(digit):
        gate.output_state = (val & (1 << idx)) != 0


if __name__ == "__main__":
    digit = [logic.constant(label=f"digit_{i}") for i in range(4)]
    segments = decode_7_segment(digit)
    circuit = logic.get_circuit(digit, segments)

    for val in range(10):
        _set_digit_constants(digit, val)
        logic.converge(circuit)

        s = format_7_segment([seg.get_output_state() for seg in segments])
        print(val)
        print(s)
