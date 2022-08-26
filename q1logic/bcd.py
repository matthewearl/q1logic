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
