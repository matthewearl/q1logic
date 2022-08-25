from . import logic
from .logic import nand, ripple_carry_add, mux, half_adder


def _full_increment_adder(a, carry_in):
    """Like a full adder, but one of the inputs is fixed to 1"""
    carry_out = nand(a, carry_in, inverted_inputs=(0, 1))
    out = nand(carry_out, nand(a, carry_in))
    return out, carry_out

# a carry_in   out carry_out
# - --------   --- ---------
# 0 0            1 0
# 0 1            0 1
# 1 0            0 1
# 1 1            1 1


def bcd_add_digit(digit1, digit2):
    binary_sum = ripple_carry_add(digit1, digit2)

    carry = nand(
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
        mux(carry, binary_sum[1], incremented_bits[0]),
        mux(carry, binary_sum[2], incremented_bits[1]),
        mux(carry, binary_sum[3], incremented_bits[2]),
    ], carry

