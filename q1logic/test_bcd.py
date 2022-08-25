from . import bcd
from . import logic

def _set_digit_constants(digit, int_):
    for idx, gate in enumerate(digit):
        gate.output_state = (int_ & (1 << idx)) != 0


def test_add_digit():
    digit1 = [logic.constant(label=f"digit1_{i}") for i in range(4)]
    digit2 = [logic.constant(label=f"digit2_{i}") for i in range(4)]
    sum_digit, carry = bcd.bcd_add_digit(digit1, digit2)

    in_gates = digit1 + digit2
    circuit = logic.get_circuit(in_gates, sum_digit + [carry])

    for int1 in range(10):
        for int2 in range(10):
            _set_digit_constants(digit1, int1)
            _set_digit_constants(digit2, int2)
            logic.converge(circuit)

            sum_int = logic.decode_number(sum_digit)
            if carry.get_output_state():
                sum_int += 10
            assert sum_int == int1 + int2

