from . import bcd
from . import logic

def _set_digit_constants(digit, val):
    for idx, gate in enumerate(digit):
        gate.output_state = (val & (1 << idx)) != 0


def _set_digits_constants(digits, val):
    for idx, digit in enumerate(digits):
        digit_val = val % 10
        val = val // 10
        _set_digit_constants(digit, digit_val)


def _decode_bcd(digits):
    val = 0
    for digit in reversed(digits):
        val *= 10
        val += logic.decode_number(digit)
    return val


def test_add_digit():
    digit1 = [logic.constant(label=f"digit1_{i}") for i in range(4)]
    digit2 = [logic.constant(label=f"digit2_{i}") for i in range(4)]
    sum_digit, carry = bcd.bcd_add_digit(digit1, digit2)
    in_gates = digit1 + digit2
    circuit = logic.get_circuit(in_gates, sum_digit + [carry])

    for val1 in range(10):
        for val2 in range(10):
            _set_digit_constants(digit1, val1)
            _set_digit_constants(digit2, val2)
            logic.converge(circuit)

            sum_val = logic.decode_number(sum_digit)
            if carry.get_output_state():
                sum_val += 10
            assert sum_val == val1 + val2


def test_bcd_ripple_carry_adder():
    digits1 = [
        [logic.constant(label=f"digits1_{i}") for i in range(4)]
        for j in range(3)
    ]
    digits2 = [
        [logic.constant(label=f"digits2_{i}") for i in range(4)]
        for j in range(3)
    ]
    test_cases = [
        [0, 0], [1, 0], [9, 1], [123, 5], [123, 10], [123, 123], [999, 999]
    ]
    test_cases.extend([[b, a] for a, b in test_cases])

    sum_digits, carry = bcd.bcd_ripple_carry_adder(digits1, digits2)
    in_gates = [gate for gates in digits1 + digits2 for gate in gates]
    out_gates = [gate for gates in sum_digits for gate in gates]
    circuit = logic.get_circuit(in_gates, out_gates)

    for val1, val2 in test_cases:
        _set_digits_constants(digits1, val1)
        _set_digits_constants(digits2, val2)
        logic.converge(circuit)

        sum_val = _decode_bcd(sum_digits)
        if carry.get_output_state():
            sum_val += 1000

        assert sum_val == val1 + val2


def test_decode_7_segment():
    expected = [
       # a  b  c  d  e  f  g
        [1, 1, 1, 1, 1, 1, 0],  # 0
        [0, 1, 1, 0, 0, 0, 0],  # 1
        [1, 1, 0, 1, 1, 0, 1],  # 2
        [1, 1, 1, 1, 0, 0, 1],  # 3
        [0, 1, 1, 0, 0, 1, 1],  # 4
        [1, 0, 1, 1, 0, 1, 1],  # 5
        [1, 0, 1, 1, 1, 1, 1],  # 6
        [1, 1, 1, 0, 0, 0, 0],  # 7
        [1, 1, 1, 1, 1, 1, 1],  # 8
        [1, 1, 1, 1, 0, 1, 1],  # 9
    ]

    digit = [logic.constant(label=f"digit_{i}") for i in range(4)]
    segments = bcd.decode_7_segment(digit)
    circuit = logic.get_circuit(digit, segments)
    for val in range(10):
        _set_digit_constants(digit, val)
        logic.converge(circuit)

        segment_values = [0 + seg.get_output_state() for seg in segments]
        assert expected[val] == segment_values
