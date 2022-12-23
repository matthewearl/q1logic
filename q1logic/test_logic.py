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
from .logic import decode_number, ripple_carry_add

def test_mux():
    a = logic.constant(label="a")
    b = logic.constant(label="b")
    select = logic.constant(label="select")
    out = logic.mux(select, a, b)

    circuit = logic.get_circuit([a, b, select], [out])

    for a_val, b_val in itertools.product([False, True], repeat=2):
        a.output_state = a_val
        b.output_state = b_val

        select.output_state = False
        logic.converge(circuit)
        assert out.get_output_state() == a_val

        select.output_state = True
        logic.converge(circuit)
        assert out.get_output_state() == b_val


def test_ripple_carry_adder():
    num1 = [logic.constant(label=f"num1_{i}") for i in range(2)]
    num2 = [logic.constant(label=f"num2_{i}") for i in range(3)]
    sum_ = ripple_carry_add(num1, num2)
    in_gates = num1 + num2
    circuit = logic.get_circuit(in_gates, sum_)

    for in_values in itertools.product([False, True], repeat=len(in_gates)):
        for in_gate, in_value in zip(in_gates, in_values):
            in_gate.output_state = in_value
        logic.converge(circuit)

        in_num1 = decode_number(num1)
        in_num2 = decode_number(num2)
        out_num = decode_number(sum_)

        assert in_num1 + in_num2 == out_num
