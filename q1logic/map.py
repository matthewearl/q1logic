import dataclasses
import functools
import io
import json
import logging
import operator
from typing import Dict, List, Any, Optional

import numpy as np

from . import logic, gol, bcd


logger = logging.getLogger(__name__)

bbox_min_planes = np.array([
    [[0, 0, 0], [0, 1, 0], [0, 0, 1]],  # x min
    [[0, 0, 0], [0, 0, 1], [1, 0, 0]],  # y min
    [[0, 0, 0], [1, 0, 0], [0, 1, 0]],  # z min
])


bbox_max_planes = np.array([
    [[0, 0, 0], [0, 0, 1], [0, 1, 0]],  # x max
    [[0, 0, 0], [1, 0, 0], [0, 0, 1]],  # y max
    [[0, 0, 0], [0, 1, 0], [1, 0, 0]],  # z max
])


def _encode_string(s):
    if '"' in s:
        raise Exception("Double-quotes not allowed in string: {s!r}")
    return f'"{s}"'


def _encode_vec(vec):
    return ' '.join(str(x) for x in vec)


@dataclasses.dataclass
class Brush:
    planes: np.ndarray
    texture: str
    comment: Optional[str] = None

    @classmethod
    def from_bbox(cls, mins: np.ndarray, maxs: np.ndarray, *args, **kwargs):
        planes = np.concatenate([
            mins + bbox_min_planes,
            maxs + bbox_max_planes,
        ])
        return cls(planes, *args, **kwargs)

    def write(self, f):
        if self.comment is not None:
            f.write(f'// {self.comment}\n')
        f.write('{\n')
        for plane in self.planes:
            f.write(' '.join('( ' + ' '.join(str(x) for x in vert) + ' )'
                               for vert in plane))
            f.write(f' {self.texture} 0 0 0 1 1\n')
        f.write('}\n')


@dataclasses.dataclass
class Entity:
    properties: Dict[str, Any]
    brushes: List[Brush]
    comment: Optional[str] = None

    def write(self, f):
        if self.comment is not None:
            f.write(f'// {self.comment}\n')
        f.write('{\n')
        for key, value in self.properties.items():
            f.write(_encode_string(key) + ' '
                    + _encode_string(str(value)) + '\n')
        for brush in self.brushes:
            brush.write(f)
        f.write('}\n')


@dataclasses.dataclass
class Map:
    entities: List[Entity]
    def write(self, f):
        f.write('// Game: Quake\n')
        f.write('// Format: Standard\n')
        for entity in self.entities:
            entity.write(f)

# monster_army bbox:  (-16, -16, -24), (16, 16, 40)

def create_input(origin: np.ndarray, name: str, target: str) -> List[Entity]:
    sled = Entity(
        {
            'classname': 'func_door',
            'angle': 90,
            'lip': 32,
            'health': 2,
            'spawnflags': 36,
            'wait': 0.1,
            'targetname': name,
        },
        [
            Brush.from_bbox(np.array([0, 0, 0]) + origin,
                            np.array([128, 16, 128]) + origin,
                            "cop1_1", "front"),
            Brush.from_bbox(np.array([64 - 24, 16, 16]) + origin,
                            np.array([64 + 24, 80, 32]) + origin,
                            "cop1_1", "bottom"),
            Brush.from_bbox(np.array([64 - 24, 64, 32]) + origin,
                            np.array([64 + 24, 80, 64]) + origin,
                            "cop1_1", "back"),
        ],
        "sled"
    )
    target_ent = Entity(
        {
            'classname': 'func_button',
            'angle': 90,
            'lip': 4,
            'health': 2,
            'target': name,
        },
        [
            Brush.from_bbox(np.array([0, 12, 0]) + origin,
                            np.array([128, 16, 128]) + origin,
                            "cop1_1", "front"),
        ],
        "target"
    )
    jump = Entity(
        {
            'classname': 'trigger_monsterjump',
            'angle': -2,
            'height': 32,
            'speed': 0,
        },
        [
            Brush.from_bbox(np.array([48, 16, 32]) + origin,
                            np.array([80, 128, 64]) + origin,
                            "cop1_1"),
        ]
    )
    door = Entity(
        {
            'classname': 'func_door',
            'angle': -1,
            'lip': 0,
            'wait': 0.5,
            'target': target,
        },
        [
            Brush.from_bbox(np.array([60, 128, 48]) + origin,
                            np.array([68, 136, 80]) + origin,
                            "*lava1"),
        ]
    )
    monster = Entity(
            {
                'classname': 'monster_army',
                'origin': _encode_vec(np.array([64, 40, 56]) + origin),
                'angle': 90,
            },
            [],
    )
    return [sled, target_ent, jump, door, monster]


def create_output(origin: np.ndarray, name: str):
    return [Entity(
        {
            'classname': 'func_door',
            'angle': 90,
            'lip': 0,
            'wait': 1.1,
            'targetname': name,
        },
        [
            Brush.from_bbox(np.array([0, 0, 0]) + origin,
                            np.array([128, 128, 128]) + origin,
                            "cop1_1"),
        ],
        "output"
    )]


def create_input_array(grid_origin, grid_shape, targets: List[List[str]]):
    entities = []
    input_spacing = 128 + 8
    for target_row, y in zip(targets, range(grid_shape[0])):
        for target, x in zip(target_row, range(grid_shape[1])):
            input_origin = np.array([input_spacing * x,
                                     0,
                                     input_spacing * y]) + grid_origin
            entities.extend(create_input(input_origin,
                                         f"input_{x}_{y}",
                                         target))

    brushes = [
        Brush.from_bbox(np.array([0, 4, 0]) + grid_origin,
                        np.array([grid_shape[1] * input_spacing, 8,
                                  grid_shape[0] * input_spacing]) + grid_origin,
              "*water0", "input curtain")
    ]
    return entities, brushes


def create_nand_gate(input_names, target, origin, *, inverted_inputs=()):
    entities = []
    for input_num, input_name in enumerate(input_names):
        if input_num in inverted_inputs:
            angle = -1
            z = 0
        else:
            angle = -2
            z = 64
        input_origin = np.array([input_num * 80, 0, z]) + origin
        entities.extend([
            # platform
            Entity(
                {
                    'classname': 'func_door',
                    'angle': angle,
                    'lip': 0,
                    'speed': 500,
                    'targetname': input_name
                },
                [Brush.from_bbox(input_origin, input_origin + 64, "cop1_1")]
            ),
            # monster
            Entity(
                {
                    'classname': 'monster_army',
                    'origin': _encode_vec(np.array([32, 32, 88])
                                          + input_origin),
                    'angle': 90,
                },
                [],
            )
        ])

    monster_min = 16
    monster_max = 80 * (len(input_names) - 1) + 48

    entities.extend([
        # output door
        Entity(
            {
                'classname': 'func_door',
                'angle': -1,
                'lip': 0,
                'target': target,
                'wait': 0.5,
            },
            [
                Brush.from_bbox(np.array([monster_min, 64, 176]) + origin,
                      np.array([monster_max, 72, 208]) + origin,
                      "*lava1")
            ]
        ),
        # jump
        Entity(
            {
                'classname': 'trigger_monsterjump',
                'angle': -2,
                'height': 32,
                'speed': 0,
            },
            [
                Brush.from_bbox(np.array([monster_min, 16, 64]) + origin,
                      np.array([monster_max, 48, 196]) + origin,
                      "*lava1")
            ]
        )
    ])

    return entities


def create_output_array(grid_origin, grid_size, names):
    entities = []
    input_spacing = 128 + 8
    for name_row, y in zip(names, range(grid_size)):
        for name, x in zip(name_row, range(grid_size)):
            input_origin = np.array([input_spacing * x,
                                     0,
                                     input_spacing * y]) + grid_origin
            entities.extend(create_output(input_origin, name))

    brushes = [
        Brush.from_bbox(np.array([0, 4, 0]) + grid_origin,
              np.array([grid_size * input_spacing, 8,
                        grid_size * input_spacing]) + grid_origin,
              "*water0", "output curtain")
    ]
    return entities, brushes


def create_7_segment_display(origin, names):
    if len(names) != 7:
        raise ValueError("Expected 7 names")
    with open('segments.json') as f:
        data = json.load(f)

    entities = [
        Entity(
            {
                'classname': 'func_door',
                'angle': 90,
                'lip': 0,
                'wait': 1.1,
                'spawnflags': 5,
                'targetname': name,
            },
            [Brush(origin + np.array(segment_planes), "*lava1")],
            f"segment {label}",
        )
        for label, name, segment_planes
        in zip("abcdefg", names, data['segments'])
    ]
    back_brush = Brush(origin + np.array(data['back']), 'cop1_1')

    return entities, [back_brush]


def _reshape_list(l, shape):
    if len(l) != functools.reduce(operator.mul, shape):
        raise ValueError(f"List of length {len(l)} cannot be reshaped to "
                         f"{shape}")

    for size in reversed(shape):
        l = [l[i:i + size] for i in range(0, len(l), size)]
    return l


def map_from_circuit(in_gates, out_gates, circuit):
    gate_ids = {
        gate: f"gate_{i}" for i, gate in enumerate(circuit)
    }

    world_brushes = []
    entities = []

    # Create inputs
    input_targets = [
        gate_ids[in_gate] for in_gate in in_gates
    ]

    # Create outputs
    output_names = [
        gate_ids[out_gate] for out_gate in out_gates
    ]

    # Map from gate inputs to the gates that connect to them.
    prev_gate = {
        (input_.gate, input_.input_num): gate
        for gate in circuit
        for input_ in gate.outputs
    }

    # Create nand gates
    origin = np.array([0, 512, 0])
    nand_gates = [gate for gate in circuit if isinstance(gate, logic.NandGate)]
    for gate in nand_gates:
        input_names = [
            gate_ids[prev_gate[gate, input_num]]
            for input_num in range(len(gate.inputs))
        ]

        nand_entities = create_nand_gate(input_names,
                                         gate_ids[gate],
                                         origin,
                                         inverted_inputs=gate.inverted_inputs)
        entities.extend(nand_entities)

        origin = origin + np.array([256 * len(input_names), 0, 0])
        if origin[0] > 8192:
            origin[0] = 0
            origin[2] += 256

    # Create the map object
    world_entity = Entity(
        {
            'classname': 'worldspawn',
            'wad': '../wads/wizard.wad;../wads/start.wad;../wads/base.wad',
            'angle': 0,
        },
        world_brushes
    )
    entities.insert(0, world_entity)
    return entities, world_brushes, input_targets, output_names


def create_map_entrypoint():
    logging.basicConfig(level=logging.INFO)

    player_origin = np.array([768, -1024, 0])

    in_gates = [logic.constant() for j in range(4)]
    out_gates = bcd.decode_7_segment(in_gates)
    circuit = logic.get_circuit(in_gates, out_gates)
    logger.info('Number of gates: %s', len(circuit))

    circuit_entities, circuit_brushes, input_targets, output_names = (
        map_from_circuit(
            in_gates, out_gates, circuit
        )
    )

    input_entities, input_brushes = create_input_array(
        np.array([0, 0, 0]), (1, 4), [input_targets]
    )

    output_entities, output_brushes = create_7_segment_display(
        np.array([768, 0, 0]), output_names
    )

    entities = [
        Entity(
            {
                'classname': 'worldspawn',
                'wad': '../wads/wizard.wad;../wads/start.wad;../wads/base.wad',
                'angle': 0,
            },
            [
                Brush.from_bbox(np.array([-32, -32, -40]) + player_origin,
                                np.array([32, 32, -24]) + player_origin,
                                "cop1_1", "platform"),
            ] + circuit_brushes + input_brushes + output_brushes
        ),
        Entity(
            {
                'classname': 'info_player_start',
                'angle': 90,
                'origin': _encode_vec(player_origin)
            },
            []
        ),
    ] + circuit_entities + input_entities + output_entities

    with open('test.map', 'w') as f:
        Map(entities).write(f)

