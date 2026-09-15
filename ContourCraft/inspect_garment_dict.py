#!/usr/bin/env python3
"""Print fixed-size garment topology and node-type evidence without modifying it."""

import argparse
import pickle
from pathlib import Path

import numpy as np


def shape(value):
    return tuple(value.shape) if hasattr(value, "shape") else type(value).__name__


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("garment_dict", type=Path)
    args = parser.parse_args()
    with args.garment_dict.open("rb") as handle:
        garment = pickle.load(handle)

    print(f"file: {args.garment_dict.resolve()}")
    print("keys:")
    for key in sorted(garment):
        print(f"  {key}: {shape(garment[key])}")

    vertices = garment.get("rest_pos")
    if vertices is None and isinstance(garment.get("lbs"), dict):
        vertices = garment["lbs"].get("v")
    faces = garment.get("faces")
    if faces is None and isinstance(garment.get("lbs"), dict):
        faces = garment["lbs"].get("f")
    print(f"vertex_count: {len(vertices) if vertices is not None else 'missing'}")
    print(f"face_count: {len(faces) if faces is not None else 'missing'}")

    node_type = garment.get("node_type")
    if node_type is None:
        print("node_type_present: false")
        print("pinned_value_3_count: 0")
    else:
        values, counts = np.unique(np.asarray(node_type), return_counts=True)
        summary = {int(value): int(count) for value, count in zip(values, counts)}
        print("node_type_present: true")
        print(f"node_type_counts: {summary}")
        print(f"pinned_value_3_count: {summary.get(3, 0)}")


if __name__ == "__main__":
    main()
