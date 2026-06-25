"""Extract object/predicate class mappings from a VG-SGG .h5 file into a JSON dict."""
import argparse
import json
import os

import h5py


def extract_mappings(h5_path: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    with h5py.File(h5_path, 'r') as f:
        print("Available keys in the h5 file:", list(f.keys()))

        object_classes = (
            [x.decode('utf-8') for x in f['object_classes'][:]] if 'object_classes' in f else []
        )
        predicate_classes = (
            [x.decode('utf-8') for x in f['predicate_classes'][:]] if 'predicate_classes' in f else []
        )
        print(f"Found {len(object_classes)} object classes, {len(predicate_classes)} predicate classes")

        class_dict = {
            "idx_to_label": {str(i): cls for i, cls in enumerate(object_classes)},
            "idx_to_predicate": {str(i): pred for i, pred in enumerate(predicate_classes)},
            "idx_to_attribute": {"0": "no_attribute"},
        }

        out_path = os.path.join(output_dir, 'VG-SGG-dicts-with-attri.json')
        with open(out_path, 'w') as json_file:
            json.dump(class_dict, json_file, indent=2)
        print(f"Saved class mappings to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5_path", help="Path to the VG-SGG .h5 file")
    parser.add_argument("output_dir", help="Directory to write the JSON mappings")
    args = parser.parse_args()
    extract_mappings(args.h5_path, args.output_dir)
