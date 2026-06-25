"""Convert a VG-SGG dictionary file to the label/predicate/attribute index format."""
import argparse
import json


def convert(dict_file: str, converted_file: str) -> None:
    with open(dict_file, 'r') as f:
        data = json.load(f)

    converted_data = {
        'label_to_idx': {v: int(k) for k, v in data['idx_to_label'].items()},
        'predicate_to_idx': {v: int(k) for k, v in data['idx_to_predicate'].items()},
    }

    if 'idx_to_attribute' in data:
        converted_data['attribute_to_idx'] = {v: int(k) for k, v in data['idx_to_attribute'].items()}
    else:
        converted_data['attribute_to_idx'] = {'__background__': 0}

    # Ensure background classes exist with index 0
    converted_data['label_to_idx'].setdefault('__background__', 0)
    converted_data['predicate_to_idx'].setdefault('__background__', 0)

    with open(converted_file, 'w') as f:
        json.dump(converted_data, f, indent=2)

    print(f"Converted file saved to {converted_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dict_file", help="Path to the input VG-SGG-dicts.json")
    parser.add_argument("converted_file", help="Path to write the converted dict")
    args = parser.parse_args()
    convert(args.dict_file, args.converted_file)
