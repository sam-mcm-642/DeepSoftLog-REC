import os
print("PYTHONPATH:", os.getenv("PYTHONPATH"))

from deepsoftlog.data import ontology_to_prolog, load_csv_file
from pathlib import Path

def generate_ontology_rules():
    base_path = Path(__file__).parent
    (base_path / 'program').mkdir(exist_ok=True)
    data = load_csv_file(base_path / f"ontology/sample_ontology.csv")
    data = ontology_to_prolog(data, name="ontology")
    file_str = [f"{query}." for query in data]
    print(file_str)
    with open(base_path / f"program/initial_program.pl", "w+") as f:
        f.write("\n".join(file_str))
    # add template stuff
    with open(base_path / f"rules.pl", "r") as f:
        templates = f.read()
    with open(base_path / f"program/initial_program.pl", "a+") as f:
        f.write("\n" + templates)

def create_rogram():
    pass

if __name__ == "__main__":
    generate_ontology_rules()