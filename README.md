# DeepSoftLog-REC

A neurosymbolic model for **referring expression comprehension** (REC): given an
image and a natural-language expression such as *"the man in the red shirt to the
left of the dog"*, localise the object being referred to. Rather than learning
this end to end, the system separates **perception** from **reasoning** —
neural models read the image and the language, and a differentiable
probabilistic-logic engine ([DeepSoftLog](https://github.com/sam-mcm-642/DeepSoftLog))
reasons over the result to make the grounding decision explicit and inspectable.

Developed as part of a Master's thesis.

## How it works

The pipeline has three stages:

1. **Perception → scene graph.** A scene-graph generation model turns the image
   into a structured graph of objects and their relationships (subject–predicate–object
   triplets grounded to regions). See `src/models/sgg/`.
2. **Language → logical query.** The referring expression is parsed into a
   structured symbolic query using an LLM. See `src/models/text/triplets_chatgpt/`.
3. **Reasoning → grounding.** DeepSoftLog reasons over the scene graph to satisfy
   the query and ground it to the correct region. Its **soft unification** lets
   symbols match by embedding similarity rather than exact string equality (so
   "guy" can unify with a detected "man"), and it yields a probabilistic proof —
   an interpretable trace of *why* a region was chosen. See `train/`, `run.py`,
   `eval.py`.

## Repository structure

```
run.py / run_eval.py / eval.py     Entry points (train / evaluate)
train/                             Training loop, trainer, evaluators
data/                              Dataset/dataloader, query generation,
                                   ontologies, and DeepSoftLog .pl programs
src/models/sgg/                    Scene-graph generation (offline)
src/models/text/triplets_chatgpt/  LLM query parser (offline)
analysis/                          Analysis & plotting utilities
scripts/                           Data-prep / formatting / ontology utilities
eval/, results/                    Run configs and result summaries
```

> Notes:
> - The dependency `deepsoftlog` is a **separate package** (see Installation),
>   not vendored in this repo.
> - Run all scripts from the repository root (e.g. `python analysis/eval_analysis.py`)
>   so relative data paths resolve correctly.

## Installation

Requires Python 3.10+. A virtual environment (conda or venv) is recommended.

**1. Install the DeepSoftLog engine first.** It has Cython extensions and must be
built, so it is not in `requirements.txt`:

```bash
pip install cython           # build-time dependency
git clone https://github.com/sam-mcm-642/DeepSoftLog.git
cd DeepSoftLog
pip install -r requirements.txt
pip install -e .
cd ..
```

**2. Install this project's dependencies:**

```bash
pip install -r requirements.txt
```

**3. Configure API keys** (only needed to re-run the LLM query parser). Copy the
example and add your key — never commit the real `.env`:

```bash
cp .env.example .env   # then edit and add ANTHROPIC_API_KEY
```

## Data

The large datasets (Visual Genome scene graphs, generated queries, scene-graph
predictions, model checkpoints) are **not committed** — they are gitignored. The
repository ships only small sample/ontology files under `data/`. To run the full
pipeline you need to regenerate or download:

- **Scene graphs** — Visual Genome, processed via `src/models/sgg/`
- **Queries** — generated via `data/query/generator.py`
- **Checkpoints** — produced by training (`run.py`)

## Usage

**Train:**

```bash
python run.py referring_expression train/config.yaml
```

**Evaluate** a trained checkpoint (set its path in the eval config):

```bash
python run_eval.py
```

Analysis and plotting helpers live in `*_analysis.py` and `visualize_loss.py`.

## Status

This is research code from a thesis project. It is functional but not packaged
for production; expect rough edges, and see the offline components (scene-graph
generation, LLM parsing) which depend on heavier external tooling noted in
`requirements.txt`.

## Acknowledgements

- **DeepSoftLog** — the neurosymbolic engine, by Jaron Maene (KU Leuven); this
  project uses a [fork](https://github.com/sam-mcm-642/DeepSoftLog).
- **Visual Genome** — scene-graph data.

## License

See `LICENSE`.
