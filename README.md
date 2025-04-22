# Conformal Language Model Reasoning with Coherent Factuality
This is the repository for the paper ["Conformal Language Model Reasoning with Cohrerent Factuality"](https://openreview.net/forum?id=AJpUZd8Clb)
_In Proceedings of the International Conference on Learning Representations (ICLR), 2025_
by [Maxon Rubin-Toles](https://maxrubintoles.github.io/), [Maya Gambhir](mayapalgambhir.com), [Keshav Ramji](https://keshavramji.com/), [Aaron Roth](https://www.cis.upenn.edu/~aaroth/), and [Surbhi Goel](surbhigoel.com).

Much of this code was forked from Mohri and Hashimoto's [repository](https://github.com/tatsu-lab/conformal-factual-lm/blob/main/README.md?plain=1) for ["Language Models with Conformal Factuality Guarantees"](https://arxiv.org/abs/2402.10978).

## Running Scripts
Python scripts are intended to be run as modules from the root directory `Conformal_LM_Reasoning`.

## API Key Setup
To store API keys, you can set environment variables as follows:

`export TOGETHER_API_KEY=your-together-key-here`  
`export OAI_KEY=your-openai-key-here`

If you would like to use an API besides OpenAI or Together, update the code in `gen_calib.py`.

## Data
We automatically load select problems from the [MATH](https://arxiv.org/abs/2103.03874) dataset, but you can add new sets to the `data` repository.

## Steps
1. Run gen_calib.py with preferred dataset/API to generate outputs split into subclaims, stored in `[dataset prefix]_[“open” if using Together]_subclaims.py`.
2. Annotate each subclaim by changing “manual_annotation” (defaults to 0.0 for false) to 1.0 if a subclaim is “independently” factual. 
3. Run `gold_annos.py` which prints all “coherent” orderings for each question (possibly exponentially many) for manual evaluation. Stored as `[filename]_annotations.json`.
4. Run `validate.py` with desired calibration/validation annotation styles to obtain plots.

### Additional Scripts
- `graphs.py`: generates dependency graphs for a list of subclaims
- `non_conformity.py`: contains helper functions to calculate non-conformity scores
- `sayless.py`: breaks down outputs into subclaims and merges subclaims back into paragraphs
- `greedy.py`, `simult.py`: implement greedy and subgraph filtering algorithms respectively
- `get_subs_scores.py`: queries model for outputs, splits into subclaims, stores in .json

### Datasets
We analyze performance on both [MATH](https://arxiv.org/abs/2103.03874) and [FELM](http://arxiv.org/abs/2310.00741) in our evaluations.
