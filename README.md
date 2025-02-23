# Conformal Language Model Reasoning with Coherent Factuality
This is the repository for the paper "Conformal Language Model Reasoning with Cohrerent Factuality" by [Maya Gambhir](mayapalgambhir.com), [Maxon Rubin Toles](https://maxrubintoles.github.io/), [Keshav Ramji](https://keshavramji.com/), [Aaron Roth](https://www.cis.upenn.edu/~aaroth/), and [Surbhi Goel](surbhigoel.com).

Much of this code was forked from the [repository](https://github.com/tatsu-lab/conformal-factual-lm/blob/main/README.md?plain=1) by Mohri and Hasimoto for their paper Language Models with Conformal Factuality Guarantees.

## Setup
Export appropriate API keys for LLama or GPT replication.  

## Files
- 'validate.py': Run this file after setup to view plots
- 'gold_annos.py': convenient annotation UI in terminal
- 'graphs.py : generates dependency graphs for a list of subclaims
- 'non_conformity.py' : helper functions to calculate non-conformity scores
- 'sayless.py' : breaks down outputs into subclaims, and merges subclaims back into paragraphs
- 'greedy.py', 'simult.py' implement both greedy and subgraph filtering algorithms respectively
- 'get_subs_scores.py': 
- 'gen_calib.py': 

## Datasets
We make use of both [MATH](https://arxiv.org/abs/2103.03874) and [FELM](http://arxiv.org/abs/2310.00741) datasets in our evaluations. 
