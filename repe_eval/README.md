# Language Model Representation Evaluation (RepE Eval)

## Overview

This framework provides an approach to evaluate the representations of LLMs on different standard benchmarks. For more details about evaluation, please check out [our RepE paper](https://arxiv.org/abs/2310.01405). 

## Install

To install `repe`, run:

```bash
git clone https://github.com/andyzoujm/representation-engineering.git
cd representation-engineering
pip install -e .
```

## Basic Usage

To evaluate a language model's representations on a specific task, use the following command:

```bash
python rep_reading_eval.py \
    --model_name_or_path $model_name_or_path \
    --task $task \
    --ntrain $ntrain \
    --seed $seed
```

## Examples

For hands-on examples on how to evaluate both decoder and encoder models, please refer to our [example notebooks](./examples). Additionally, [command line scripts](./scripts) are provided to reproduce the results reported in [our RepE paper](https://arxiv.org/abs/2310.01405).

## Citation
If you find this useful in your research, please consider citing:

```bibtex
@misc{zou2023transparency,
      title={Representation Engineering: A Top-Down Approach to AI Transparency}, 
      author={Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, Alexander Pan, Xuwang Yin, Mantas Mazeika, Ann-Kathrin Dombrowski, Shashwat Goel, Nathaniel Li, Michael J. Byun, Zifan Wang, Alex Mallen, Steven Basart, Sanmi Koyejo, Dawn Song, Matt Fredrikson, Zico Kolter, Dan Hendrycks},
      year={2023},
      eprint={2310.01405},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```





