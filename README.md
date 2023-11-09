# Representation Engineering (RepE)
This is the official repository for "[Representation Engineering: A Top-Down Approach to AI Transparency](https://arxiv.org/abs/2310.01405)"  
by [Andy Zou](https://andyzoujm.github.io/), [Long Phan](https://longphan.ai/), [Sarah Chen](https://www.linkedin.com/in/sarah-chen1/), [James Campbell](https://www.linkedin.com/in/jamescampbell57), [Phillip Guo](https://www.linkedin.com/in/phillip-guo), [Richard Ren](https://github.com/notrichardren), [Alexander Pan](https://aypan17.github.io/), [Xuwang Yin](https://xuwangyin.github.io/), [Mantas Mazeika](https://www.linkedin.com/in/mmazeika), [Ann-Kathrin Dombrowski](https://scholar.google.com/citations?user=YoNVKCYAAAAJ&hl=en), [Shashwat Goel](https://in.linkedin.com/in/shashwatgoel42), [Nathaniel Li](https://nat.quest/), [Michael J. Byun](https://www.linkedin.com/in/michael-byun), [Zifan Wang](https://sites.google.com/west.cmu.edu/zifan-wang/home), [Alex Mallen](https://www.linkedin.com/in/alex-mallen-815b01176), [Steven Basart](https://stevenbas.art/), [Sanmi Koyejo](https://cs.stanford.edu/~sanmi/), [Dawn Song](https://dawnsong.io/), [Matt Fredrikson](https://www.cs.cmu.edu/~mfredrik/), [Zico Kolter](https://zicokolter.com/), and [Dan Hendrycks](https://people.eecs.berkeley.edu/~hendrycks/).

Check out our [website and demo here](https://www.ai-transparency.org/).

<img align="center" src="assets/repe_splash.png" width="750">

## Introduction
In this paper, we introduce and characterize the emerging area of representation engineering (RepE), an approach to enhancing the transparency of AI systems that draws on insights from cognitive neuroscience. RepE places population-level representations, rather than neurons or circuits, at the center of analysis, equipping us with novel methods for monitoring and manipulating high-level cognitive phenomena in deep neural networks (DNNs). We provide baselines and an initial analysis of RepE techniques, showing that they offer simple yet effective solutions for improving our understanding and control of large language models. We showcase how these methods can provide traction on a wide range of safety-relevant problems, including truthfulness, memorization, power-seeking, and more, demonstrating the promise of representation-centered transparency research. We hope that this work catalyzes further exploration of RepE and fosters advancements in the transparency and safety of AI systems.

## Installation

To install `repe` from the github repository main branch, run:

```bash
git clone https://github.com/andyzoujm/representation-engineering.git
cd representation-engineering
pip install -e .
```
## Quickstart

Our RepReading and RepControl pipelines inherit the [🤗 Hugging Face pipelines](https://huggingface.co/docs/transformers/main_classes/pipelines) for both classification and generation.

```python
from repe import repe_pipeline_registry # register 'rep-reading' and 'rep-control' tasks into Hugging Face pipelines
repe_pipeline_registry()

# ... initializing model and tokenizer ....

rep_reading_pipeline =  pipeline("rep-reading", model=model, tokenizer=tokenizer)
rep_control_pipeline =  pipeline("rep-control", model=model, tokenizer=tokenizer, **control_kwargs)
```

## RepReading and RepControl Experiments
Check out [example frontiers](./examples) of Representation Engineering (RepE), containing both RepControl and RepReading implementation. We welcome community contributions as well!

## RepE_eval
We also release a language model evaluation framework [RepE_eval](./repe_eval) based on RepReading that can serve as an additional baseline beside zero-shot and few-shot on standard benchmarks. Please check out our [paper](https://arxiv.org/abs/2310.01405) for more details.

## Citation
If you find this useful in your research, please consider citing:

```
@misc{zou2023transparency,
      title={Representation Engineering: A Top-Down Approach to AI Transparency}, 
      author={Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, Alexander Pan, Xuwang Yin, Mantas Mazeika, Ann-Kathrin Dombrowski, Shashwat Goel, Nathaniel Li, Michael J. Byun, Zifan Wang, Alex Mallen, Steven Basart, Sanmi Koyejo, Dawn Song, Matt Fredrikson, Zico Kolter, Dan Hendrycks},
      year={2023},
      eprint={2310.01405},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
