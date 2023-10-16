import fire
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import numpy as np
from itertools import islice
from repe import repe_pipeline_registry
repe_pipeline_registry()

from tasks import task_dataset

def main(
        model_name_or_path,
        task,
        ntrain,
        n_components = 1,
        rep_token = -1,
        max_length = 2048,
        n_difference = 1,
        direction_method = 'pca',
        batch_size = 8,
        seed=0,
):
    print("model_name_or_path", model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto")
    use_fast_tokenizer = "LlamaForCausalLM" not in model.config.architectures
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=use_fast_tokenizer, padding_side="left", legacy=False)
    tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    tokenizer.bos_token_id = 1
    hidden_layers = list(range(-1, -model.config.num_hidden_layers, -1))

    rep_pipeline =  pipeline("rep-reading", model=model, tokenizer=tokenizer)
    dataset = task_dataset(task)(ntrain=ntrain, seed=seed)

    n_difference = 1
    direction_finder_kwargs= {"n_components": n_components}

    rep_reader = rep_pipeline.get_directions(
        dataset['train']['data'], 
        rep_token=rep_token, 
        hidden_layers=hidden_layers, 
        n_difference=n_difference, 
        train_labels=dataset['train']['labels'], 
        direction_method=direction_method,
        direction_finder_kwargs=direction_finder_kwargs,
        batch_size=batch_size,
        max_length=max_length,
        padding="longest",
    )

    results = {'val': [], 'test': []}
    datasets = [('val', dataset['val']), ('test', dataset['test'])]

    for t, eval_data in datasets:
        if not eval_data: continue

        H_tests = rep_pipeline(
            eval_data['data'],
            rep_token=rep_token,
            hidden_layers=hidden_layers,
            rep_reader=rep_reader,
            batch_size=batch_size,
            max_length=max_length,
            padding="longest"
        )

        labels = eval_data['labels']
        for layer in hidden_layers:
            H_test = [H[layer] for H in H_tests]

            # unflatten into chunks of choices
            unflattened_H_tests = [list(islice(H_test, sum(len(c) for c in labels[:i]), sum(len(c) for c in labels[:i+1]))) for i in range(len(labels))]

            sign = rep_reader.direction_signs[layer]
            eval_func = np.argmin if sign == -1 else np.argmax
            cors = np.mean([labels[i].index(1) == eval_func(H) for i, H in enumerate(unflattened_H_tests)])

            results[t].append(cors)

    if dataset['val']:
        best_layer_idx = results['val'].index(max(results['val']))
        best_layer = hidden_layers[best_layer_idx]
        print(f"Best validation acc at layer: {best_layer}; acc: {max(results['val'])}")
        print(f"Test Acc for chosen layer: {best_layer} - {results['test'][best_layer_idx]}")
    else:
        best_layer_idx = results['test'].index(max(results['test']))
        best_layer = hidden_layers[best_layer_idx]
        print(f"Best test acc at layer: {best_layer}; acc: {max(results['test'])}")
        
if __name__ == "__main__":
    fire.Fire(main)