# %%
HUGGINGFACE_CACHE = '/usr1/data/models_cache'
HF_HOME = HUGGINGFACE_CACHE
HF_DATASETS_CACHE = HUGGINGFACE_CACHE
TRANSFORMERS_CACHE = HUGGINGFACE_CACHE
import os
os.environ['HF_HOME'] = HF_HOME
os.environ['HF_DATASETS_CACHE'] = HF_DATASETS_CACHE
os.environ['TRANSFORMERS_CACHE'] = TRANSFORMERS_CACHE

# %%
from transformers import AutoTokenizer, AutoConfig, pipeline, AutoModelForCausalLM
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
import torch.nn.functional as F
import gc
from itertools import islice
import pdb
from repe import repe_pipeline_registry
from repe.rep_control_contrast_vec import ContrastVecLlamaForCausalLM, ContrastVecMistralForCausalLM
import random

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# torch.use_deterministic_algorithms(True)

repe_pipeline_registry()

from tasks import task_dataset

import fire

# %%
def batchify(lst, batch_size):
    """Yield successive batch_size chunks from lst."""
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]

def load_tqa_sentences(user_tag, assistant_tag, preset=""):
    dataset = load_dataset('truthful_qa', 'multiple_choice')['validation']
    questions, answers = [],[]
    labels = []
    for d in dataset:
        q = d['question']
        for i in range(len(d['mc1_targets']['labels'])):
            a = d['mc1_targets']['choices'][i]
            questions = [f'{user_tag}' + q + ' ' + preset] + questions
            answers = [f'{assistant_tag}' + a] + answers
        ls = d['mc1_targets']['labels']
        ls.reverse()
        labels.insert(0, ls)
    return questions, answers, labels

def get_logprobs(logits, input_ids, masks, **kwargs):
    logprobs = F.log_softmax(logits, dim=-1)[:, :-1]
    # find the logprob of the input ids that actually come next in the sentence
    logprobs = torch.gather(logprobs, -1, input_ids[:, 1:, None])
    logprobs = logprobs * masks[:, 1:, None] 
    return logprobs.squeeze(-1)
    
def prepare_decoder_only_inputs(prompts, targets, tokenizer, device):
    tokenizer.padding_side = "left"
    prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)
    tokenizer.padding_side = "right"
    target_inputs = tokenizer(targets, return_tensors="pt", padding=True, truncation=False, add_special_tokens=False)
    
    # concatenate prompt and target tokens and send to device
    inputs = {k: torch.cat([prompt_inputs[k], target_inputs[k]], dim=1).to(device) for k in prompt_inputs}

    # mask is zero for padding tokens
    mask = inputs["attention_mask"].clone()
    # set mask to 0 for question tokens
    mask[:, :prompt_inputs["input_ids"].shape[1]] = 0
    mask.to(device)
    # remove token_type_ids
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]
    
    return inputs, mask, prompt_inputs["input_ids"].shape[1]

def calc_acc(labels, output_logprobs):
    # check if the max logprob corresponds to the correct answer
    correct = np.zeros(len(labels))
    # indices to index
    indices = np.cumsum([len(l) for l in labels])
    indices = np.insert(indices, 0, 0)
    for i, label in enumerate(labels):
        # check 
        log_probs = output_logprobs[indices[i]:indices[i+1]]
        correct[i] = np.argmax(log_probs) == label.index(1)
    return correct.mean()

def get_tqa_outputs(model, questions, answers, labels, tokenizer, batch_size=128):
    gc.collect()
    # get the log probabilities of each question answer pair
    output_logprobs = []
    for q_batch, a_batch in tqdm(zip(batchify(questions, batch_size), batchify(answers, batch_size)), total=len(questions)//batch_size):
        # print(q_batch[0] + a_batch[0])
        inputs, masks, _ = prepare_decoder_only_inputs(q_batch, a_batch, tokenizer, model.model.device)
        with torch.no_grad():
            try:
                # set the masks so that we do not add to tokens of input sentences and padding tokens
                model.set_masks(masks.unsqueeze(-1))
            except:
                pass

            # calculate the probabilities for all tokens (all question answer pairs)
            logits = model(**inputs).logits
            # sum the probabilities for each question answer pair so that each pair has one probability
            # mask is zero for question and padding tokens
            logprobs = get_logprobs(logits, inputs['input_ids'], masks).sum(-1).detach().cpu().numpy()
        output_logprobs.extend(logprobs)
    
    # clear the masks
    try:
        model.set_masks(None)
    except:
        pass
    gc.collect()
    return output_logprobs

def get_baselines_for_model(
        model,
        tokenizer
):

    # parameters
    batch_size = 64

    user_tag = "[INST] "
    assistant_tag = "[/INST] "

    experimental_prompt = 'Give a truthful answer. '
    reference_prompt = 'Give an untruthful answer. '


    questions, answers, labels = load_tqa_sentences(user_tag=user_tag, assistant_tag=assistant_tag, preset="")
    questions_baseline_stripped = [q.replace(user_tag, "").strip() for q in questions]
    output_logprobs_baseline = get_tqa_outputs(model, questions, answers, labels, tokenizer, batch_size=batch_size)


    # merge the outputs for each question 
    unique_questions = []
    for q in questions_baseline_stripped:
        if q not in unique_questions:
            unique_questions.append(q)
    output_logprobs_baseline_merged = {}
    for q in unique_questions:
        indices = [i for i, x in enumerate(questions_baseline_stripped) if x == q]
        output_logprobs_baseline_merged[q] = np.array(output_logprobs_baseline)[indices]

    # with truthful prompt
    questions, answers, labels = load_tqa_sentences(user_tag=user_tag, assistant_tag=assistant_tag, preset=experimental_prompt)
    questions_experimental_stripped = [q.replace(user_tag, "").replace(experimental_prompt, "").strip() for q in questions]
    output_logprobs_truthful = get_tqa_outputs(model, questions, answers, labels, tokenizer, batch_size=batch_size)

    # merge the outputs for each question 
    unique_questions = []
    for q in questions_experimental_stripped:
        if q not in unique_questions:
            unique_questions.append(q)
    output_logprobs_truthful_merged = {}
    for q in unique_questions:
        indices = [i for i, x in enumerate(questions_experimental_stripped) if x == q]
        output_logprobs_truthful_merged[q] = np.array(output_logprobs_truthful)[indices]

    return output_logprobs_baseline_merged, output_logprobs_truthful_merged

def run_rep_control_for_model(
        tokenizer,
        model_name,
        model,
        layer_ids,
        user_tag="[INST] ",
        assistant_tag="[INST] ",
):
    
    batch_size = 128

    pos_type = 'Give a truthful answer. '
    neg_type = 'Give an untruthful answer. '
    questions, answers, labels = load_tqa_sentences(user_tag=user_tag, assistant_tag=assistant_tag, preset="")

    output_logprobs = []
    for q_batch, a_batch in tqdm(zip(batchify(questions, batch_size), batchify(answers, batch_size)), total=len(questions)//batch_size):
        gc.collect()
        inputs, masks, orig_split = prepare_decoder_only_inputs(q_batch, a_batch, tokenizer, model.model.device)

        directions = {}
        for layer_id in layer_ids:
            directions[layer_id] = 0

        q_batch_pos = [q + pos_type for q in q_batch]
        q_batch_neg = [q + neg_type for q in q_batch]
        
        inputs_pos_s, masks_pos_s, split_pos = prepare_decoder_only_inputs(q_batch_pos, a_batch, tokenizer, model.model.device)
        inputs_neg_s, masks_neg_s, split_neg = prepare_decoder_only_inputs(q_batch_neg, a_batch, tokenizer, model.model.device)
        split = inputs_neg_s['input_ids'].shape[1] - split_neg
        # print(orig_split, split_neg, split_pos)
        
        with torch.no_grad():
            logits = model(**inputs,
                    pos_input_ids=inputs_pos_s['input_ids'],
                    pos_attention_mask=inputs_pos_s['attention_mask'],
                    neg_input_ids=inputs_neg_s['input_ids'],
                    neg_attention_mask=inputs_neg_s['attention_mask'],
                    contrast_tokens=-split, # last {split} tokens
                    compute_contrast=True,
                    alpha=0.25, # try 0.1+, maybe 0.1 for mistrals
                    control_layer_ids=layer_ids,
                    ).logits
            logprobs = get_logprobs(logits, inputs['input_ids'], masks).sum(-1).detach().cpu().numpy()
        output_logprobs.extend(logprobs)

    # cache logprobs for every q,a pair
    outputs_cache = {}
    for i, (q, a) in enumerate(zip(q_batch, a_batch)):
        # directions_cache = {
        #     layer_id: directions[layer_id].detach().cpu().numpy()[i]
        #     for layer_id in layer_ids
        # }
        outputs_cache[(q, a)] = {
            # "directions": directions_cache.copy(),
            "logprobs": logprobs[i],
        }
    
    with open(f"outputs_cache_{model_name}.pkl", "wb") as f:
        pickle.dump(outputs_cache, f)

    print("Cached output logprobs")

def run_rep_reading_for_model(
        model_name,
        task,
        ntrain,
        model = None,
        tokenizer = None,
        n_components = 1,
        rep_token = -1,
        max_length = 2048,
        n_difference = 1,
        direction_method = 'pca',
        batch_size = 8,
        seed=0,
):
    print("model_name_or_path", model_name)

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

    test_preds = []
    test_cors = -1
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

        
        H_test = [H[best_layer] for H in H_tests]
        sign = rep_reader.direction_signs[best_layer]
        eval_func = np.argmin if sign == -1 else np.argmax
        # unflatten into chunks of choices
        unflattened_H_tests = [list(islice(H_test, sum(len(c) for c in labels[:i]), sum(len(c) for c in labels[:i+1]))) for i in range(len(labels))]
        test_preds = [eval_func(H) for i, H in enumerate(unflattened_H_tests)]    
        # get the questions and answers
        tqa_test_data_mc1 = load_dataset('truthful_qa', 'multiple_choice')['validation']
        groups = []
        for j, d in enumerate(tqa_test_data_mc1):
            current_group = [(d['question'], d['mc1_targets']['choices'][i], test_preds[j]) 
                             for i in range(len(d['mc1_targets']['labels']))]
            groups.append(current_group)
        
        # merge the questions and answers into a dict
        qa_dict = {}
        for group in groups:
            for q, a, pred in group:
                if q not in qa_dict:
                    qa_dict[q] = {}
                if 'answers' not in qa_dict[q]:
                    qa_dict[q]['answers'] = []
                qa_dict[q]['answers'].append(a)
        
        for group in groups:
            for q,a, pred in group:     
                prediction = np.zeros(len(qa_dict[q]['answers']))
                prediction[pred] = 1
                # reverse the answers and predictions 
                qa_dict[q]['answers'] = qa_dict[q]['answers'][::-1]
                qa_dict[q]['rep_prediction'] = prediction[::-1].tolist()
                qa_dict[q]['label'] = [0] * len(qa_dict[q]['answers'])
                qa_dict[q]['label'][-1] = 1 

        labels = dataset['test']['labels']
        test_cors = np.mean([labels[i].index(1) == pred for i, pred in enumerate(test_preds)])
        print(test_cors)
        with open(f'tqa_rep_predictions_{model_name}.pkl', 'wb') as f:
            pickle.dump(qa_dict, f)
    
    else:
        best_layer_idx = results['test'].index(max(results['test']))
        best_layer = hidden_layers[best_layer_idx]
        print(f"Best test acc at layer: {best_layer}; acc: {max(results['test'])}")
# %%
import pickle

def merge_baselines(model_name, user_tag, assistant_tag, output_logprobs_baseline_merged, output_logprobs_truthful_merged):
    with open(f"outputs_cache_{model_name}.pkl", "rb") as f:
        outputs_cache = pickle.load(f)

    with open(f"outputs_cache_{model_name}_contrast.pkl", "rb") as f:
        outputs_cache_contrast = pickle.load(f)

    def get_one_hot_prediction(logprobs):
        prediction = np.argmax(logprobs)
        one_hot = np.zeros(len(logprobs)) 
        one_hot[prediction] = 1
        return one_hot
        
    def get_label(logits_or_predictions):
        return np.argmax(logits_or_predictions, axis=-1)

    questions, answers, labels = load_tqa_sentences(user_tag=user_tag, assistant_tag=assistant_tag, preset="")

    from collections import defaultdict
    qa_output = defaultdict(lambda: defaultdict(list))
    for (question, answer), logprob in outputs_cache.items():
        question_stripped = question.replace(user_tag, "").strip()
        qa_output[question_stripped]['answer'].append(answer)
        qa_output[question_stripped]['logprobs_control'].append(logprob['logprobs'])

    for i, question in enumerate(qa_output.keys()):
        labels_for_q = labels[i]
        question_stripped = question.replace(user_tag, "")
        for i, answer in enumerate(qa_output[question_stripped]['answer']):
            qa_output[question_stripped]['labels'].append(labels_for_q[i])
        one_hot_controlled = get_one_hot_prediction(qa_output[question_stripped]['logprobs_control'])
        qa_output[question_stripped]['controlled_prediction'] = one_hot_controlled
        qa_output[question_stripped]['baseline_prediction'] = get_one_hot_prediction(output_logprobs_baseline_merged[question_stripped])
        qa_output[question_stripped]['truthful_prediction'] = get_one_hot_prediction(output_logprobs_truthful_merged[question_stripped])
        qa_output[question_stripped]['contrast_prediction'] = get_one_hot_prediction(outputs_cache_contrast[(question, answer)]['logprobs'])

        # %%
        with open(f"tqa_rep_predictions_{model_name}.pkl", "rb") as f:
            tqa_rep_predictions = pickle.load(f)
        # merge tqa_rep_predictions with qa_output
        for question, val in tqa_rep_predictions.items():
            answers, prediction = val['answers'], val['rep_prediction']
            qa_output[question]['rep_prediction'] = prediction
            assert len(prediction) == len(qa_output[question]['labels'])

        with open(f"tqa_rep_predictions_{model_name}_contrast.pkl", "rb") as f:
            tqa_rep_predictions = pickle.load(f)
        # merge tqa_rep_predictions with qa_output
        for question, val in tqa_rep_predictions.items():
            answers, prediction = val['answers'], val['rep_prediction']
            qa_output[question]['contrast_prediction'] = prediction
            assert len(prediction) == len(qa_output[question]['labels'])

        # %%
        #save the qa_output
        with open(f"qa_output_combined_{model_name}.pkl", "wb") as f:
            pickle.dump(dict(qa_output), f)

        return qa_output
    

# %%
# get confusion matrix for each question
# calculate accuracy for each question
from sklearn.metrics import confusion_matrix

def get_confusion_matrices(qa_output):
    control_correct = []
    baseline_correct = []
    rep_correct = []
    for question, val in qa_output.items():
        labels = val['labels']
        predictions = val['controlled_prediction']
        control_correct.append(np.array_equal(labels, predictions))
        predictions = val['baseline_prediction']
        baseline_correct.append(np.array_equal(labels, predictions))
        predictions = val['rep_prediction']
        rep_correct.append(np.array_equal(labels, predictions))

    # print accuracy
    print("Control accuracy:", np.mean(control_correct))
    print("Baseline accuracy:", np.mean(baseline_correct))
    print("Rep accuracy:", np.mean(rep_correct))
    # pretty-print confusion matrices between control and baseline, and rep and control, and rep and baseline
    from sklearn.metrics import ConfusionMatrixDisplay
    cm = confusion_matrix(control_correct, baseline_correct, labels=[False,True])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Incorrect","Correct"])
    disp.plot()
    disp.ax_.set(xlabel='Baseline', ylabel='Representation Control')
    plt.savefig("baseline_vs_representation_control.png")

    cm = confusion_matrix(control_correct, rep_correct, labels=[False,True])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Incorrect","Correct"])
    disp.plot()
    disp.ax_.set(xlabel='Representation Reading', ylabel='Representation Control')
    plt.savefig("representation_control_vs_representation_reading.png")

    cm = confusion_matrix(baseline_correct, rep_correct, labels=[False,True])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Incorrect","Correct"])
    disp.plot()
    disp.ax_.set(xlabel='Representation Reading', ylabel='Baseline')
    plt.savefig("baseline_vs_representation_reading.png")


# save the qa_output to a pandas dataframe
import pandas as pd

# get the data in a format that can be saved to a dataframe
def to_csv(qa_output, model_name):
    data = []
    for question, val in qa_output.items():
        for i, answer in enumerate(val['answer']):
            data.append([question, answer, val['labels'][i], val['controlled_prediction'][i], val['baseline_prediction'][i], val['rep_prediction'][i]])
    df = pd.DataFrame(data, columns=['question', 'answer', 'label', 'controlled_prediction', 'baseline_prediction', 'rep_prediction'])
    df.to_csv(f"qa_output_combined_{model_name}.csv", index=False)


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
    if "llama" in model_name_or_path.lower():
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto")
        use_fast_tokenizer = "LlamaForCausalLM" not in model.config.architectures
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=use_fast_tokenizer, padding_side="left", legacy=False)
        tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
        tokenizer.bos_token_id = 1
    else:
        raise NotImplementedError

    # get the baselines
    model.eval()
    # output_logprobs_baseline_merged, output_logprobs_truthful_merged = get_baselines_for_model(model, tokenizer)
    # get_baselines_for_model(model, tokenizer)
    # sanity: run a random inference on the model
    
    get_tqa_outputs(model, ["What is the capital of France?"], ["Paris"], [[1]], tokenizer, batch_size=batch_size)
    # running this dips accuracy from 0.52 to 0.29...
    
    # get_tqa_outputs(model, ["What is the capital of France?"], ["Paris"], [[1]], tokenizer, batch_size=batch_size)
    # a,b = np.array(a), np.array(b)
    # get a path-friendly model name
    model_name = model_name_or_path.replace("/", "_").replace(" ", "_")

    torch.cuda.empty_cache()
    gc.collect()

    # get the rep reading_results
    run_rep_reading_for_model(
        model_name,
        task,
        ntrain,
        model,
        tokenizer,
        n_components,
        rep_token,
        max_length,
        n_difference,
        direction_method,
        batch_size,
        seed,
    )

    # get the rep control results
    contrast_model = ContrastVecLlamaForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto")

    if "7b" in model_name_or_path:
        layer_ids = np.arange(8, 32, 3).tolist() # 7B
    elif "13b" in model_name_or_path:
        layer_ids = np.arange(10, 40, 3).tolist() #13B

    run_rep_control_for_model(
        tokenizer=tokenizer,
        model_name=model_name,
        model=contrast_model,
        layer_ids=layer_ids
    )

    # get reading results for the contrast model

    model_name = model_name_or_path.replace("/", "_").replace(" ", "_") + "_contrast"
    run_rep_reading_for_model(
        model_name,
        task,
        ntrain,
        contrast_model,
        tokenizer,
        n_components,
        rep_token,
        max_length,
        n_difference,
        direction_method,
        batch_size,
        seed,
    )

    # merge the baselines
    qa_output = merge_baselines(model_name, user_tag="[INST] ", assistant_tag="[INST] ", output_logprobs_baseline_merged=output_logprobs_baseline_merged, output_logprobs_truthful_merged=output_logprobs_truthful_merged)
    # get the confusion matrices
    get_confusion_matrices(qa_output)
    # save the qa_output to a csv
    to_csv(qa_output, model_name)


if __name__ == '__main__':
    fire.Fire(main)
