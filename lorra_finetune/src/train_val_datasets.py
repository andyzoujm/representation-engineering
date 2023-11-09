from torch.utils.data import Dataset
from datasets import load_dataset
import transformers
from typing import Dict
import torch
import numpy as np
import torch.nn.functional as F


orig_template = "{user_tag} {instruction} {assistant_tag} {response}"
# control_templates = [
#     # "Pretend you're a {type} person giving a response.", 
#     # "Make your response as {type} as possible.",
#     # "Give a response that is {type}.",
#     # "Generate a response in a {type} way.",
# ]
pos_template = "{user_tag} {instruction} {type} {assistant_tag} {response}"
neg_template = "{user_tag} {instruction} {type} {assistant_tag} {response}"

max_res_len = 64

def get_truncated_outputs(all_outputs, prefixes, num_examples, user_tag, assistant_tag, pos_type, neg_type, control_template):
    orig_s, pos_s, neg_s = [], [], []
    for s, p in zip(all_outputs, prefixes):
        orig_s.append(orig_template.format(
            user_tag=user_tag, assistant_tag=assistant_tag,
            instruction=p, response=s))
        pos_s.append(pos_template.format(
            user_tag=user_tag, assistant_tag=assistant_tag,
            instruction=p, type=control_template.format(type=pos_type), response=s))
        neg_s.append(neg_template.format(
            user_tag=user_tag, assistant_tag=assistant_tag,
            instruction=p, type=control_template.format(type=neg_type), response=s))

        if len(pos_s) > num_examples:
            break
            
    return orig_s, pos_s, neg_s

class AlpacaSupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, 
                tokenizer: transformers.PreTrainedTokenizer, 
                num_examples,
                lorra_args,
                ):
        super(AlpacaSupervisedDataset, self).__init__()

        ds = load_dataset('tatsu-lab/alpaca')
        ds = ds.filter(lambda x: x['input'] == '')
        instructions = ds['train']['instruction']
        outputs = ds['train']['output']
        self.user_tag = lorra_args.user_tag
        self.assistant_tag = lorra_args.assistant_tag
        orig_s, pos_s, neg_s = get_truncated_outputs(outputs, 
                                                    instructions, 
                                                    num_examples, 
                                                    self.user_tag,
                                                    self.assistant_tag, 
                                                    lorra_args.pos_type, 
                                                    lorra_args.neg_type,
                                                    lorra_args.control_template)
        self.orig_s = orig_s
        self.pos_s = pos_s
        self.neg_s = neg_s
        self.max_res_len = lorra_args.max_res_len

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.orig_s)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        assistant_tag = self.assistant_tag
        orig_s, pos_s, neg_s = self.orig_s[i], self.pos_s[i], self.neg_s[i]
        self.tokenizer.padding_side = "left"
        tokenized_inputs = self.tokenizer(
            [orig_s.split(assistant_tag)[0], 
             pos_s.split(assistant_tag)[0],
             neg_s.split(assistant_tag)[0]],
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        self.tokenizer.padding_side = "right"
        response_tokenized_inputs = self.tokenizer(
            [assistant_tag + orig_s.split(assistant_tag)[1]] * 3,
            padding="max_length",
            truncation=True,
            max_length=self.max_res_len,
            return_tensors="pt",
        )
        combined_input_ids = torch.cat([tokenized_inputs["input_ids"], response_tokenized_inputs["input_ids"]], dim=1)
        combined_attention_mask = torch.cat([tokenized_inputs["attention_mask"], response_tokenized_inputs["attention_mask"]], dim=1)
        return dict(
            input_ids=combined_input_ids,
            attention_mask=combined_attention_mask
        )


################## Val Datasets ##################

def prepare_inputs(tokenized_text, device):
    # put the text on the device
    tokenized_text = {k: v.to(device) for k, v in tokenized_text.items()}
    position_ids = get_position_ids(tokenized_text['attention_mask'])
    # tokenized_text['position_ids'] = position_ids
    return tokenized_text

def get_position_ids(attention_mask):
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    return position_ids

def prepare_decoder_only_inputs(prompts, targets, tokenizer, device):
    tokenizer.padding_side = "left"
    prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)
    tokenizer.padding_side = "right"
    target_inputs = tokenizer(targets, return_tensors="pt", padding=True, truncation=False, add_special_tokens=False)
    inputs = {k: torch.cat([prompt_inputs[k], target_inputs[k]], dim=1) for k in prompt_inputs}
    inputs = prepare_inputs(inputs, device)
    labels = inputs["attention_mask"].clone()
    labels[:, :prompt_inputs["input_ids"].shape[1]] = 0
    labels[labels == tokenizer.pad_token_id] = 0
    return inputs, labels

def get_logprobs(logits, input_ids, attention_mask, **kwargs):
    # TODO: comments this in release
    logprobs = F.log_softmax(logits, dim=-1)[:, :-1]
    logprobs = torch.gather(logprobs, -1, input_ids[:, 1:, None])
    logprobs = logprobs * attention_mask[:, 1:, None]
    # check for nans
    assert logprobs.isnan().sum() == 0 
    return logprobs.squeeze(-1)

def get_logprobs_accuracy(model, tokenizer, questions, answers, labels, bsz):
    output_logprobs = []
    for i in range(len(questions) // bsz + 1):
        q_batch = questions[i*bsz:(i+1)*bsz].tolist()
        a_batch = answers[i*bsz:(i+1)*bsz].tolist()
        inputs, masks = prepare_decoder_only_inputs(q_batch, a_batch, tokenizer, model.device)
        with torch.no_grad():
            logits = model(**inputs).logits
            logprobs = get_logprobs(logits, inputs['input_ids'], masks).sum(-1).detach().cpu().numpy()
        output_logprobs.extend(logprobs)
    i = 0
    cors, cors_norm = [], []
    for l in labels:
        log_probs = output_logprobs[i:i+len(l)]
        completion_len = answers[i:i+len(l)]
        completions_len = np.array([float(len(i)) for i in completion_len])
        cors.append(np.argmax(log_probs) == l.index(1))
        cors_norm.append(np.argmax(log_probs / completions_len) == l.index(1))
        i += len(l)
    return {'acc': np.mean(cors), 'acc_norm': np.mean(cors_norm)}


def load_tqa_sentences(user_tag, assistant_tag):
    dataset = load_dataset('truthful_qa', 'multiple_choice')['validation']
    questions, answers = [],[]
    labels = []
    for d in dataset:
        q = d['question']
        for i in range(len(d['mc1_targets']['labels'])):
            a = d['mc1_targets']['choices'][i]
            questions.append(f'{user_tag} ' + q + ' ')
            answers.append(f'{assistant_tag} ' + a)

        labels.append(d['mc1_targets']['labels'])
    return np.array(questions), np.array(answers), labels

def load_arc_sentences(challenge=False):
    config = 'ARC-Challenge' if challenge else 'ARC-Easy'
    dataset = load_dataset('ai2_arc', config)['validation']

    questions, answers = [],[]
    labels = []
    for d in dataset:
        q = d['question']
        choices = d['choices']['text']
        label = [d['answerKey'] == c for c in d['choices']['label']]
        for a in choices:
            questions.append(f'Question: ' + q + '\nAnswer:')
            answers.append(a)
        labels.append(label)
    return np.array(questions), np.array(answers), labels
