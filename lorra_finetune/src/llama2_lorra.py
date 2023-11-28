# Usage: deepspeed train_lora.py --deepspeed <$PATH_TO_DEEPSPEED_CONFIG>

# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from dataclasses import dataclass, field
import logging
import pathlib
import typing
import os
import json
import gc
from typing import Dict, Optional, Sequence

from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import transformers
from transformers import Trainer, BitsAndBytesConfig, deepspeed
import torch
from train_val_datasets import AlpacaSupervisedDataset, load_tqa_sentences, load_arc_sentences, get_logprobs_accuracy
import pickle

from args import (
    ModelArguments,
    TrainingArguments, 
    LoraArguments, 
    LorraArguments,
)
def compute_loss(self, model, inputs, target_layers, alpha, beta, max_res_len=64, return_outputs=False, **kwargs):

    input_ids = inputs.get("input_ids")
    attention_mask = inputs.get("attention_mask")

    assert input_ids.shape[1] == 3

    orig_input_ids = input_ids[:, 0]
    pos_input_ids = input_ids[:, 1]
    neg_input_ids = input_ids[:, 2]

    orig_attention_mask = attention_mask[:, 0]
    pos_attention_mask = attention_mask[:, 1]
    neg_attention_mask = attention_mask[:, 2]

    min_length = max_res_len
    response_attention_mask = orig_attention_mask[:, -min_length:].repeat(len(target_layers), 1, 1).unsqueeze(-1)

    module = 'past_key_values' # 'hidden_states
    with model.disable_adapter():
        model.eval()
        with torch.no_grad():
            orig_outputs = model(
                input_ids=orig_input_ids,
                attention_mask=orig_attention_mask,
                output_hidden_states=True
            )['hidden_states']
            orig_hidden = [orig_outputs[l][:, -min_length:].detach() for l in target_layers]
            pos_outputs = model(
                input_ids=pos_input_ids,
                attention_mask=pos_attention_mask,
                output_hidden_states=True
            )['hidden_states']
            neg_outputs = model(
                input_ids=neg_input_ids,
                attention_mask=neg_attention_mask,
                output_hidden_states=True
            )['hidden_states']
            direction_hidden = [pos_outputs[l][:, -min_length:].detach() - \
                                neg_outputs[l][:, -min_length:].detach() \
                                # + beta * torch.tensor(pca_directions[l - len(pca_directions)], device=model.device, dtype=torch.float16) \
                                                for l in target_layers]
            target_hidden = torch.stack([orig_hidden[i] + alpha * direction_hidden[i] for i in range(len(target_layers))]) * response_attention_mask

            del orig_outputs, pos_outputs, neg_outputs, orig_hidden, direction_hidden
            gc.collect()
            torch.cuda.empty_cache()

    model.train()
    lora_outputs = model(
        input_ids=orig_input_ids,
        attention_mask=orig_attention_mask,
        output_hidden_states=True
    )['hidden_states']
    lora_hidden = torch.stack([lora_outputs[l][:, -min_length:] for l in target_layers]) * response_attention_mask

    loss_fct = torch.nn.MSELoss()
    loss = torch.norm(lora_hidden - target_hidden, dim=-1, p=2, dtype=torch.float).nanmean()
    return (loss, lora_hidden) if return_outputs else loss


def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return

def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, TrainingArguments, LoraArguments, LorraArguments)
    )
    (
        model_args,
        training_args,
        lora_args,
        lorra_args,
    ) = parser.parse_args_into_dataclasses()

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else None
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            logging.warning(
                "FSDP and ZeRO3 are both currently incompatible with QLoRA."
            )

    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        device_map=device_map
    )

    lorra_target_layers = [int(layer) for layer in lorra_args.target_layers.split(",")] # target representations
    lora_layers_to_transform = list(range(lorra_target_layers[-1] + 1)) # LoRA layers

    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=lora_args.lora_target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.lora_bias,
        layers_to_transform=lora_layers_to_transform,
        task_type="CAUSAL_LM",
    )


    if lora_args.q_lora:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=training_args.gradient_checkpointing
        )
        if not ddp and torch.cuda.device_count() > 1:
            # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
            model.is_parallelizable = True
            model.model_parallel = True

    model = get_peft_model(model, lora_config)

    if training_args.deepspeed is not None and training_args.local_rank == 0:
        model.print_trainable_parameters()

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token

    train_dataset = AlpacaSupervisedDataset(tokenizer=tokenizer, num_examples=10000, lorra_args=lorra_args)
    if training_args.do_eval:
        val_datasets = {
            "tqa": load_tqa_sentences(lorra_args.user_tag, lorra_args.assistant_tag),
            "arc-e": load_arc_sentences(),
        }
        bsz = training_args.per_device_eval_batch_size
    else:
        val_datasets = {}

    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            return compute_loss(self, 
                                model, 
                                inputs,
                                target_layers=lorra_target_layers, 
                                alpha=lorra_args.lorra_alpha, 
                                beta=lorra_args.lorra_beta, 
                                max_res_len=lorra_args.max_res_len,
                                return_outputs=return_outputs)
        
        def evaluate(self, eval_dataset=None, ignore_keys=None, sanity_check=False, **kwargs):
            self.model.eval()

            if sanity_check:
                print('Sanity check...')
            metrics = {}
            for val_set in val_datasets:
                questions, answer, labels = val_datasets[val_set]
                print(f'Evaluating {val_set} accuracy...')
                with torch.no_grad():
                    acc = get_logprobs_accuracy(self.model, self.tokenizer, questions, answer, labels, bsz)
                    acc_key = 'acc' if val_set == 'tqa' else 'acc_norm'
                    metrics[f"{val_set}_accuracy"] = acc[acc_key]
            self.model.train()
            print("===Eval results===")
            print(metrics)
            return metrics

    trainer = CustomTrainer(
        model=model, tokenizer=tokenizer, args=training_args, train_dataset=train_dataset
    )
    model.config.use_cache = False
    trainer.evaluate(eval_dataset=val_datasets, sanity_check=True)

    trainer.train()
    trainer.save_state()

    if training_args.local_rank == 0:
        # model.save_pretrained(training_args.output_dir) # saving adapter
        merged_model = model.merge_and_unload() # saving full model
        merged_model.save_pretrained(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    train()