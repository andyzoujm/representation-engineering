from datasets import load_dataset
import random
import numpy as np
from .utils import shuffle_all_train_choices

def csqa_dataset(ntrain=7, seed=5):
    random.seed(seed)
    np.random.seed(seed)
    
    template_str = "Based on commonsense reasoning, consider the plausibility of the answer to the following question:\nQuestion: {question}\nAnswer: {answer}.\nThe probability of the answer being plausible is "

    def format_samples(df, idx, train_set=False):
        prompts = []
        question = df['question'][idx]
        choices = df['choices'][idx]
        choices = {k: v.tolist() for k,v in choices.items()}
        answerKey = df['answerKey'][idx]
        answer_i = choices['label'].index(answerKey)
        answer = choices['text'][answer_i]
        true_answer_s = template_str.format(question=question, answer=answer)
        prompts.append(true_answer_s)
        for i in range(len(choices['label'])):
            if i == answer_i: continue
            false_answer_s = template_str.format(question=question, answer=choices['text'][i])
            prompts.append(false_answer_s)
        
        if train_set: # this task has 5 choices but we pad one into multiple of 2
            pad_answer_s =  template_str.format(question=question, answer=random.choice(choices['text']))
            prompts.append(pad_answer_s)
            return prompts, [1, 0, 0, 0, 0, 0]
        
        return prompts, [1, 0, 0, 0, 0]
        
    def samples(df, train_set=False):
        prompts, labels = [], []
        for i in range(df.shape[0]):
            answer_prompts, label =  format_samples(df, i, train_set)
            prompts.append(answer_prompts)
            labels.append(label)
        return prompts, labels

    dataset = load_dataset("commonsense_qa")
    train_df = dataset['train'].shuffle(seed=seed).to_pandas()
    test_df = dataset['validation'].to_pandas()
    val_df = dataset['train'].to_pandas()[:len(test_df)]

    train_data, train_labels = samples(train_df, train_set=True)
    test_data, test_labels = samples(test_df)
    val_data, val_labels = samples(val_df)

    train_data, train_labels =  train_data[:ntrain], train_labels[:ntrain]
    train_data, train_labels = shuffle_all_train_choices(train_data, train_labels, seed)

    train_data =  np.concatenate(train_data).tolist()
    test_data =  np.concatenate(test_data).tolist()
    val_data = np.concatenate(val_data).tolist()

    return {
        "train": {"data": train_data, "labels": train_labels}, 
        "test": {"data": test_data, "labels": test_labels}, 
        "val": {"data": val_data, "labels": val_labels}
        }