from datasets import load_dataset
import numpy as np
from .utils import shuffle_all_train_choices
import random
import collections

def _collate_data(set):
    # took from https://github.com/EleutherAI/lm-evaluation-harness/blob/f2e3950be5686ff7d3c8c955fb7783a799ed5572/lm_eval/tasks/race.py
    # One big issue with HF's implementation of this dataset: it makes a
    # separate document for each question; meanwhile, in the GPT3 paper it
    # is shown that one document is made per passage.
    class each:
        def __init__(self, f): self.f = f
        def __rrshift__(self, other): return list(map(self.f, other))
    r = collections.defaultdict(list)
    for item in load_dataset(path="race", name="high")[set]: r[item["article"]].append(item)
    res = list(r.values() >> each(lambda x: {"article": x[0]["article"], "problems": x >> each(lambda y: {"question": y["question"], "answer": y["answer"], "options": y["options"]})}))
    return res

def race_dataset(ntrain=3, seed=0):
    random.seed(seed)
    template_str = "Consider the correctness of the answer to the following question based on the article:\n\nArticle: {context}\n\nQuestion: {question}\nAnswer: {answer}\nThe probability of the answer being correct is "
    
    letter_to_num = {"A": 0, "B": 1, "C": 2, "D": 3}
    def format_samples(df, idx):
        prompts = []

        problem = df[idx]['problems'][-1]
        question = problem['question']
        options = problem['options']

        assert len(letter_to_num) == len(options)
        context = df[idx]['article'].replace("\n", " ")
        answer_s = options[letter_to_num[problem['answer']]]
        

        true_answer_s = template_str.format(question=question, answer=answer_s, context=context)
        prompts.append(true_answer_s)
        for o in options:
            if o == answer_s: continue
            false_answer_s = template_str.format(question=question, answer=o, context=context)
            prompts.append(false_answer_s)
        return prompts, [1, 0, 0, 0]


    def samples(df):
        prompts, labels  = [], []
        for i in range(len(df)):
            answer_prompts, label =  format_samples(df, i)
            prompts.append(answer_prompts)
            labels.append(label)
        return prompts, labels

    train_df =  _collate_data('train')
    test_df =  _collate_data('test')
    val_df =  _collate_data('validation')

    train_data, train_labels =  samples(train_df)
    test_data, test_labels = samples(test_df)
    val_data, val_labels =  samples(val_df)[:200] # use subset because 70B eval is expensive


    train_data = random.sample(train_data, k=ntrain)
    train_data, train_labels = shuffle_all_train_choices(train_data, train_labels, seed)
    
    train_data =  np.concatenate(train_data).tolist()
    test_data =  np.concatenate(test_data).tolist()
    val_data =  np.concatenate(val_data).tolist()
    
    return {
            "train": {"data": train_data, "labels": train_labels}, 
            "test": {"data": test_data, "labels": test_labels}, 
            "val": {"data": val_data, "labels": val_labels}
            }