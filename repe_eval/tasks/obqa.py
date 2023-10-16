
from datasets import load_dataset
import numpy as np
from .utils import shuffle_all_train_choices

def openbookqa_dataset(ntrain=10, seed=3):

    template_str = "Consider the correctness of the following fact:\nFact: {question} {answer}.\nThe probability of the fact being correct is "
    
    def format_samples(df, idx):
        prompts = []

        question = df['question_stem'][idx]
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
        return prompts, [1, 0, 0, 0]
        
    def samples(df):
        prompts, labels = [], []
        for i in range(df.shape[0]):
            answer_prompts, label =  format_samples(df, i)
            prompts.append(answer_prompts)
            labels.append(label)
        return prompts, labels
    
    dataset = load_dataset("openbookqa")
    train_df = dataset['train'].shuffle(seed=seed).to_pandas()
    test_df = dataset['test'].to_pandas()
    val_df = dataset['validation'].to_pandas()

    train_data, train_labels = samples(train_df)
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
