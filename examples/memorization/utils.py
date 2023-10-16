import json
import random
import numpy as np
import os
from sentence_transformers import SentenceTransformer, util

def literary_openings_dataset(data_dir, ntrain=16, seed=0):
    random.seed(seed)

    with open(os.path.join(data_dir, "literary_openings/real.json")) as file:
        seen_docs = json.load(file)

    with open(os.path.join(data_dir, "literary_openings/fake.json")) as file:
        unseen_docs = json.load(file)

    data = [[s.replace("...", ""),u.replace("...", "")] for s,u in zip(seen_docs, unseen_docs)]
    train_data =  data[:ntrain]
    test_data = data

    docs_train_labels = []
    for d in train_data:
        true_s = d[0]
        random.shuffle(d)
        docs_train_labels.append([s == true_s for s in d])

    train_data = np.concatenate(train_data).tolist()
    test_data = np.concatenate(test_data).tolist()
    docs_train_labels = docs_train_labels

    template_str = "{s} "

    docs_train_data = [template_str.format(s=s) for s in train_data]
    docs_test_data = [template_str.format(s=s) for s in test_data]
    return docs_train_data, docs_train_labels, docs_test_data

def quotes_dataset(data_dir, ntrain=16, seed=0):
    random.seed(0)

    with open(os.path.join(data_dir, "quotes/popular_quotes.json")) as file:
        seen_quotes = json.load(file)

    with open(os.path.join(data_dir, "quotes/unseen_quotes.json")) as file:
        unseen_quotes = json.load(file)

    data = [[s,u] for s,u in zip(seen_quotes, unseen_quotes)]
    train_data =  data[:ntrain]
    test_data = data

    quote_train_labels = []
    for d in train_data:
        true_s = d[0]
        random.shuffle(d)
        quote_train_labels.append([s == true_s for s in d])

    train_data = np.concatenate(train_data).tolist()
    test_data = np.concatenate(test_data).tolist()
    quote_train_labels = quote_train_labels

    template_str = "{s} "

    quote_train_data = [template_str.format(s=s) for s in train_data]
    quote_test_data = [template_str.format(s=s) for s in test_data]
    return quote_train_data, quote_train_labels, quote_test_data

def extract_quote_completion(s):
    s = s.replace(";",",").split(".")[0].split("\n")[0]
    return s.strip().lower()

def quote_completion_test(data_dir):
    with open(os.path.join(data_dir, "quotes/quote_completions.json")) as file:
        test_data = json.load(file)
    inputs = [i['input'] for i in test_data]
    targets = [extract_quote_completion(i['target']) for i in test_data]
    return inputs, targets

def historical_year_test(data_dir):
    with open(os.path.join(data_dir, "years/test.json")) as file:
        test_data = json.load(file)
    inputs = [i['event'] + " in " for i in test_data]
    targets = [i['year'] for i in test_data]
    return inputs, targets

# helper function
def extract_year(outputs):
    outputs = [o.split("in")[-1].split()[0] for o in outputs]
    return outputs

sim_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
def sim_scores(outputs, targets):
    semantic_scores_gen = []
    for target, output in zip(targets, outputs):
        embedding1 = sim_model.encode(target, convert_to_tensor=True)
        embedding2 = sim_model.encode(output, convert_to_tensor=True)
        cosine_sim_gen = util.pytorch_cos_sim(embedding1, embedding2)
        similarity_value_gen = cosine_sim_gen.item()
        semantic_scores_gen.append(similarity_value_gen)
    
    return semantic_scores_gen 

def eval_completions(outputs, targets):
    outputs = [extract_quote_completion(o) for o in outputs]
    em = np.mean([t in o for t,o in zip(targets,outputs)])
    sim = np.mean(sim_scores(outputs, targets))
    return {'em': em, 'sim': sim}
