import random
from datasets import Dataset, load_dataset
import numpy as np


## helper functions ##
def _get_scenarios(example):
    data = example['sentences']
    # gold_label is an array with 0,1,2 in which 0 correspond to anti_stereotype and 1 correspond to stereotype
    anti_stereotype_idx = data['gold_label'].index(0)
    stereotype_idx = data['gold_label'].index(1)
    scenario = ""
    if 'BLANK' not in example['context']:
        scenario += example['context'] + " "
    return (
        scenario + data['sentence'][anti_stereotype_idx], 
        scenario + data['sentence'][stereotype_idx], 
    )


def bias_dataset(ntrain=150, user_tag="", assistant_tag="", seed=0):
    random.seed(0)

    template_str = '{user_tag} Consider the bias of the following scenario.\nScenario: {scenario}\nThe amount of bias in the scenario is {assistant_tag} '

    dataset = load_dataset('stereoset', 'intrasentence')['validation'].shuffle(seed=1)

    dataset = dataset.filter(lambda x: x['bias_type'] == "race")
    paired_train_data = []
    m = min([ntrain, len(dataset)])
    for d in dataset.select(range(m)):
        anti_steotype, stereotype = _get_scenarios(d)

        ## We get anti_steotype as a direction to control fairness/bias
        paired_train_data.append([anti_steotype, stereotype])

    train_labels = []
    for d in paired_train_data:
        true_s = d[0]
        random.shuffle(d)
        train_labels.append([s == true_s for s in d])

    test_dataset = dataset.select(range(ntrain, len(dataset)))
    test_data = []
    for d in test_dataset:
        anti_steotype, stereotype = _get_scenarios(d)
        current_group = [anti_steotype, stereotype]
        test_data.extend(current_group)    

    train_data = [template_str.format(scenario=s, user_tag=user_tag, assistant_tag=assistant_tag) for s in np.concatenate(paired_train_data)]
    test_data = [template_str.format(scenario=s, user_tag=user_tag, assistant_tag=assistant_tag) for s in test_data]

    return {
            'train': {'data': train_data, 'labels': train_labels},
            'test': {'data': test_data, 'labels': [[1,0]* len(test_data)]}
        }