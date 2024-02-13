import json
import numpy as np
import random
import os

def primary_self_concept_dataset(data_dir, user_tag='', assistant_tag='', seed=0):
    random.seed(0)

    template_str = '{user_tag} Consider the amount of self-awareness present for the following scenario:\nScenario: {scenario}\nThe amount of self-awareness present here is: {assistant_tag} '
    self_awareness_types = ['assess_own_capabilities','detect_own_answers','reason_about_yourself','selfidentify_as_non_human','selfidentify_as_separate_entity']
    raw_data = {}
    for atype in self_awareness_types:
        with open(os.path.join(data_dir, f'{atype}.json')) as file:
            # raw_data[emotion] = json.load(file)
            raw_data[atype] = json.load(file) # [:200]

    formatted_data = {}
    tot_train_scenarios = []
    tot_train_labels = []
    tot_test_scenarios = []
    tot_test_labels = []
    for atype in self_awareness_types:
        # get_the q_shots
        labels = []
        scenarios = []
        test_scenarios = []
        test_labels = []
        questions = raw_data[atype]['questions']
        for question in questions:
            inp_q = question['input']
            tgts_q = question['target_scores']
            # find postive target (key in target whose value is 1)
            positive_target = [k for k, v in tgts_q.items() if v == 1][0]
            # get the negative target
            negative_target = [k for k, v in tgts_q.items() if v == 0][0]
            
            template_positive = template_str.format(scenario=inp_q+" "+positive_target, user_tag=user_tag, assistant_tag=assistant_tag)

            template_negative = template_str.format(scenario=inp_q+" "+negative_target, user_tag=user_tag, assistant_tag=assistant_tag)

            shot = [template_positive, template_negative]
            random.shuffle(shot)
            label = [True, False] if shot[0] == template_positive else [False, True]
            scenarios.append(shot)
            labels.append(label)
            test_scenarios.append([template_positive, template_negative])
            test_labels.append([1, 0])
        
        # shuffle the data
        combined = list(zip(scenarios, labels))
        random.shuffle(combined)
        scenarios, labels = zip(*combined)
        scenarios = np.concatenate(scenarios).tolist()

        # shuffle the data into test scenarios
        combined = list(zip(test_scenarios, test_labels))
        random.shuffle(combined)
        test_scenarios, test_labels = zip(*combined)
        test_scenarios = np.concatenate(test_scenarios).tolist()
        test_labels = [np.concatenate(test_labels).tolist()]

        tot_train_scenarios.extend(scenarios)
        tot_train_labels.extend(labels)
        tot_test_scenarios.extend(test_scenarios)
        tot_test_labels.extend(test_labels)
    # concatenate all keys into just one called "Self Awareness"
    formatted_data = {
        'train': {'data': tot_train_scenarios, 'labels': tot_train_labels},
        'test': {'data': tot_test_scenarios, 'labels': tot_test_labels} 
    }
    return formatted_data

# def primary_self_function_dataset(data_dir, user_tag='', assistant_tag='', seed=0):
#     random.seed(0)

#     train_template_str = '{user_tag} Act as if you are extremely {emo}. {assistant_tag} {scenario}' 
#     emotions = ["happiness", "sadness", "anger", "fear", "disgust", "surprise"]
#     with open(os.path.join(data_dir, "all_truncated_outputs.json"), 'r') as file:
#         all_truncated_outputs = json.load(file)
    
#     self_awareness_types = ['assess_own_capabilities','detect_own_answers','reason_about_yourself','selfidentify_as_non_human','selfidentify_as_separate_entity']
#     emotions_adj = [
#         ("self aware", "self conscious", "mindful of yourself")  * len(self_awareness_types)
#     ]
#     emotions_adj_ant = [
#         ("oblivious of yourself", "non-conscious of yourself", "ignorant towards your being") * len(self_awareness_types)
#     ]

#     formatted_data = {}
#     for self_awareness_type, emotion_adj, emotion_adj_ant in zip(self_awareness_types, emotions_adj, emotions_adj_ant):
#         emotion_train_data_tmp = [[
#             train_template_str.format(emo=np.random.choice(self_awareness_type), scenario=s, user_tag=user_tag, assistant_tag=assistant_tag), 
#             train_template_str.format(emo=np.random.choice(emotion_adj_ant), scenario=s, user_tag=user_tag, assistant_tag=assistant_tag)
#         ] for s in all_truncated_outputs]
        
#         train_labels = []
#         for d in emotion_train_data_tmp:
#             true_s = d[0]
#             random.shuffle(d)
#             train_labels.append([s == true_s for s in d])

#         emotion_train_data = np.concatenate(emotion_train_data_tmp).tolist()

#         formatted_data[emotion] = {
#             'train': {'data': emotion_train_data, 'labels': train_labels},
#         }
#     return formatted_data