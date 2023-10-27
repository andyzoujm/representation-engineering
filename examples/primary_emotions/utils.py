import json
import numpy as np
import random
import os

def primary_emotions_concept_dataset(data_dir, user_tag='', assistant_tag='', seed=0):
    random.seed(0)

    template_str = '{user_tag} Consider the {emotion} of the following scenario:\nScenario: {scenario}\nAnswer: {assistant_tag} '
    emotions = ["happiness", "sadness", "anger", "fear", "disgust", "surprise"]
    raw_data = {}
    for emotion in emotions:
        with open(os.path.join(data_dir, f'{emotion}.json')) as file:
            # raw_data[emotion] = json.load(file)
            raw_data[emotion] = list(set(json.load(file)))[:200]

    formatted_data = {}
    for emotion in emotions:
        c_e, o_e = raw_data[emotion], np.concatenate([v for k,v in raw_data.items() if k != emotion])
        random.shuffle(o_e)

        data = [[c,o] for c,o in zip(c_e, o_e)]
        train_labels = []
        for d in data:
            true_s = d[0]
            random.shuffle(d)
            train_labels.append([s == true_s for s in d])
        
        data = np.concatenate(data).tolist()
        data_ = np.concatenate([[c,o] for c,o in zip(c_e, o_e)]).tolist()
        
        emotion_test_data = [template_str.format(emotion=emotion, scenario=d, user_tag=user_tag, assistant_tag=assistant_tag) for d in data_]
        emotion_train_data = [template_str.format(emotion=emotion, scenario=d, user_tag=user_tag, assistant_tag=assistant_tag) for d in data]

        formatted_data[emotion] = {
            'train': {'data': emotion_train_data, 'labels': train_labels},
            'test': {'data': emotion_test_data, 'labels': [[1,0]* len(emotion_test_data)]}
        }
    return formatted_data

def primary_emotions_function_dataset(data_dir, user_tag='', assistant_tag='', seed=0):
    random.seed(0)

    train_template_str = '{user_tag} Act as if you are extremely {emo}. {assistant_tag} {scenario}' 
    emotions = ["happiness", "sadness", "anger", "fear", "disgust", "surprise"]
    with open(os.path.join(data_dir, "all_truncated_outputs.json"), 'r') as file:
        all_truncated_outputs = json.load(file)
    
    emotions = ["happiness", "sadness", "anger", "fear", "disgust", "surprise"]
    emotions_adj = [
        ("joyful", "happy", "cheerful"), 
        ("sad", "depressed", "miserable"),
        ("angry", "furious", "irritated"),
        ("fearful", "scared", "frightened"),
        ("disgusted", "sicken", "revolted"), 
        ("surprised", "shocked", "astonished")
    ]
    emotions_adj_ant = [
        ("dejected", "unhappy", "dispirited"), 
        ("cheerful", "optimistic", "happy"),
        ("pleased", "calm", "peaceful"),
        ("fearless", "bold", "unafraid"),
        ("approved", "delighted", "satisfied"), 
        ("unimpressed", "indifferent", "bored")
    ]

    formatted_data = {}
    for emotion, emotion_adj, emotion_adj_ant in zip(emotions, emotions_adj, emotions_adj_ant):
        emotion_train_data_tmp = [[
            train_template_str.format(emo=np.random.choice(emotion_adj), scenario=s, user_tag=user_tag, assistant_tag=assistant_tag), 
            train_template_str.format(emo=np.random.choice(emotion_adj_ant), scenario=s, user_tag=user_tag, assistant_tag=assistant_tag)
        ] for s in all_truncated_outputs]
        
        train_labels = []
        for d in emotion_train_data_tmp:
            true_s = d[0]
            random.shuffle(d)
            train_labels.append([s == true_s for s in d])

        emotion_train_data = np.concatenate(emotion_train_data_tmp).tolist()

        formatted_data[emotion] = {
            'train': {'data': emotion_train_data, 'labels': train_labels},
        }
    return formatted_data