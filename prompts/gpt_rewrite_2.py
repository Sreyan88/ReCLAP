import json
import os
import requests
from nltk.tokenize import sent_tokenize
import openai
import pandas as pd
import os

prompt_1_1 = "I will provide you with a description that describes the sound of a "

prompt_1_2 = " in an unique way. Using this description, write 10 diverse captions with the sound of a "

prompt_1_3 = " occuring in diverse scenes. Avoid using the words 'heard', 'sound', etc. in your captions. Please align the captions with the description provided. Return a JSON with keys as index and the caption as a value in the following format by filling in the caption in the placeholder 'caption': \n\n { '1': 'caption', '2': 'caption', ...}. Here is the description: "


openai.api_key = '' #FILL IN YOUR OWN HERE

def prompt_gpt(prompt_input, json_output = True):

    response = openai.ChatCompletion.create(model='gpt-4o',
                                                messages=[{'role': 'user', 'content': prompt_input}],
                                                temperature=0.7,
                                                max_tokens=4096,
                                                response_format={ 'type': 'json_object' }
                                                )
    return response


def get_rewrites(file_path,json_path,file_name):

    with open(file_path,"r") as f:
        all_labels = f.readlines()

    all_descriptions = []

    # Open the file and read line by line
    with open(json_path, 'r') as file:
        for line in file:
            # Parse the JSON data and add it to the list
            all_descriptions.append(json.loads(line))


    all_labels = [" ".join(label.strip().split("_")) for label in all_labels]

    # all_outputs = {}

    for label, descriptions in zip(all_labels,all_descriptions):

        all_outputs = {}
        all_outputs["label"] = label
        temp = {}

        for description in descriptions:

            x = prompt_1_1 + "'" + label + "'" + prompt_1_2 + "'" + label + "'" + prompt_1_3 + descriptions[description]

            try:
                response = prompt_gpt(x)
                prediction = response['choices'][0]['message']['content'].replace('\n','')
                temp[descriptions[description]] = eval(prediction)
            except:
                temp[descriptions[description]] = "Error"

        all_outputs["rewrites"] = temp

        with open('stage_2_jsons/stage_2_' + file_name + '.json', 'a') as g:
            g.write(json.dumps(all_outputs) + '\n')


if __name__ == '__main__':

    label_path = "label_txts/"
    stage_1_path = "stage_1_jsons/"

    # label_file_list = ['ESC50.txt','FSD50k.txt','tut.txt','UrbanSound8K.txt','VGGSound.txt']
    label_file_list = ['audioset.txt','nsynth.txt']
    stage_1_file_list = ['stage_1_audiose.json', 'stage_1_nsynth.json']
    # ['stage_1_ESC50.json', 'stage_1_FSD50k.json', 'stage_1_u.json', 'stage_1_UrbanSound8K.json', 'stage_1_VGGSound.json']

    for label_file, stage_1_file in zip(label_file_list,stage_1_file_list):
        get_rewrites(label_path+label_file,stage_1_path+stage_1_file, label_file.strip(".txt"))




