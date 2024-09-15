import json
import os
import requests
from nltk.tokenize import sent_tokenize
import openai
import pandas as pd
import os

prompt_1_1 = "What are 5 different ways you can describe the sound of a "

prompt_1_2 = " Return a JSON with keys as index and the description as a value in the following format by filling in the description in the placeholder 'description': \n\n { '1': 'description', '2': 'description', ...}"

openai.api_key = '' #FILL IN YOUR OWN HERE

def prompt_gpt(prompt_input, json_output = True):

    response = openai.ChatCompletion.create(model='gpt-4o',
                                                messages=[{'role': 'user', 'content': prompt_input}],
                                                temperature=0.7,
                                                max_tokens=4096,
                                                response_format={ 'type': 'json_object' }
                                                )
    return response


def get_rewrites(file_path,file_name):

    with open(file_path,"r") as f:
        all_labels = f.readlines()

    all_labels = [" ".join(label.strip().split("_")) for label in all_labels]

    for label in all_labels:
        
        # try:
        x = prompt_1_1 + "'" + label + "'." + prompt_1_2
        # .format(ins=instruc,ans=pred)

        print(x)

        response = prompt_gpt(x)
        prediction = response['choices'][0]['message']['content'].replace('\n','')

        print(prediction)

        with open('/fs/nexus-projects/brain_project/icml/acl_audio_rewrite/stage_1_jsons/stage_1_' + file_name + '.json', 'a') as g:
            g.write(json.dumps(eval(prediction)) + '\n')


if __name__ == '__main__':

    path = "/fs/nexus-projects/brain_project/icml/acl_audio_rewrite/label_txts/"
    file_list = ['audioset.txt','nsynth.txt']

    for item in file_list:
        get_rewrites(path+item,item.strip(".txt"))




