import openai
import time
import yaml
from ruamel.yaml import YAML
from rasa.shared.nlu.training_data.formats.rasa_yaml import RasaYAMLReader 


openai.api_key = ''

messages = [ {"role": "system", "content": 
              ""} ]


###########

input_file = '../data/nlu.yml'

actual_data = RasaYAMLReader().read(input_file)

training_sets = actual_data.training_examples
negative_utterances={}
intents = []

for ts in training_sets:
    time.sleep(40)
    ts = ts.as_dict()
    intent = ts.get("intent")
    text = ts.get("text")
    if intent != "out_of_scope":
        if intent in intents:    
            messages.append(
            {"role": "user", "content": f" Write a negative example of the following sentence {text}"},
            )
            chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages
            )
            reply = chat.choices[0].message.content.split('"')[3] if len(chat.choices[0].message.content.split('"'))>4 else ""
            negative_utterances[f"not_{intent}"].append(reply)

        else:
            intents.append(intent)
            messages.append(
            {"role": "user", "content": f" What is the opposite of {text}"},
            )
            chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages
            )
            reply = chat.choices[0].message.content.split('"')[3] if len(chat.choices[0].message.content.split('"'))>4 else ""
            negative_utterances[f"not_{intent}"] = [reply]

file=open("../data/negative_nlu.yaml","w")
file.write('version: \"3.1\"\nnlu:\n')
yaml.dump(negative_utterances,file)
file.close()

