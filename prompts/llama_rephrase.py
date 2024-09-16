import os
# os.environ['TRANSFORMERS_CACHE'] = './cache/hub'
from dataclasses import dataclass
from itertools import chain
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Literal, Union, Tuple, Optional
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, GenerationConfig
from torch.utils.data import DataLoader
from huggingface_hub.hf_api import HfFolder
import datasets
import argparse
import pandas as pd
import re
from datasets import load_dataset, Dataset
from enum import Enum, unique

from formatter import EmptyFormatter, FunctionFormatter, StringFormatter, ToolFormatter
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union


parser = argparse.ArgumentParser(description="Split a CSV file into n equal parts")
parser.add_argument("--file_path", type=str, help="Path to the input CSV file")
parser.add_argument("--iteration", type=str, help="iter")
args = parser.parse_args()


@unique
class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"
    OBSERVATION = "observation"


@dataclass
class Template:
    format_user: "Formatter"
    format_assistant: "Formatter"
    format_system: "Formatter"
    format_function: "Formatter"
    format_observation: "Formatter"
    format_tools: "Formatter"
    format_separator: "Formatter"
    format_prefix: "Formatter"
    default_system: str
    stop_words: List[str]
    image_token: str
    efficient_eos: bool
    replace_eos: bool

    def encode_oneturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: Sequence[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
    ) -> Tuple[List[int], List[int]]:
        r"""
        Returns a single pair of token ids representing prompt and response respectively.
        """
        encoded_messages = self._encode(tokenizer, messages, system, tools)

        prompt_ids = encoded_messages[0]
        return prompt_ids, " "

    def encode_multiturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: Sequence[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
    ) -> List[Tuple[List[int], List[int]]]:
        r"""
        Returns multiple pairs of token ids representing prompts and responses respectively.
        """
        encoded_messages = self._encode(tokenizer, messages, system, tools)
        return [(encoded_messages[i], encoded_messages[i + 1]) for i in range(0, len(encoded_messages), 2)]

    def extract_tool(self, content: str) -> Union[str, List[Tuple[str, str]]]:
        r"""
        Extracts tool message.
        """
        return self.format_tools.extract(content)

    def _encode(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: Sequence[Dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
    ) -> List[List[int]]:
        r"""
        Encodes formatted inputs to pairs of token ids.
        Turn 0: prefix + system + query        resp
        Turn t: sep + query                    resp
        """
        system = system or self.default_system
        encoded_messages = []
        for i, message in enumerate(messages):
            elements = []

            if i == 0:
                elements += self.format_prefix.apply()
                if system or tools:
                    tool_text = self.format_tools.apply(content=tools)[0] if tools else ""
                    elements += self.format_system.apply(content=(system + tool_text))

            if i > 0 and i % 2 == 0:
                elements += self.format_separator.apply()

            if message["role"] == Role.USER.value:
                elements += self.format_user.apply(content=message["content"], idx=str(i // 2))
            elif message["role"] == Role.ASSISTANT.value:
                elements += self.format_assistant.apply(content=message["content"])
            elif message["role"] == Role.OBSERVATION.value:
                elements += self.format_observation.apply(content=message["content"])
            elif message["role"] == Role.FUNCTION.value:
                elements += self.format_function.apply(content=message["content"])
            else:
                raise NotImplementedError("Unexpected role: {}".format(message["role"]))

            encoded_messages.append(self._convert_elements_to_ids(tokenizer, elements))

        return encoded_messages

    def _convert_elements_to_ids(self, tokenizer: "PreTrainedTokenizer", elements: "SLOTS") -> List[int]:
        r"""
        Converts elements to token ids.
        """
        token_ids = []
        for elem in elements:
            if isinstance(elem, str):
                if len(elem) != 0:
                    token_ids += tokenizer.encode(elem, add_special_tokens=False)
            elif isinstance(elem, dict):
                token_ids += [tokenizer.convert_tokens_to_ids(elem.get("token"))]
            elif isinstance(elem, set):
                if "bos_token" in elem and tokenizer.bos_token_id is not None:
                    token_ids += [tokenizer.bos_token_id]
                elif "eos_token" in elem and tokenizer.eos_token_id is not None:
                    token_ids += [tokenizer.eos_token_id]
            else:
                raise ValueError("Input must be string, set[str] or dict[str, str], got {}".format(type(elem)))

        return token_ids

def remove_text_within_brackets(text):
    # This pattern matches text within brackets
    # It handles (), [], and {} brackets
    pattern = r'\[.*?\]|\(.*?\)|\{.*?\}'
    return re.sub(pattern, '', text)

model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
HfFolder.save_token('hf_key')

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, access_token="hf_key", padding_side='left')
dataset = load_dataset('csv', data_files={'train': args.file_path})

templates: Dict[str, Template] = {}

def register_template(
    name: str,
    format_user: Optional["Formatter"] = None,
    format_assistant: Optional["Formatter"] = None,
    format_system: Optional["Formatter"] = None,
    format_function: Optional["Formatter"] = None,
    format_observation: Optional["Formatter"] = None,
    format_tools: Optional["Formatter"] = None,
    format_separator: Optional["Formatter"] = None,
    format_prefix: Optional["Formatter"] = None,
    default_system: str = "",
    stop_words: Sequence[str] = [],
    image_token: str = "<image>",
    efficient_eos: bool = False,
    replace_eos: bool = False,
) -> None:
    r"""
    Registers a chat template.

    To add the following chat template:
    ```
    [HUMAN]:
    user prompt here
    [AI]:
    model response here

    [HUMAN]:
    user prompt here
    [AI]:
    model response here
    ```

    The corresponding code should be:
    ```
    _register_template(
        name="custom",
        format_user=StringFormatter(slots=["[HUMAN]:\n{{content}}\n[AI]:\n"]),
        format_separator=EmptyFormatter(slots=["\n\n"]),
        efficient_eos=True,
    )
    ```
    """
    eos_slots = [] if efficient_eos else [{"eos_token"}]
    template_class = Template
    default_user_formatter = StringFormatter(slots=["{{content}}"])
    default_assistant_formatter = StringFormatter(slots=["{{content}}"] + eos_slots)
    default_function_formatter = FunctionFormatter(slots=eos_slots, tool_format="default")
    default_tool_formatter = ToolFormatter(tool_format="default")
    default_separator_formatter = EmptyFormatter()
    default_prefix_formatter = EmptyFormatter()
    templates[name] = template_class(
        format_user=format_user or default_user_formatter,
        format_assistant=format_assistant or default_assistant_formatter,
        format_system=format_system or default_user_formatter,
        format_function=format_function or default_function_formatter,
        format_observation=format_observation or format_user or default_user_formatter,
        format_tools=format_tools or default_tool_formatter,
        format_separator=format_separator or default_separator_formatter,
        format_prefix=format_prefix or default_prefix_formatter,
        default_system=default_system,
        stop_words=stop_words,
        image_token=image_token,
        efficient_eos=efficient_eos,
        replace_eos=replace_eos,
    )


register_template(
    name="llama3",
    format_user=StringFormatter(
        slots=[
            (
                "<|start_header_id|>user<|end_header_id|>\n\n{{content}}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        ]
    ),
    format_system=StringFormatter(slots=["<|start_header_id|>system<|end_header_id|>\n\n{{content}}<|eot_id|>"]),
    format_observation=StringFormatter(
        slots=[
            (
                "<|start_header_id|>tool<|end_header_id|>\n\n{{content}}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        ]
    ),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    stop_words=["<|eot_id|>"],
    replace_eos=True,
)


def get_template_and_fix_tokenizer(
    name: str,
    tokenizer: "PreTrainedTokenizer"
) -> Template:
    
    # if tokenizer.eos_token_id is None:
    #     tokenizer.eos_token = "<|endoftext|>"
    #     # logger.info("Add eos token: {}".format(tokenizer.eos_token))

    # if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
        # logger.info("Add pad token: {}".format(tokenizer.pad_token))

    if name is None:
        return None

    template = templates.get(name, None)
    assert template is not None, "Template {} does not exist.".format(name)
    tokenizer.add_special_tokens(
        dict(additional_special_tokens=template.stop_words),
        replace_additional_special_tokens=False
    )
    return template
    
template = get_template_and_fix_tokenizer("llama3", tokenizer)

prompt = """I will provide you with a caption of an audio. The caption may describe the scene in which the particular audio occurs by taking names of particular objects that are making the sound. Rewrite the audio caption by describing the individual sounds. You should describe the sounds with their unique characteristics and how humans perceive them. Keep it to less than thirty words and don't change the overall description that describes the scene, only rewrite.\n\nHere are some input and output examples:\n\nInput Caption: The person plays a guitar, strumming and plucking the strings, followed by a loud explosion..\nOutput Caption: Gentle strumming and plucking of guitar strings, followed by a sudden, thunderous explosion.\n\nInput Caption: The lively rock and roll music plays in the background while performers dance and interact on stage, followed by a dramatic haze effect.\nOutput Caption: Energetic, pulsating rock and roll music in the background, rhythmic dancing and interactions on stage, concluding with a dramatic, hissing haze effect.\n\nInput Caption: A young girl is singing along to music while speaking on a video call, with a gentle breeze blowing her hair.\nOutput Caption: Melodic singing blending with music, interspersed with conversation on a video call, accompanied by the soft rustling of hair in a gentle breeze.\nHere is the input caption you need to rewrite: """

def construct_example(examples: Dict[str, List[Any]]) -> Generator[Any, None, None]:
    for i in range(len(examples['caption'])):
        query = examples["caption"][i]
        temp = tokenizer.tokenize(query)[:1024]
        try:
            last_occurence=len(temp)-temp[::-1].index('.')-1
        except:
            last_occurence=1023
        temp = temp[:last_occurence+1]
        query = tokenizer.convert_tokens_to_string(temp)
        query = prompt + query
        yield query

def preprocess_dataset(examples: Dict[str, List[Any]]) -> Dict[str, List[List[int]]]:

    model_inputs = {"input_ids": [], "attention_mask": []}

    for query in construct_example(examples):
        # print(query)
        query = [{"role": "user", "content": query}]
        input_ids, _ = template.encode_oneturn(tokenizer, query, "")

        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append([1] * len(input_ids))

    return model_inputs

kwargs = dict(num_proc=4, load_from_cache_file=True, desc="Running tokenizer on dataset")

dataset = dataset.map(preprocess_dataset, batched=True, remove_columns=["path","caption","dataset","start","end","split_name"], **kwargs)

data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=None,
        label_pad_token_id=tokenizer.pad_token_id
    )

dataloader = DataLoader(dataset['train'], batch_size=16, collate_fn=data_collator)

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

generating_args = {}
generating_args.update(dict(
    do_sample=True,
    temperature=0.4,
    top_p=1.0,
    top_k=50,
    num_return_sequences=1,
    # repetition_penalty=1.0,
    eos_token_id=[tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids,
    pad_token_id=tokenizer.pad_token_id,
    max_length = 2000
))


from transformers import InfNanRemoveLogitsProcessor, LogitsProcessorList

def get_logits_processor() -> LogitsProcessorList:
    r"""
    Gets logits processor that removes NaN and Inf logits.
    """
    logits_processor = LogitsProcessorList()
    logits_processor.append(InfNanRemoveLogitsProcessor())
    return logits_processor


for data in tqdm(dataloader):
    
    gen_kwargs = dict(
            generation_config=GenerationConfig(**generating_args),
            logits_processor=get_logits_processor()
        )
    
    try:
        generate_output = model.generate(input_ids = data['input_ids'].cuda(), attention_mask = data['attention_mask'].cuda(),**gen_kwargs)

        response = tokenizer.batch_decode(generate_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        with open('llama33_rephrased_captions_icassp_' + str(args.iteration) + '.txt', 'a') as f:
            for i in response:
                temp_prompt = i.split('\nHere is the input caption you need to rewrite: ')[1].split('assistant\n\n')[0]
                temp_res = i.split('\n\n')[-1]
                write_text = f'{temp_prompt}\t,\t{temp_res}'.encode('ascii', 'ignore').decode('ascii')
                f.write(write_text.replace("\n", "<br>") + "\n")
    except Exception as e:
        print(e)
        print("Error in parsing.")
