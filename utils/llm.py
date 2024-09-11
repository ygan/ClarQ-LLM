
import openai
from openai import OpenAI
import os, re, json, copy
from utils.log import llm_log
import time

import pickle
from collections import OrderedDict

import boto3
from botocore.exceptions import ClientError

class LLM:
    def __init__(self, cache = None) -> None:
        self.cache_path = cache
        self.cache = None
        self.cache_capacity = 100000
        if self.cache_path:
            if os.path.exists(self.cache_path):
                with open(self.cache_path, 'rb') as file:
                    self.cache = pickle.load(file)
            else:
                self.cache = OrderedDict()




    def extract_json_string(self, input_string):
        def process_colons_string(input_string, colon_positions):
            def find_string_bounds(s, start_pos, next_pos=None):
                colon_pos = s.find(':', start_pos)
                quote_start = colon_pos + 1
                
                while quote_start < len(s) and s[quote_start] in ' \t\n':
                    quote_start += 1
                
                if s[quote_start] != '"':
                    return s
                
                string_start = quote_start
                
                if next_pos:
                    substring = s[string_start:next_pos]
                    quote_end = next_pos
                    escaped = 0
                    while quote_end > string_start:
                        quote_end -= 1
                        if s[quote_end] == '"' and escaped != 2:
                            escaped += 1
                        elif s[quote_end] == '"' and escaped == 2:
                            break
                else:
                    substring = s[string_start:]
                    quote_end = len(s)
                    while quote_end > string_start:
                        quote_end -= 1
                        if s[quote_end] == '"':
                            break
                        
                actual_string = s[string_start+1:quote_end]
                
                escaped_string = ""
                is_previous_backslash = False
            
                for char in actual_string:
                    if char == '"' and not is_previous_backslash:
                        escaped_string += '\\"'
                    else:
                        escaped_string += char
                    is_previous_backslash = (char == '\\')

                new_string = s[:string_start+1] + escaped_string + s[quote_end:]
                
                return new_string
            
            result_string = input_string
            length_change = 0
            for i, pos in enumerate(colon_positions):
                end_pos = colon_positions[i+1] if i+1 != len(colon_positions) else None
                if end_pos:
                    result_string = find_string_bounds(result_string, pos + length_change, end_pos+ length_change)
                else:
                    result_string = find_string_bounds(result_string, pos + length_change)
                length_change = len(result_string) - len(input_string)
            return result_string

        def find_colons(input_string):
            pattern = r'"\s*:\s*'
            matches = []
            for match in re.finditer(pattern, input_string):
                colon_pos = match.start() + match.group().find(':')
                matches.append(colon_pos)
            return matches
        
        def clean_json_string(json_str):
            json_str = json_str.replace("'", '"')
            json_str = re.sub(r'\bTrue\b', 'true', json_str)
            json_str = re.sub(r'\bFalse\b', 'false', json_str)  
            colon_positions = find_colons(json_str)
            json_str = process_colons_string(json_str, colon_positions)
            return json_str
        
        start = -1
        brace_count = 0
        
        for i, char in enumerate(input_string):
            if char == '{':
                if brace_count == 0:
                    start = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start != -1:
                    json_str = input_string[start:i+1]
                    json_str = clean_json_string(json_str)
                    
                    try:
                        json.loads(json_str)
                        return json_str
                    except json.JSONDecodeError as e:
                        print(json_str)
                        pass
                    start = -1

        return ''

    def from_cache(self, message):
        if self.cache:
            message_str = str(message)
            if message_str in self.cache:
                refresh = self.cache[message_str]
                self.cache.pop(message_str)
                self.cache[message_str] = refresh

                # Extract data from the existing cache.
                # if os.path.exists(self.cache_path+'.back'):
                #     with open(self.cache_path+'.back', 'rb') as file:
                #         self.cache_back = pickle.load(file)
                # else:
                #     self.cache_back = OrderedDict()
                # if message_str not in self.cache_back:
                #     self.cache_back[message_str] = refresh
                #     with open(self.cache_path+'.back', 'wb') as file:
                #         pickle.dump(self.cache_back, file)

                return refresh
        return None
    
    def save_to_cache(self, message, response):
        if self.cache_path:
            message_str = str(message)
            self.cache[message_str] = response
            with open(self.cache_path, 'wb') as file:
                pickle.dump(self.cache, file)
            if len(self.cache) > self.cache_capacity:
                self.cache.popitem(last=False)


    def request(self, prompt, stop, **kwargs):
        return
    
    def log(self, input, output, **kwargs):
        llm_log(input, output, **kwargs)
        pass



class ChatGPT(LLM):
    def __init__(self, name, cache = None) -> None:
        super().__init__(cache)
        name = name.lower()
        if name in ["gpt-4","gpt4"]:
            self.model_name = "gpt-4-0613"
        elif name == 'gpt-4-1106-preview':
            self.model_name = "gpt-4-1106-preview"  
        elif  name in ["gpt-3.5","gpt3.5"]:
            self.model_name = "gpt-3.5-turbo-0125"
        else:
            self.model_name = name
        openai.api_key = os.environ["OPENAI_API_KEY"]
        self.client = OpenAI()

        
    def request(self, prompt, stop, **kwargs):
        message = [{
                    "role": "user",
                    "content": prompt
                }]
        if 'previous_message' in kwargs and kwargs['previous_message']:
            message = kwargs['previous_message'].extend(message)
            message = kwargs['previous_message']
        json_format = True if 'json_format' in kwargs and kwargs['json_format'] else False

        response = self.from_cache(message)
        if response:
            message.append({"role": "assistant", "content": response})
            return response, message
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                temperature=0,
                messages=message,
                stop = stop,
                seed = 8848,
                **({"response_format": {"type": "json_object"}} if json_format else {})
            )
            time.sleep(0.5)
        except:
            time.sleep(5)
            completion = self.client.chat.completions.create(
                model=self.model_name,
                temperature=0,
                messages=message,
                stop = stop,
                seed = 8848,
                **({"response_format": {"type": "json_object"}} if json_format else {})
            )
            time.sleep(0.5)
        super().log(message, completion.choices[0].message.content, model=completion.model, system_fingerprint = completion.system_fingerprint, usage = [completion.usage.prompt_tokens, completion.usage.completion_tokens, completion.usage.total_tokens])
        self.save_to_cache(message, completion.choices[0].message.content)
        
        message.append({"role": "assistant", "content": completion.choices[0].message.content})
        
        return completion.choices[0].message.content, message



class QianFan(LLM):
    def __init__(self, name, cache = None) -> None:
        import qianfan
        super().__init__(cache)
        self.chat_comp = qianfan.ChatCompletion(model=name)

    def request(self, prompt, stop, **kwargs):
        message = [{
                    "role": "user",
                    "content": prompt
                }]
        if 'previous_message' in kwargs and kwargs['previous_message']:
            message = kwargs['previous_message'].extend(message)
            message = kwargs['previous_message']

        response = self.from_cache(message)
        if response:
            message.append({"role": "assistant", "content": response})
            return response, message
        
        try:
            completion = self.chat_comp.do(messages=message, top_p=1, temperature=0.0000001, penalty_score=1.0)
            time.sleep(0.5)
        except:
            time.sleep(5)
            completion = self.chat_comp.do(messages=message, top_p=1, temperature=0.0000001, penalty_score=1.0)
            time.sleep(0.5)
        super().log(message, completion.body['result'], model=self.chat_comp._model, system_fingerprint = completion.body['id'], usage = [completion.body['usage']['prompt_tokens'], completion.body['usage']['completion_tokens'], completion.body['usage']['total_tokens']])
        self.save_to_cache(message, completion.body['result'])
        
        message.append({"role": "assistant", "content": completion.body['result']})

        return completion.body['result'], message




class LLAMA(LLM):
    def __init__(self, name, cache = None) -> None:
        import torch
        from transformers import LlamaForCausalLM
        from transformers import LlamaTokenizer
        
        # Set the seeds for reproducibility
        torch.cuda.manual_seed(8848)
        torch.manual_seed(8848)

        super().__init__(cache)
        self.model_name = "llama"
        self.is_chat_version = False

        self.max_new_tokens = 0
        all_names = name.split(" ")
        if "-chat" in all_names[0]:
            self.is_chat_version = True

        for w in all_names:
            if ":" in w:
                k,v = w.split(":")
                if k == "max_new_tokens":
                    self.max_new_tokens = int(v)
        self.model = LlamaForCausalLM.from_pretrained(
            all_names[0],
            return_dict=True,
            load_in_8bit=False,
            device_map="auto",
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16
            )
        self.model.eval()
        self.tokenizer = LlamaTokenizer.from_pretrained(all_names[0])
        self.tokenizer.pad_token = self.tokenizer.eos_token
        

    def request(self, prompt, stop, **kwargs):
        import torch       
        message = [{ 
                    "role": "user",
                    "content": prompt
                }]
        
        if self.is_chat_version and "[/INST]" not in prompt and  'previous_message' not in kwargs:
            prompt = "<s>[INST]<<SYS>>" + prompt + "<</SYS>>[/INST]"
        elif self.is_chat_version and 'previous_message' in kwargs and kwargs['previous_message'][0]['role'] == 'system':
            system = "<s>[INST] <<SYS>>" + kwargs['previous_message'][0]['content'] + "<</SYS>>\n"
            for i in range(len(kwargs['previous_message'])-1):
                i += 1
                system += kwargs['previous_message'][i]['content'] 
                if i % 2 == 1:
                    system += " [/INST]\n"
                else:
                    system += " </s><s>[INST]\n"
            prompt = system + prompt + " [/INST]"
        
        if 'previous_message' in kwargs and kwargs['previous_message']:
            message = kwargs['previous_message'].extend(message)
            message = kwargs['previous_message']

        response = self.from_cache(message)
        if response:
            message.append({"role": "assistant", "content": response})
            return response, message

        with torch.no_grad():
            batch = self.tokenizer(prompt, padding='max_length', truncation=True, max_length=None, return_tensors="pt")
            batch = {k: v.to("cuda") for k, v in batch.items()}

            outputs = self.model.generate(
                **batch,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                top_p=1.0,
                temperature=1.0,
                min_length=None,
                use_cache=True,
                top_k=50,
                repetition_penalty=1.0,
                length_penalty=1,
            )
            
            output_text = self.tokenizer.decode(outputs[0][batch['input_ids'].shape[-1]:], skip_special_tokens=True)
            
        super().log(prompt, output_text, model=self.model_name)
        self.save_to_cache(message, output_text)
        
        message.append({"role": "assistant", "content": output_text})
        return output_text, message
    





class AWSBedrockLLAMA(LLM):
    def __init__(self, name, cache = None) -> None:
        super().__init__(cache)
        name = name.lower()
        if name == 'llama3.1-405b':
            self.model_name = "meta.llama3-1-405b-instruct-v1:0"
        else:
            self.model_name = name
        session = boto3.Session(
            aws_access_key_id=os.environ["aws_access_key_id"],
            aws_secret_access_key=os.environ["aws_secret_access_key"],
            region_name=os.environ["region_name"]
        )
        # Create a Bedrock Runtime client in the AWS Region you want to use.
        self.client = session.client("bedrock-runtime")

    def request(self, prompt, stop, **kwargs):
        message = [{
                    "role": "user",
                    "content": [{"text": prompt}],
                }]
        
        system_prompts = None
        message_for_cache = message

        if 'previous_message' in kwargs and kwargs['previous_message']:
            message = kwargs['previous_message'].extend(message)
            message = kwargs['previous_message']
            if message[0]['role'] == 'system':
                system_prompts = [{"text": message[0]['content']}]
                message_for_cache = copy.deepcopy(message)
                del message[0]
            for m in message:
                if type(m['content']) == str:
                    m['content'] = [{"text": m['content']}]

        response = self.from_cache(message_for_cache)
        
        if response:
            message.append({"role": "assistant", "content": [{"text": response}]})
            return (self.extract_json_string(response), message) if 'json_format' in kwargs and kwargs['json_format'] else (response, message)
        
        temperature = 0.00001
        topP = 0.9999
        while True:
            try:
                # Send the message to the model, using a basic inference configuration.
                if system_prompts:
                    response = self.client.converse(
                        modelId=self.model_name,
                        messages=message,   system = system_prompts,
                        inferenceConfig={"maxTokens":512,"temperature":temperature,"topP":topP},
                        additionalModelRequestFields={}
                    )
                else:
                    response = self.client.converse(
                        modelId=self.model_name, messages=message,
                        inferenceConfig={"maxTokens":512,"temperature":temperature,"topP":topP},additionalModelRequestFields={}
                    )

                # Extract and print the response text.
            except (ClientError, Exception) as e:
                time.sleep(10)
                if system_prompts:
                    response = self.client.converse(
                        modelId=self.model_name,
                        messages=message, system = system_prompts,
                        inferenceConfig={"maxTokens":512,"temperature":temperature,"topP":topP},
                        additionalModelRequestFields={}
                )
                else:
                    response = self.client.converse(
                        modelId=self.model_name, messages=message,
                        inferenceConfig={"maxTokens":512,"temperature":temperature,"topP":topP},additionalModelRequestFields={}
                    )
                time.sleep(0.5)
            temperature += 0.333 
            response_text = response["output"]["message"]["content"][0]["text"].strip()
            super().log(message_for_cache, response_text, system_fingerprint = response['stopReason'], usage = [response['usage']['inputTokens'], response['usage']['outputTokens'], response['usage']['totalTokens']])
            
            if 'json_format' in kwargs and kwargs['json_format'] and not self.extract_json_string(response_text):
                assert temperature < 1
            else:
                break        
        self.save_to_cache(message_for_cache, response_text)
        
        message.append({"role": "assistant", "content": [{"text": response_text}]})
        
        return (self.extract_json_string(response_text), message) if 'json_format' in kwargs and kwargs['json_format'] else (response_text, message)
