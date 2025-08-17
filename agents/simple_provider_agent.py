from ALL_KEYS import *
from utils.data_loader import *
from utils.llm import ChatGPT, AWSBedrockLLAMA
from utils.utils import detect_language

class gold_responses:
    def __init__(self, gold, gold_structure):
        self.gold = gold
        self.gold_structure = gold_structure
        self._process_initial_data()

    def _process_initial_data(self):
        del self.gold[0]
        del self.gold_structure[0]

        self.structure_to_text = {structure: self.gold[idx] for idx, structure in enumerate(self.gold_structure)}

        self.current_levels = {structure for structure in self.gold_structure if '.' not in structure}
        self.children = {structure: [] for structure in self.gold_structure}
        for structure in self.gold_structure:
            parts = structure.split('.')
            if len(parts) > 1:
                parent = '.'.join(parts[:-1])
                if parent in self.children:
                    self.children[parent].append(structure)

        self.current_display = self._get_current_display()

    def _get_current_display(self):
        display = []
        for structure in sorted(self.current_levels, key=lambda x: (int(x.split('.')[0]), float('inf') if '.' not in x else int(x.split('.')[1]))):
            if structure in self.structure_to_text:
                display.append(self.structure_to_text[structure])
        return display

    def use(self, order):
        if 1 <= int(order) <= len(self.current_display):
            structure = list(sorted(self.current_levels, key=lambda x: (int(x.split('.')[0]), float('inf') if '.' not in x else int(x.split('.')[1]))))[int(order) - 1]
            return_ = self.structure_to_text[structure]
            self.current_levels.discard(structure)
            self.current_levels.update(self.children[structure])
            self.current_display = self._get_current_display()
            return return_
        return None
    
    def get_response(self, order):
        if 1 <= int(order) <= len(self.current_display):
            structure = list(sorted(self.current_levels, key=lambda x: (int(x.split('.')[0]), float('inf') if '.' not in x else int(x.split('.')[1]))))[int(order) - 1]
            return_ = self.structure_to_text[structure]
            return return_
        return None

    def none_available_knowledges(self):
        return len(self.current_display) == 0

    def __iter__(self):
        self.iter_index = 0
        return self

    def __next__(self):
        if self.iter_index < len(self.current_display):
            result = self.current_display[self.iter_index]
            self.iter_index += 1
            return result
        else:
            raise StopIteration


class helper:
    def __init__(self, gold, task_data, gold_structure, all_task_data, llm='gpt4o') -> None:
        if llm == 'llama3.1-405B':
            self.llm = AWSBedrockLLAMA("llama3.1-405B", 'log/llm_helpers_cache_llama3.1-405B.pkl')
        else:
            self.llm = ChatGPT("gpt-4o-2024-05-13", 'log/llm_helpers_cache.pkl')
        
        self.count = 0
        self.gold = [s[4:].strip() if s.lower().startswith("jax:") else s.strip() for s in gold]
        self.gold_first = self.gold[0]
        self.chinese = True if detect_language(self.gold_first) == "Chinese" else False
        self.task_data = task_data
        if 'all_response_exaplain' in all_task_data:
            self.gold_explain = {}
            for i,g in enumerate(self.gold):
                if i == 0:
                    continue
                self.gold_explain[g] = all_task_data['all_response_exaplain'][i-1]
        self.gold = gold_responses(self.gold, gold_structure)

    def add_speaker(self, conv):
        speaker = self.task_data[0]
        for i in range(len(conv)):
            if i % 2 == 1:
                if not conv[i].strip().startswith(speaker + "：") and not conv[i].strip().startswith(speaker + ":") and not conv[i].strip().startswith(speaker + " ：") and not conv[i].strip().startswith(speaker + " :"):
                    conv[i] = speaker + ": " + conv[i]
        return conv


    def generate_reponse(self, previous_content):
        if not previous_content:
            if self.chinese:
                strat_word = 'Jax: 有什么可以帮助你？'
            else:
                strat_word = 'Jax: what can help you?'
            return strat_word
        
        if self.count == 0:
            self.count += 1
            return 'Jax: '+self.gold_first
        previous_content = self.add_speaker(previous_content)
        prompt = self.data2prompt(previous_content)
        print(prompt)
        response = self.prompt_pure(prompt)
        if response['index'] > 0:
            self.gold.use(str(response['index']))
        final_response = response['response'] if response['response'].lower().startswith("jax:") else "Jax: " + response['response'].strip()
        return final_response



    def prompt_pure(self, prompt):
        resonse,_ =  self.llm.request(prompt, None, json_format=True)
        return json.loads(resonse)


    def data2prompt(self, previous_content):
        if self.chinese:
            start = '你是Jax，一个游戏中的信息提供者。{0}需要与你交谈以获得所有信息来完成{1}的任务。你需要根据{0}对你的提问给出一个合理的回复。\n'.format(self.task_data[0],self.task_data[1])
        
            obj_skill_scenarios = ''
            if self.task_data[2]:
                obj_skill_scenarios = '\n{0}拥有下面的物品：\n'.format(self.task_data[0]) + self.task_data[2] + '\n'
            if self.task_data[3]:
                obj_skill_scenarios += '\n{0}拥有以下技能/动作：\n'.format(self.task_data[0]) + self.task_data[3] + '\n'
            if self.task_data[4]:
                obj_skill_scenarios += '\n此外，如果没有明确限制，{0}可以前往下列游戏中的场景：\n'.format(self.task_data[0]) + self.task_data[4] + '\n'
            
            previous_content = '\n这是你和{0}之前的对话内容：\n'.format(self.task_data[0]) + '\n'.join(previous_content) + '\n'
            
            numbered_gold = [f'{i+1}. {line}' for i, line in enumerate(self.gold)]
            knowledge = '\n这是你可以回复给{0}的知识库：（禁止融合多条知识的内容进行回复！优先选择前面的知识进行回复。）\n'.format(self.task_data[0]) + '\n'.join(numbered_gold) + '\n'

            end_1 = '现在你需要根据{0}跟你说的最后一句话，给{0}一个合理的回复。请你从知识库中选择一行知识进行回复。如果{0}同时提了多个问题，请优先选择排在前面的知识。需要注意的是，你只能选择一个知识且不能修改知识原文来进行回复。'.format(self.task_data[0])
            end_2 = '如果{0}问多个问题，其中一部分问题可以从知识库回答，则优先使用知识库序号靠前的知识进行回复。但如果{0}的所有提问都没有明确与知识库的知识点相关，或知识库中找不到用于回复{0}问题的内容，那么你需要生成一个积极的回复给他。例如，{0}问："有什么其他需要注意的吗？"这类不具体的问题时请你回复：应该没有其他需要注意的事项了。'.format(self.task_data[0])
            end_3 = '下面是常用的在知识库找不到答案的回复模板：1.多虑了，这个不会影响你的完成任务。2.你直接前往xxx即可，不需要担心。3.应该没有其他需要注意的事项了。4.你直接使用你的xxx技能即可完成任务。'
            end_4 = '如果{0}重复询问之前对话中你已经答复过的问题，请你重复之前的回复。对于与任务无关的提问，请直接回复：xxx与你的任务无关，所以我也不清楚。再次强调：禁止融合多条知识的内容进行回复！优先选择前面的知识进行回复。'.format(self.task_data[0])
            end_5 = "请返回包含两个字段的JSON对象：一个名为 'response' 的字段，它包回复{0}的内容，以及一个名为 'index' 的整数类型字段，它表示知识点的编号，如果回复的内容不来自知识点，则index为-1。格式应如下所示：{{ 'response': '应该没有其他需要注意的事项了。', 'index': -1 }}。"

            end = '\n' + '\n'.join([end_1,end_2,end_3,end_4,end_5]) + '\n'
            
            return start + obj_skill_scenarios + previous_content + knowledge  + end
        else:
            start = 'You are a {0} in a game. You need to talk to Jax to find out how to complete the task of {1} based on your existing items and skills.\n'.format(content[0], content[1])
            middle = ''
            end = '\nYour task is to ask Jax how to complete the task of {1} based on these scenes, and your items and skills/actions. If Jax\'s answer is unclear or ambiguity, you need to ask clarification questions to understand his intentions and gather enough information to complete the task. Note that based on the above scenes, your items and your skills/actions will definitely allow you to complete the task of {1}. Once you figure out how to complete the task, say "goodbye" to Jax to end your conversation.\nRemember, you are a {0}, please talk to Jax and ask him questions to find the specific steps to complete the task.'.format(content[0], content[1])
            return start + middle + end    

    def is_conv_end(self, conv):
        if not conv:
            return False
        if self.chinese:
            if '再见' == conv[-1] or '再见！' in conv[-1] or '再见。' in conv[-1] or '再见，' in conv[-1] or '再见!' in conv[-1] or '再见.' in conv[-1] or '再见,' in conv[-1] or '再见Jax' in conv[-1] or '，再见' in conv[-1] or '。再见' in conv[-1] or '.再见' in conv[-1] or ',再见' in conv[-1] or ' 再见' in conv[-1] or '再见 ' in conv[-1] or '\n再见' in conv[-1] or '再见\n' in conv[-1]:
                return True
        else:
            if 'Goodbye' == conv[-1]  or 'Goodbye!' in conv[-1] or 'Goodbye.' in conv[-1] or 'Goodbye,' in conv[-1] or 'Goodbye Jax' in conv[-1] or ', Goodbye' in conv[-1] or '. Goodbye' in conv[-1] or '.Goodbye' in conv[-1] or ',Goodbye' in conv[-1] or ' Goodbye' in conv[-1] or 'Goodbye ' in conv[-1] or '\nGoodbye' in conv[-1] or 'Goodbye\n' in conv[-1]:
                return True
        return False

