import copy
from ALL_KEYS import *
from utils.data_loader import *
from utils.llm import ChatGPT, AWSBedrockLLAMA
from agents.simple_provider_agent import helper



class helpers(helper):
    def __init__(self, gold, task_data, gold_structure, all_task_data, llm='gpt4o') -> None:
        super().__init__(gold, task_data, gold_structure, all_task_data)
        if llm == 'llama3.1-405B':
            self.llm = AWSBedrockLLAMA("llama3.1-405B", 'log/llm_helpers_cache_llama3.1-405B.pkl')
        else:
            self.llm = ChatGPT("gpt-4o-2024-05-13", 'log/llm_helpers_cache.pkl')
        self.strict = True


    def predifine_info(self, previous_content, response):
        g_r = None
        analysis = []
        for it, type_2_r in enumerate(self.gold):
            prompt = self.type2_double_check_one(previous_content, type_2_r)
            response_double_check = self.prompt_pure(prompt)
            if response_double_check['answerable']:
                g_r = self.gold.use(str(it+1))
                response['type'] = -1
                response['response'] = g_r
                break
            else:
                analysis.append(response_double_check['analysis'])

        if not g_r:
            for (it, type_2_r),a in zip(enumerate(self.gold),analysis):
                prompt = self.type2_double_check_one_3(previous_content, type_2_r, a)
                response_double_check = self.prompt_pure(prompt)
                if response_double_check['answerable']:
                    g_r = self.gold.use(str(it+1))
                    response['type'] = -1
                    response['response'] = g_r
                    break
        
        if not g_r:
            for it, type_2_r in enumerate(self.gold):
                if '存在多个' in self.gold_explain[type_2_r] or ' that there are multiple ' in self.gold_explain[type_2_r]:
                    prompt = self.type2_double_check_one_22(previous_content, type_2_r)
                    response_double_check = self.prompt_pure(prompt)
                    response_double_check['correct'] = response_double_check['answerable']
                else:
                    prompt = self.type2_double_check_one_2(previous_content, type_2_r)
                    response_double_check = self.prompt_pure(prompt)
                if response_double_check['correct']:
                    g_r = self.gold.use(str(it+1))
                    response['type'] = -1
                    response['response'] = g_r
                    break
        
        if not g_r and response['type'] in [2,3]:
            response['type'] = 3
        return response, g_r


    def generate_reponse(self, previous_content):
        previous_content = copy.deepcopy(previous_content)
        if not previous_content:
            if self.chinese:
                strat_word = 'Jax: 有什么可以帮助你？'
            else:
                strat_word = 'Jax: what can help you?'
            return strat_word
        
        previous_content = self.add_speaker(previous_content)
        if self.count == 0:
            prompt = self.data2prompt_0(previous_content)
            response = self.prompt_pure(prompt)
            if response['related']:
                self.count += 1
                return 'Jax: '+ self.gold_first
            return 'Jax: '+ '抱歉，我可能没有理解你的诉求，而且与任务无关的事情我不太清楚。' if self.chinese else 'Jax: '+ 'I am sorry, I might not have understood your request, and I am not very clear on matters unrelated to the task.'
        
        prompt = self.data2prompt_1(previous_content)
        response = self.prompt_pure(prompt)
        if self.gold.none_available_knowledges() and response['type'] in [2,3]:
                return 'Jax: '+ '你已经收集到足够完成任务的信息。' if self.chinese else 'Jax: '+ 'You have already collected enough information to complete the task.'
        
        if response['type'] in [4]:
            prompt = self.type_4_double_check(previous_content)
            response_double_check = self.prompt_pure(prompt)
            if not response_double_check['repeat']:
                response['type'] = 2
        if self.strict and response['type'] in [3,2]:
            response, g_r = self.predifine_info(previous_content, response)

        if response['type'] == -1:
            pass
        elif response['type'] == 1:
            prompt = self.data2prompt_main(previous_content)
            response = self.prompt_pure(prompt)
        elif response['type'] == 2:
            assert False
            prompt = self.data2prompt_partial_with_answer(previous_content)
            response = self.prompt_pure(prompt)
            if response['index'] > 0:
                g_r = self.gold.use(str(response['index']))
                if g_r:
                    response['response'] = g_r
        elif response['type'] in [3,6]:
            prompt = self.data2prompt_partial_without_answer(previous_content)
            response = self.prompt_pure(prompt)
        elif response['type'] == 4:
            prompt = self.data2prompt_repeat(previous_content)
            response = self.prompt_pure(prompt)
        elif response['type'] == 5:
            response['response'] = 'Jax: '+ '抱歉，我可能没有理解你的诉求，而且与任务无关的事情我不太清楚。如果你想结束对话，你可以和我说再见。' if self.chinese else 'Jax: '+ 'I am sorry, I might not have understood your request, and I am not very clear on matters unrelated to the task. If you want to end the conversation, you can said Goodbye to me.'
        else:
            assert False
        
        final_response = response['response'] if response['response'].lower().startswith("jax:") else "Jax: " + response['response'].strip()
        return final_response

    def prompt_pure(self, prompt):
        resonse,_ =  self.llm.request(prompt, None, json_format=True)
        return json.loads(resonse)


    def data2prompt_0(self, previous_content):
        if self.chinese:
            start = "你是Jax，一个信息分辨专家，并且你知道{1}任务的解决办法是:'{2}'。现在你要分辨{0}跟你说的话是否在向你咨询有关{1}任务的信息？\n".format(self.task_data[0],self.task_data[1],self.gold_first)

            previous_content = '\n这是你和{0}之前的对话内容：\n'.format(self.task_data[0]) + '\n'.join(previous_content) + '\n'

            end = "\n已知{1}任务的解决办法是: {2}。现在你要分辨{0}跟你说的话是否在向你咨询有关{1}任务的信息？请返回包含'related'字段的JSON对象： 'related'字段的值表示{0}是否向你咨询了与任务有关的内容（布尔类型）。格式应如下所示：{{ 'related': True 或者 False }}。".format(self.task_data[0], self.task_data[1], self.gold_first)
            return start + previous_content + end
        else:
            start = "You are Jax, an information discernment expert, and you know that the solution to the {1} task is: '{2}'. You need to determine whether what {0} is saying to you is inquiring about information related to the {1} task?\n".format(self.task_data[0], self.task_data[1], self.gold_first)

            previous_content = '\nThis is the previous conversation betweem you and {0}:\n'.format(self.task_data[0]) + '\n'.join(previous_content) + '\n'

            end = "\nIt is known that the solution to the {1} task is: {2}. You need to determine whether what {0} is saying to you is inquiring about information related to the {1} task? Return a JSON object containing a 'related' key: the value (boolean type) of the 'related' key indicates whether {0} has inquired about content related to the task. The format should be as follows: {{ 'related': True or False }}.".format(self.task_data[0], self.task_data[1], self.gold_first)
            return start + previous_content + end


    def data2prompt_1(self, previous_content):
        if self.chinese:
            start = "你是Jax，一个认真细心的对话分辨专家。{0}与你对话的目的是为了收集有关完成{1}任务的信息。前面你给{0}提供的信息是不完整的，如果{0}能发现缺失或者不明确的信息并就此提问或提出担忧，我们就把缺失的信息回复给{0}。现在你需要分辨{0}最后一句话属于以下哪一种类型：\n1.对整个{1}任务进行咨询和确认。\n2.对前面缺失的信息进行提问或提出担忧，且在缺失的回复信息列表中能找到一行合适的信息内容来回复{0}的提问。\n3.对任务细节进行清晰提问，但在缺失的回复信息中无法找到一行合适的信息来回复{0}。\n4.重复之前对话中已经被你回答过的问题。\n5.与任务无关的咨询或者闲聊。\n6.对任务细节进行提问，但是提问内容模凌两可容易让人产生困惑。或者直接提出一些空泛的提问（例如：下一步怎么做？还有什么需要注意的呢？其他细节呢？）。\n".format(self.task_data[0],self.task_data[1])

            previous_content = '\n你和{0}之前的对话内容：\n'.format(self.task_data[0]) + '\n'.join(previous_content) + '\n'
            if hasattr(self, 'gold_explain'):
                numbered_gold = [f'({i+1}) 信息回复内容： {line}\t\t\t\t信息用途：{self.gold_explain[line]}' for i, line in enumerate(self.gold)]
                knowledge = '\n{0}缺失的回复信息及其用途：\n'.format(self.task_data[0]) + '\n'.join(numbered_gold) + '\n'
            else:
                numbered_gold = [f'({i+1}) {line}' for i, line in enumerate(self.gold)]
                knowledge = '\n任务信息：\n' + '\n'.join(numbered_gold) + '\n'

            end = "\n请分辨对话内容中{0}最后一句话属于以下哪种类型：\n1.对整个{1}任务进行咨询和确认。\n2.对前面缺失的信息进行提问或提出担忧，且在缺失的回复信息列表中能找到一行合适的信息内容来回复{0}的提问。\n3.对任务细节进行清晰提问，但在缺失的回复信息中无法找到一行合适的信息来回复{0}。\n4.重复之前对话中已经被你回答过的问题。\n5.与任务无关的咨询或者闲聊。\n6.对任务细节进行提问，但是提问内容模凌两可容易让人产生困惑。或者直接提出一些空泛的提问（例如：下一步怎么做？还有什么需要注意的呢？其他细节呢？）。\n请返回包含一个'type'字段的JSON对象： 'type'字段的值是1到6，对应六种不同的对话目的。格式如下所示：{{ 'type': 1到6 }}。".format(self.task_data[0],self.task_data[1])
            return start + previous_content + knowledge + end
        else:
            start = "You are Jax, a conscientious and meticulous conversation discernment expert. {0} is conversing with you to gather information about completing the {1} task. The information you previously provided to {0} was incomplete. If {0} identifies any missing or unclear information and asks questions or raises concerns about it, we should provide the missing information to {0}. Now, you need to identify which type the last sentence spoken by {0} falls into from the following options:\n1. Inquiring and confirming about the entire {1} task.\n2. Asking clear questions about task details, with a line of missing information that can respond to {0}'s query.\n3. Asking clear questions about task details, but there is no line of information that can can respond to {0}'s query.\n4. Repeating a question that has already been answered by Jax in previous conversations.\n5. Consulting or chatting unrelated to the task.\n6. Asking questions about task details that are ambiguous and can cause confusion, or directly asking some vague questions (for example: What's the next step? What else should be paid attention to? Any other details?).\n".format(self.task_data[0],self.task_data[1])

            previous_content = '\nThis is the previous conversation betweem you and {0}:\n'.format(self.task_data[0]) + '\n'.join(previous_content) + '\n'
            if hasattr(self, 'gold_explain'):
                numbered_gold = [f'({i+1}) Information Content: {line}\t\t\t\tPurpose of Information: {self.gold_explain[line]}' for i, line in enumerate(self.gold)]
                knowledge = "\n{0}'s missing information and its purpose:\n".format(self.task_data[0]) + '\n'.join(numbered_gold) + '\n'
            else:
                numbered_gold = [f'({i+1}) {line}' for i, line in enumerate(self.gold)]
                knowledge = '\Task detail information:\n' + '\n'.join(numbered_gold) + '\n'

            end = "\nIdentify which type the last sentence spoken by {0} falls into from the following options:\n1. Inquiring and confirming about the entire {1} task.\n2. Asking clear questions about task details, with a line of missing information that can respond to {0}'s query.\n3. Asking clear questions about task details, but there is no line of information that can can respond to {0}'s query.\n4. Repeating a question that has already been answered by Jax in previous conversations.\n5. Consulting or chatting unrelated to the task.\n6. Asking questions about task details that are ambiguous and can cause confusion, or directly asking some vague questions (for example: What's the next step? What else should be paid attention to? Any other details?).\nReturn a JSON object containing a 'type' key: the value of the 'type' key is from 1 to 6, corresponding to the six different types of conversation purposes. The format should be as follows: {{ 'type': 1 to 6 }}.".format(self.task_data[0], self.task_data[1])
            return start + previous_content + knowledge + end


    def type_4_double_check(self, previous_content): 
        if self.chinese:
            start = "你是Jax，一个聪明的语言理解专家。认真分析{0}最后一句话所提的全部问题是否已被你回答过了？\n".format(self.task_data[0],self.task_data[1])

            previous_content = '\n你和{0}之前的对话内容：\n'.format(self.task_data[0]) + '\n'.join(previous_content) + '\n'

            end_1 = "\n分析{0}最后一句话所提的全部问题是否已被你回答过了？".format(self.task_data[0],self.task_data[1])
            end_5 = "返回包含两个字段的JSON对象：一个'analysis'字段和一个'repeat'字段。'analysis'字段是字符串，内容为分析{0}最后一句话的全部问题是否已在前面的对话中被回答？'repeat'字段的值是布尔类型，true表示{0}所有问题均为重复的被解答过的问题，反之为Fasle。格式如下所示：{{ 'analysis': '', 'repeat': Fasle  }}。".format(self.task_data[0])
            end = '\n' + '\n'.join([end_1,end_5]) + '\n'
            
        else:
            start = "You are Jax, a clever language understanding expert. Carefully analyze whether all the questions in {0}'s last sentence have already been answered by you.\n".format(self.task_data[0], self.task_data[1])

            previous_content = '\nThis is the previous conversation betweem you and {0}:\n'.format(self.task_data[0]) + '\n'.join(previous_content) + '\n'

            end_1 = "\nAnalyze whether all the questions in {0}'s last sentence have already been answered by you.".format(self.task_data[0], self.task_data[1])

            end_5 = "Return a JSON object containing two keys: a 'analysis' key, whose value is a string that contains the analysis of whether all the questions in {0}'s last sentence have been answered in the previous conversation; and an 'repeat' key, whose value is a boolean: True indicates that all of {0}'s questions have been previously answered, otherwise False. The format is as follows: {{ 'analysis': '', 'repeat': False }}.".format(self.task_data[0])
            end = '\n' + '\n'.join([end_1,end_5]) + '\n'

        return start + previous_content  + end
    
    def extract_reference(self, explain):
        if self.chinese: 
            assert False
        else:
            start_phrase = 'that there are multiple '
            end_phrase = ' and'
            
            # Find the positions of the start and end phrases
            start_index = explain.find(start_phrase)
            end_index = explain.find(end_phrase, start_index + len(start_phrase))
            
            # If both phrases are found
            if start_index != -1 and end_index != -1:
                # Extract the substring between the start and end phrases
                start_index += len(start_phrase)
                substring = explain[start_index:end_index]
                
                # Split the substring into words
                words_in_between = substring.split()
                
                # Check the number of words in between
                if 1 <= len(words_in_between) <= 3:
                    return " ".join(words_in_between)
            
            # If conditions are not met, return None
            return None
        return None
    
    def type2_double_check_one(self, previous_content, type_2_response): 
        if self.chinese: 
            start = "结合{0}的提问、Jax的回复以及回复的用途分析该回复是否合理，是否应该用于回答{0}提出的部分问题。一般来说，只有Jax回复清晰解决了{0}的部分问题，Jax回复才有意义。\n".format(self.task_data[0])
            previous_content = '{0}的提问：\n'.format(self.task_data[0]) + previous_content[-1] + '\n'
            jax_content = "Jax的回复：\nJax: "+type_2_response+'\n\nJax这个回复的用途：\n'+self.gold_explain[type_2_response] + '\n'
            end_1 = "结合{0}的提问、Jax的回复以及回复的用途分析该回复是否合理，是否应该用于回答{0}提出的部分问题。一般来说，只有Jax回复清晰解决了{0}的部分问题，Jax回复才有意义。只要Jax的回复能解决{0}部分问题，那么Jax的回复就是合理的。\n返回包含两个字段的JSON对象：一个'analysis'字段和一个' answerable'字段。'analysis'字段是字符串，内容为分析{0}的部分提问能否被Jax回复解答。如果{0}的提问关注点与Jax的回复的关注点不同，那么Jax的回复是答非所问，解决不了{0}的疑惑。'answerable'字段的值是布尔类型，false表示Jax的回复无法解答{0}的任何问题即答非所问，反之为true。".format(self.task_data[0]) + "格式如下所示：{ 'analysis': '', 'answerable': Fasle  }。"
        else:
            multiple_obj = None
            if ' that there are multiple ' in self.gold_explain[type_2_response] and self.gold_explain[type_2_response].count(' that there are multiple ') == 1:
                multiple_obj = self.extract_reference(self.gold_explain[type_2_response])
            if multiple_obj:
                start = "Does {0} acknowledge that there are multiple {1} and raise a clarification question to inquire which {1} {0} should choose? If {0} has inquired this, then Jax's response can precisely answer part of {0}'s question.\n".format(self.task_data[0],multiple_obj)
                end_1 = "Does {0} acknowledge that there are multiple {1} and raise a clarification question to inquire which {1} {0} should choose or what types of {1} should {0} choose? Sometimes, {0} mentions the multiple {1} but does not ask for clarification, making Jax's response inappropriate. If {0} has inquired about this, then Jax's response can precisely answer part of {0}'s question.\nReturn a JSON object with two keys: an 'analysis' key and an 'answerable' key. The 'analysis' key contains a string that analyzes whether {0} acknowledges multiple {1} needing clarification. The 'answerable' key contains a boolean value: false indicates that {0} does not acknowledge this and Jax's response should not be used to reply to {0}'s questions, while true indicates the opposite.".format(self.task_data[0],multiple_obj) + "The format is as follows: { 'analysis': '', 'answerable': false }."
            else:
                start = "Combine {0}'s questions, Jax's responses, and the purpose of Jax's responses to analyze whether Jax's response is reasonable and whether it can be used to answer some of {0}'s questions. Generally, Jax's response only makes sense if it clearly resolves some of {0}'s questions.\n".format(self.task_data[0])
                end_1 = "Combine {0}'s questions, Jax's responses, and the purpose of Jax's responses to analyze whether Jax's response is reasonable and whether it can be used to answer some of {0}'s questions. Generally, Jax's response only makes sense if it clearly resolves some of {0}'s questions. As long as Jax's response can solve some of {0}'s questions, then Jax's response is reasonable.\nReturn a JSON object with two keys: an 'analysis' key and an 'answerable' key. The 'analysis' key contains a string that analyzes whether some of {0}'s questions can be answered by Jax's response. If the focus of {0}'s questions differs from the focus of Jax's response, then Jax's response is irrelevant and does not resolve {0}'s confusion. The 'answerable' key contains a boolean value; false indicates that Jax's response does not address any of {0}'s questions (i.e., is irrelevant), and true indicates the opposite.".format(self.task_data[0]) + "The format is as follows: { 'analysis': '', 'answerable': false }."
            previous_content = '{0}\'s questions:\n'.format(self.task_data[0]) + previous_content[-1] + '\n'
            jax_content = "Jax's response:\nJax: " + type_2_response + '\n\nPurpose of Jax\'s response:\n' + self.gold_explain[type_2_response] + '\n'

        prompt = '\n'.join([start, previous_content, jax_content, end_1])
        return prompt
    
    
    def type2_double_check_one_3(self, previous_content, type_2_response, analysis): 
        if self.chinese: 
            start = "根据你之前的分析，告诉我Jax的回复是否清晰解决了{0}提出的部分问题。Jax的回复只要能解决{0}提出的部分疑问即可，不需要解决全部疑问。\n".format(self.task_data[0])
            previous_content = '{0}的提问：\n'.format(self.task_data[0]) + previous_content[-1] + '\n'
            jax_content = "Jax的回复：\nJax: "+type_2_response+'\n\nJax这个回复的用途：\n'+self.gold_explain[type_2_response] + '\n'
            analysis = '这是你之前的分析：\n' + analysis + '\n'
            end_1 = "根据你之前的分析，告诉我Jax的回复是否清晰解决了{0}提出的部分问题。Jax的回复只要能解决{0}提出的部分疑问即可，不需要解决全部疑问。\n返回包含一个' answerable'字段的JSON对象：'answerable'字段的值是布尔类型，True表示{0}的部分提问能否被Jax的回复解答，但{0}仍有部分问题无法在Jax的回复中找到答案。False表示Jax的回复与{0}的提问完全无关。".format(self.task_data[0]) + "格式如下所示：{ 'answerable': Fasle  }。"
        else:
            start = "Based on your analysis, tell me whether Jax's response clearly resolved some of {0}'s questions. Jax's response only needs to address some of {0}'s questions, not all of them.\n".format(self.task_data[0])
            previous_content = '{0}\'s questions:\n'.format(self.task_data[0]) + previous_content[-1] + '\n'
            jax_content = "Jax's response:\nJax: " + type_2_response + '\n\nPurpose of Jax\'s response:\n' + self.gold_explain[type_2_response] + '\n'
            analysis = 'This is your analysis:\n' + analysis + '\n'
            end_1 = "Based on your analysis, tell me whether Jax's response clearly resolved some of {0}'s questions. Jax's response only needs to address some of {0}'s questions, not all of them.\nReturn a JSON object with an 'answerable' key. The 'answerable' key contains a boolean value; true indicates that some of {0}'s questions can be answered by Jax's response, although some questions remain unanswered. False indicates that Jax's response is completely unrelated to {0}'s questions.".format(self.task_data[0]) + "The format is as follows: { 'answerable': false }."

        prompt = '\n'.join([start, previous_content, jax_content, analysis, end_1])
        return prompt

    def type2_double_check_one_2(self, previous_content, type_2_response): 
        if self.chinese: 
            start = "Jax的回复是正确的。结合Jax的回复以及回复的用途分析{0}的语言是否存在错误的假设或推理。如果{0}存在错误的假设或推理，请问Jax的回复能否帮助{0}修正该错误？注意，这里的错误不包括：{0}对Jax回复中内容的忽略！\n".format(self.task_data[0])
            previous_content = '{0}的提问：\n'.format(self.task_data[0]) + previous_content[-1] + '\n'
            jax_content = "Jax的回复：\nJax: "+type_2_response+'\n\nJax这个回复的用途：\n'+self.gold_explain[type_2_response] + '\n'
            end_1 = "结合Jax的回复以及回复的用途分析{0}的语言是否存在错误的假设或推理。如果{0}存在错误的假设或推理，请问Jax的回复能否帮助{0}修正自己错误的假设或推理？注意，这里的错误不包括：{0}对Jax回复中内容的忽略！\n返回包含两个字段的JSON对象：一个'analysis'字段和一个'correct'字段。'analysis'字段是字符串，内容为分析从Jax的回复能否推理出{0}存在错误的假设或推理。'correct'字段的值是布尔类型，False表示没有发现{0}存在错误的假设或推理亦或者Jax的回复无法修正{0}的错误假设或推理。".format(self.task_data[0]) + "格式如下所示：{ 'analysis': '', 'correct': Fasle  }。"
        else:
            start = "Jax's response is correct. Combine Jax's response and its purpose to analyze whether there are any incorrect assumptions or reasoning in {0}'s statement. If {0}'s statement contains incorrect assumptions or reasoning, can Jax's response help {0} correct them? Note that errors do not include {0}'s neglect of the content in Jax's response!\n".format(self.task_data[0])
            previous_content = '{0}\'s statement:\n'.format(self.task_data[0]) + previous_content[-1] + '\n'
            jax_content = "Jax's response:\nJax: " + type_2_response + '\n\nPurpose of Jax\'s response:\n' + self.gold_explain[type_2_response] + '\n'
            end_1 = "Combine Jax's response and its purpose to analyze whether there are any incorrect assumptions or reasoning in {0}'s statement. If {0}'s statement contains incorrect assumptions or reasoning, can Jax's response help {0} correct them? Note that errors do not include {0}'s neglect of the content in Jax's response!\nReturn a JSON object with two keys: an 'analysis' key and a 'correct' key. The 'analysis' key contains a string value that analyzes whether Jax's response indicates that {0} has incorrect assumptions or reasoning. The 'correct' key contains a boolean value: false indicates that no incorrect assumptions or reasoning were found in {0}'s statement or that Jax's response cannot correct {0}'s incorrect assumptions or reasoning; true indicates the opposite.".format(self.task_data[0]) + "The format is as follows: { 'analysis': '', 'correct': false }."

        prompt = '\n'.join([start, previous_content, jax_content, end_1])
        return prompt
    
    def type2_double_check_one_22(self, previous_content, type_2_response): 
        if self.chinese: 
            start = "从{0}的提问能否看出{0}注意到了存在多个可能的选项并需要Jax的回复对此进行一定程度的澄清。\n".format(self.task_data[0])
            previous_content = '{0}的提问：\n'.format(self.task_data[0]) + previous_content[-1] + '\n'
            jax_content = "Jax的回复：\nJax: "+type_2_response+'\n\nJax这个回复的用途：\n'+self.gold_explain[type_2_response] + '\n'
            end_1 = "从{0}的提问能否看出{0}注意到了存在多个可能的选项并需要Jax的回复对此进行一定程度的澄清。\n返回包含两个字段的JSON对象：一个'analysis'字段和一个' answerable'字段。'analysis'字段是字符串，内容为分析{0}是否注意到与Jax回复有关的多个需要被澄清的对象。'answerable'字段的值是布尔类型，True表示{0}对多个可能选项的困惑正好能被Jax的回复所澄清，反之为False。".format(self.task_data[0]) + "格式如下所示：{ 'analysis': '', 'answerable': Fasle  }。"
        else:
            start = "Can it be inferred from {0}'s question that {0} has noticed the existence of multiple possible options and needs Jax's response to provide some clarification?\n".format(self.task_data[0])
            previous_content = '{0}\'s question:\n'.format(self.task_data[0]) + previous_content[-1] + '\n'
            jax_content = "Jax's response:\nJax: " + type_2_response + '\n\nPurpose of Jax\'s response:\n' + self.gold_explain[type_2_response] + '\n'
            end_1 = "Can it be inferred from {0}'s question that {0} has noticed the existence of multiple possible options and needs Jax's response to provide some clarification?\nReturn a JSON object with two keys: an 'analysis' key and an 'answerable' key. The 'analysis' key contains a string that analyzes whether {0} has noticed multiple options related to Jax's response that need clarification. The 'answerable' key contains a boolean value: true indicates that Jax's response can clarify {0}'s confusion about multiple possible options, otherwise false.".format(self.task_data[0]) + "The format is as follows: { 'analysis': '', 'answerable': false }."

        prompt = '\n'.join([start, previous_content, jax_content, end_1])
        return prompt
    
    
    def data2prompt_main(self, previous_content): 
        if self.chinese:
            start = "请你根据你和{0}之前的对话，给{0}一个积极的回复。\n".format(self.task_data[0],self.task_data[1])

            previous_content = '\n这是你和{0}之前的对话内容：\n'.format(self.task_data[0]) + '\n'.join(previous_content) + '\n'

            end = "\n请你根据你和{0}之前的对话，给{0}一个积极的恢复。例如：你的计划看起来不错，你还有什么具体问题吗？但注意：你不可以告诉{0}他已掌握了所有必要的信息或步骤之类的话，是否掌握所有信息必须由{0}自己判断，你不能告诉他或误导他。\n请返回包含'response'字段的JSON对象： 'response' 字段的值是回复{0}的内容。格式应如下所示：{{ 'response': '' }}。".format(self.task_data[0])
        else:
            start = 'Based on your previous conversation with {0}, give {0} a positive response.\n'.format(self.task_data[0], self.task_data[1])

            previous_content = '\nThis is the previous conversation betweem you and {0}:\n'.format(self.task_data[0]) + '\n'.join(previous_content) + '\n'
            end = "\nBased on your previous conversation with {0}, give {0} a positive response. For example: 'Your plan looks good, do you have any specific questions?' But note: you cannot tell {0} that they have all the necessary information or steps for the task. Whether {0} has all the information must be determined by {0} themselves, and you cannot tell or mislead them.\nReturn a JSON object with a 'response' key: the value of the 'response' key is the reply content to {0}. The format should be as follows: {{ 'response': '' }}.".format(self.task_data[0])

        return start + previous_content + end



    def data2prompt_partial_without_answer(self, previous_content): 
        if self.chinese:
            start = '你是Jax，一个游戏中的信息提供者。这是一个很简单的游戏，{0}在游戏中能做的主要事情为：使用物品、技能/动作或者抵达指定场景、与人对话。出此之前的事情基本上不会影响任务的执行。目前，{0}似乎考虑过多了，超出游戏任务的要求。请根据上下文生成一个回复安慰他，让他别想太多了。\n'.format(self.task_data[0],self.task_data[1])
     
            previous_content = '\n这是你和{0}之前的对话内容：\n'.format(self.task_data[0]) + '\n'.join(previous_content) + '\n'

            end_1 = '这是一个很简单的游戏，{0}在游戏中能做的主要事情为：使用物品、技能/动作或者抵达指定场景、与人对话。出此之前的事情基本上不会影响任务的执行。目前，{0}似乎考虑过多了，超出游戏任务的要求。请根据上下文生成一个回复安慰他，让他别想太多了。\n如果无法从之前的对话找到答案，那你可以尝试以下常用的积极回复例子："这个怎么处理都没关系，任务重点不在这"；"你想多了，任务没那么复杂，不需要考虑这几个细节。"；"你直接前往xxx场景即可，不需要担心。"；"你直接使用你的xxx技能即可。"；"我不太理解你的提问，你能问得更具体一点吗？"；"如果场景没有明确限制，那么你就可以直接前往，游戏地图会导航你到那。"。注意：你不能杜撰内容给{0}，所有内容不能超出你们之前的对话里的内容。简单安慰她即可。请根据你们对话的上下文和前面提供的积极回复例子生成一个合理的回复给{0}。但注意：你不可以告诉{0}他已掌握了所有必要的信息或步骤之类的话，也不能说没有其他需要注意的地方了，是否掌握所有信息必须由{0}自己判断，你不能告诉他或误导他。'.format(self.task_data[0])
            end_5 = "\n请返回包含'response'字段的JSON对象： 'response' 字段的值是回复{0}的内容。格式应如下所示：{{ 'response': '' }}。".format(self.task_data[0])

            end = '\n' + '\n'.join([end_1,end_5]) + '\n'
        else:
            start = 'You are Jax, an information provider in a very simple game where the main actions include: using items, skills/actions, or reaching specified scenes and talking to people. Actions outside of these basics are generally not needed for completing the game tasks. Currently, {0} seems to be overthinking beyond the requirements of the game tasks. Generate a response based on Jax\'s previous response to reassure {0} and let {0} know not to worry too much.\n'.format(self.task_data[0])

            previous_content = '\nThis is the previous conversation betweem you and {0}:\n'.format(self.task_data[0]) + '\n'.join(previous_content) + '\n'

            end_1 = 'This is a very simple game where the main actions include: using items, skills/actions, or reaching specified scenes and talking to people. Actions outside of these basics are generally not needed for completing the game tasks. Currently, {0} seems to be overthinking beyond the requirements of the game tasks. Generate a response based on the previous Jax\'s response to reassure {0} and let {0} know not to worry too much. Note: Your reponse can not contain {0}\'s statement or questions.\nIf you cannot find an proper response from the previous previous Jax\'s response, you can try the following common positive response examples: "It does not matter how you handle this; it is not the focus of the task."; "You are overthinking it; the task is not that complicated and you do not need to worry about these details."; "Just go directly to the xxx scenes, no need to worry."; "Just use your xxx skill."; "If the scene is not explicitly restricted, you can go directly there, the game map will navigate you." Note: You cannot make up content for {0}; all content must be within the your previous response. Please generate a reasonable response for {0} based on the your previous reponse and the positive response examples provided above. But note: you cannot tell {0} that they have all the necessary information or steps, nor can you say there are no other points to consider. Whether they have all the information must be determined by {0} themselves, and you cannot tell or mislead them.'.format(self.task_data[0])

            end_5 = "\nReturn a JSON object containing a 'response' key: the value of the 'response' key is the content of your reply to {0}. The format should be as follows: {{ 'response': '' }}.".format(self.task_data[0])

            end = '\n' + '\n'.join([end_1,end_5]) + '\n'
        return start   + previous_content  + end


    def data2prompt_repeat(self, previous_content):
        if self.chinese:
            start = "看起来{0}又问了之前提过的问题，请你根据你和{0}之前的对话，重复回答一遍。\n".format(self.task_data[0],self.task_data[1])

            previous_content = '\n这是你和{0}之前的对话内容：\n'.format(self.task_data[0]) + '\n'.join(previous_content) + '\n'

            end = "\n看起来{0}又问了之前提过的问题，请你根据你和{0}之前的对话，以积极方式重复回答一遍。如果{0}提问很空泛，例如：'我还需要注意什么？'你可以回：'我也不太清楚，你还有什么具体问题吗？'。注意不要提供任何自己杜撰的信息来回复，你回复所提供的信息必须出现在之前的对话中。\n请返回包含'response'字段的JSON对象： 'response' 字段的值是回复{0}的内容。格式应如下所示：{{ 'response': '' }}。".format(self.task_data[0])
            return start + previous_content + end
        else:
            start = 'It seems that {0} has asked a question that was previously addressed. Based on your prior conversation with {0}, repeat the answer once more.\n'.format(self.task_data[0], self.task_data[1])

            previous_content = '\nThis is the previous conversation betweem you and {0}:\n'.format(self.task_data[0]) + '\n'.join(previous_content) + '\n'

            end = '\nIt seems that {0} has asked a question that was previously addressed. Based on your prior conversation with {0}, repeat the answer once more in a positive manner. If {0}\'s question is vague, such as: "What else should I be aware of?" you can reply: "I\'m not quite sure either, do you have any specific questions?" Be careful not to provide any information you\'ve made up; the information in your reply must have appeared in your previous conversations.'.format(self.task_data[0])
            end_5 = "\nReturn a JSON object containing a 'response' key: the value of the 'response' key is the content of your reply to {0}. The format should be as follows: {{ 'response': '' }}.".format(self.task_data[0])

            return start + previous_content + end + end_5