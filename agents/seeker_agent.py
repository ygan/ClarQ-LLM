from ALL_KEYS import *
from utils.data_loader import *
from utils.utils import detect_language


class player:
    def __init__(self,  task_data, llm, chat_mode = True) -> None:
        self.llm = llm
        self.task_data = task_data
        self.chat_mode = chat_mode
        self.chinese = True if detect_language(task_data[0]) == "Chinese" else False

    def generate_reponse(self, previous_content):
        background = self.data2prompt(self.task_data)
        if self.chat_mode:
            return self.prompt_chat(background, previous_content[:-1], previous_content[-1])
        else:
            return self.prompt_pure(background, previous_content[:-1], previous_content[-1])


    def prompt_chat(self, background, previous_content, manual_pesponse):
        previous_message = []
        previous_message.append({"role": "system","content": background}) 
        if previous_content:                
            for i in range(len(previous_content)):
                if i % 2 == 1:
                    previous_message.append({"role": "assistant","content": previous_content[i]}) 
                else:
                    previous_message.append({"role": "user","content": previous_content[i]})         
            resonse,_ =  self.llm.request(manual_pesponse, None, previous_message=previous_message)
        else:
            resonse,_ =  self.llm.request(manual_pesponse, None, previous_message=previous_message)
        return resonse



    def prompt_pure(self, background, previous_content, manual_pesponse):
        if self.chinese:
            strat_word = 'Jax: 有什么可以帮助你？'
            prompt_connection = '你和Jax之前的对话:'  if previous_content else 'Jax对你说的话:'
            prompt_end = '现在请基于你们之前的对话内容，生成一个回复给Jax。\n\nYou:' if previous_content else '现在请基于Jax对你说的话生成一个回复给Jax。\n\nYou:'
        else:
            strat_word = 'Jax: what can help you?'
            prompt_connection = 'Your previous conversation with Jax:' if previous_content else 'What Jax said to you:'
            prompt_end = 'Now, based on the previous conversation, generate a reply to Jax.\n\nYou:' if previous_content else 'Now, based on what Jax said to you, generate a reply to Jax.\n\nYou:'

        
        if previous_content:
            previous_message = [] if previous_content[0] in ['Jax: 有什么可以帮助你？','Jax: what can help you?'] else [strat_word]
            div2 = 0 if previous_message else 1
            for i in range(len(previous_content)):
                if i % 2 == div2:
                    if type(previous_content[i]) == dict:
                        previous_message.append('You: ' + previous_content[i]['data']) 
                    else:
                        previous_message.append('You: ' + previous_content[i]) 
                else:
                    if type(previous_content[i]) == dict:
                        previous_message.append(previous_content[i]['data']) 
                    else:
                        previous_message.append(previous_content[i]) 
            previous_message.append(manual_pesponse)
            strat_word = '\n\n'.join([background, prompt_connection+'\n'+'\n'.join(previous_message),prompt_end])
            resonse,_ =  self.llm.request(strat_word, None)
        else:
            strat_word = '\n\n'.join([background, prompt_connection+'\n'+strat_word,prompt_end])
            resonse,_ =  self.llm.request(strat_word, None)
        if resonse.startswith('You: ') or resonse.startswith('you: '):
            resonse = resonse[5:]
        return resonse



    
        

    def data2prompt(self, content):
        if not content[0] or not content[1]:
            return ''
        if detect_language(content[0]) == "Chinese":
            start = '你是一个游戏中的{0}。你需要通过与Jax交谈了解如何基于你现有的物品和技能来完成{1}的任务。\n'.format(content[0],content[1])
            
            obj_skill_scenarios = ''
            if content[2]:
                obj_skill_scenarios = '\n你有一个背包，背包里面有下面的物品：\n' + content[2] + '\n'
            if content[3]:
                obj_skill_scenarios += '\n你拥有以下技能/动作：\n' + content[3] + '\n'
            if content[4]:
                obj_skill_scenarios += '\n此外，如果没有明确限制，你可以前往下列游戏中的场景：\n' + content[4] + '\n'
            if content[2] and content[3] and content[4]:
                end = '\n你的任务是向Jax询问如何基于这些场景以及你的物品和技能/动作完成{1}的任务。如果Jax的回答不清楚或含糊不清，你需要提出澄清问题，以便理解他的意图，从而收集足够的信息来完成任务。需要注意的是，基于上述场景，你的物品和你的技能肯定能完成{1}的任务。一旦你弄清楚如何完成任务，请对Jax说“再见”来结束你们的对话。\n记住，你是{0}，请与Jax对话，向Jax提问找到完成任务的具体步骤。'.format(content[0],content[1])
            elif content[2] and content[3]:
                end = '\n你的任务是向Jax询问如何基于你的物品和技能/动作完成{1}的任务。如果Jax的回答不清楚或含糊不清，你需要提出澄清问题，以便理解他的意图，从而收集足够的信息来完成任务。需要注意的是，基于上述你的物品和你的技能肯定能完成{1}的任务。一旦你弄清楚如何完成任务，请对Jax说“再见”来结束你们的对话。\n记住，你是{0}，请与Jax对话，向Jax提问找到完成任务的具体步骤。'.format(content[0],content[1])
            elif content[2] and content[4]:
                end = '\n你的任务是向Jax询问如何基于这些场景以及你的物品完成{1}的任务。如果Jax的回答不清楚或含糊不清，你需要提出澄清问题，以便理解他的意图，从而收集足够的信息来完成任务。需要注意的是，基于上述场景和你的物品肯定能完成{1}的任务。一旦你弄清楚如何完成任务，请对Jax说“再见”来结束你们的对话。\n记住，你是{0}，请与Jax对话，向Jax提问找到完成任务的具体步骤。'.format(content[0],content[1])
            elif content[3] and content[4]:
                end = '\n你的任务是向Jax询问如何基于这些场景以及你的技能/动作完成{1}的任务。如果Jax的回答不清楚或含糊不清，你需要提出澄清问题，以便理解他的意图，从而收集足够的信息来完成任务。需要注意的是，基于上述场景和你的技能肯定能完成{1}的任务。一旦你弄清楚如何完成任务，请对Jax说“再见”来结束你们的对话。\n记住，你是{0}，请与Jax对话，向Jax提问找到完成任务的具体步骤。'.format(content[0],content[1])
            else:
                end = '\n你的任务是向Jax询问如何基于这些场景以及你的物品和技能/动作完成{1}的任务。如果Jax的回答不清楚或含糊不清，你需要提出澄清问题，以便理解他的意图，从而收集足够的信息来完成任务。需要注意的是，基于上述场景，你的物品和你的技能肯定能完成{1}的任务。一旦你弄清楚如何完成任务，请对Jax说“再见”来结束你们的对话。\n记住，你是{0}，请与Jax对话，向Jax提问找到完成任务的具体步骤。'.format(content[0],content[1])
            
            return start + obj_skill_scenarios + end
        else:
            start = 'You are a {0} in a game. You need to talk to Jax to find out how to complete the task of {1} based on your existing items and skills.\n'.format(content[0], content[1])

            obj_skill_scenarios = ''
            if content[2]:
                obj_skill_scenarios = '\nYou have a backpack containing the following items:\n' + content[2] + '\n'
            if content[3]:
                obj_skill_scenarios += '\nYou can use/do the following skills/actions:\n' + content[3] + '\n'
            if content[4]:
                obj_skill_scenarios += '\nAdditionally, unless explicitly restricted, you can visit the following scenes in the game:\n' + content[4] + '\n'
            if content[2] and content[3] and content[4]:
                end = '\nYour task is to ask Jax how to complete the task of {1} based on these scenes, and your items and skills/actions. If Jax\'s answer is unclear or ambiguity, you need to ask clarification questions to understand his intentions and gather enough information to complete the task. Note that based on the above scenes, your items and your skills/actions will definitely allow you to complete the task of {1}. Once you figure out how to complete the task, say "goodbye" to Jax to end your conversation.\nRemember, you are a {0}, please talk to Jax and ask him questions to find the specific steps to complete the task.'.format(content[0], content[1])
            elif content[2] and content[3]:
                end = '\nYour task is to ask Jax how to complete the task of {1} based on these your items and skills/actions. If Jax\'s answer is unclear or ambiguity, you need to ask clarification questions to understand his intentions and gather enough information to complete the task. Note that based on your items and your skills/actions will definitely allow you to complete the task of {1}. Once you figure out how to complete the task, say "goodbye" to Jax to end your conversation.\nRemember, you are a {0}, please talk to Jax and ask him questions to find the specific steps to complete the task.'.format(content[0], content[1])
            elif content[2] and content[4]:
                end = '\nYour task is to ask Jax how to complete the task of {1} based on these scenes and your items. If Jax\'s answer is unclear or ambiguity, you need to ask clarification questions to understand his intentions and gather enough information to complete the task. Note that based on the above scenes and your items will definitely allow you to complete the task of {1}. Once you figure out how to complete the task, say "goodbye" to Jax to end your conversation.\nRemember, you are a {0}, please talk to Jax and ask him questions to find the specific steps to complete the task.'.format(content[0], content[1])
            elif content[3] and content[4]:
                end = '\nYour task is to ask Jax how to complete the task of {1} based on these scenes and your skills/actions. If Jax\'s answer is unclear or ambiguity, you need to ask clarification questions to understand his intentions and gather enough information to complete the task. Note that based on the above scenes and your skills/actions will definitely allow you to complete the task of {1}. Once you figure out how to complete the task, say "goodbye" to Jax to end your conversation.\nRemember, you are a {0}, please talk to Jax and ask him questions to find the specific steps to complete the task.'.format(content[0], content[1])
            else:
                end = '\nYour task is to ask Jax how to complete the task of {1} based on these scenes, and your items and skills/actions. If Jax\'s answer is unclear or ambiguity, you need to ask clarification questions to understand his intentions and gather enough information to complete the task. Note that based on the above scenes, your items and your skills/actions will definitely allow you to complete the task of {1}. Once you figure out how to complete the task, say "goodbye" to Jax to end your conversation.\nRemember, you are a {0}, please talk to Jax and ask him questions to find the specific steps to complete the task.'.format(content[0], content[1])

            return start + obj_skill_scenarios + end

if __name__ == "__main__":
    pass