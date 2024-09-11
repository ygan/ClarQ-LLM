from ALL_KEYS import *
from utils.data_loader import *
from agents.provider_agent import helpers



class helpers_m(helpers):

    def unified_info(self, all_info):
        if self.chinese:
            start = "把多个回复句子组成一个字符串。需要注意的是，你可以在句子之间添加一些连词，但是，除每个句子最后一个标点符号外，每个句子的内容、标点、顺序都不可以修改。\n"

            previous_content = '\n需要合并的句子：\n' + '\n'.join(all_info)

            end_1 = "\n把多个回复句子组成一个字符串。需要注意的是，你可以在句子之间添加一些连词，但是，除每个句子最后一个标点符号外，每个句子的内容、标点、顺序都不可以修改。"
            end_5 = "返回包含一个'output'字段的JSON对象：'output'字段是字符串，内容为合并后的句子。格式如下所示：{{ 'output': '' }}。"
            end = '\n' + '\n'.join([end_1,end_5]) + '\n'  
        else:
            start = "Combine multiple response sentences into a single string. Note that you can add some conjunctions between the sentences, but you cannot modify the original sentence content, punctuation, or order of each sentence, except for the punctuation at the end of each sentence.\n"

            previous_content = '\nSentences to be combined:\n' + '\n'.join(all_info)

            end_1 = "\nCombine multiple response sentences into a single string. Note that you can add some conjunctions between the sentences, but you cannot modify the original sentence content, punctuation, or order of each sentence, except for the punctuation at the end of each sentence."
            end_5 = "Return a JSON object containing an 'output' field: the 'output' field should be a string with the combined sentences. The format is as follows: {{ 'output': '' }}."
            end = '\n' + '\n'.join([end_1, end_5]) + '\n'


        prompt = start + previous_content  + end
        response = self.prompt_pure(prompt)
        meet = True
        for s in all_info:
            if self.chinese:
                if s not in response['output'] and s[:-1] not in response['output']:
                    meet = False
                    break
            else:
                if s.lower() not in response['output'].lower() and s[:-1].lower() not in response['output'].lower():
                    meet = False
                    break
        if meet:
            return response['output']
        else:
            return_sent = ''
            if self.chinese:
                bg = ['此外，','还有，','而且，','还需要注意的是，','最后，']
            else:
                bg = [' Besides, ', ' Moreover, ', ' In addition, ', ' Furthermore, ', ' Finally, ']
            if len(bg) < len(all_info) + 1:
                pass
            else:
                for i, sent in enumerate(all_info):
                    if i != 0 and not self.chinese:
                        sent = sent[0].lower()+sent[1:]
                    if i == 0:
                        return_sent = sent
                    elif i == len(all_info) - 1 and i >= 2:
                        return_sent += bg[-1] + sent
                    else:
                        return_sent += bg[i-1] + sent
                return return_sent
            return '\n'.join(all_info)


    def predifine_info(self, previous_content, response):
        g_r = None
        analysis = []
        all_info = []
        all_info_idx = []
        for it, type_2_r in enumerate(self.gold):
            prompt = self.type2_double_check_one(previous_content, type_2_r)
            response_double_check = self.prompt_pure(prompt)
            if response_double_check['answerable']:
                all_info_idx.append(it+1)
                response['type'] = -1
                all_info.append(type_2_r)
            else:
                analysis.append(response_double_check['analysis'])

        if len(all_info) == 1:
            response['response'],g_r = all_info[0],all_info[0]
            self.gold.use(str(all_info_idx[0]))
            return response, g_r
        elif all_info:
            sorted_all_info_idx = sorted(all_info_idx, reverse=True)
            for idx in sorted_all_info_idx:
                self.gold.use(str(idx))
            g_r = self.unified_info(all_info)
            response['response'] = g_r
            return response, g_r
        else:
            for (it, type_2_r),a in zip(enumerate(self.gold),analysis):
                prompt = self.type2_double_check_one_3(previous_content, type_2_r, a)
                response_double_check = self.prompt_pure(prompt)
                if response_double_check['answerable']:
                    g_r = self.gold.use(str(it+1))
                    response['type'] = -1
                    response['response'] = g_r
                    return response, g_r
        
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