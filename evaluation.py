from ALL_KEYS import *
from utils.data_loader import *
from utils.llm import ChatGPT, AWSBedrockLLAMA
from utils.utils import detect_language
import sys

def data2prompt_mini(gold, glod_explain, predict):
    def add_punctuation(sentence, Chinese = False):
        if sentence and sentence[-1] not in ['.','?','!',',','。','？','！','，']:
            if Chinese:
                return sentence + '。'
            return sentence + '.'
        return sentence

    if detect_language(gold[0]) == "Chinese":
        g_name = '正确信息及其信息用途'
        p_name = '生成信息'
        start = "下面展示了两段文字。第一段是：{0}，第二段是：{1}。第一段的信息用途可以帮助你理解正确信息的用途。{1}缺少这些用途的说明，你需要自行分析{1}的用途，并判断其是否包含了正确信息的用途。\n判断方法是先分析{1}的用途，然后检查是否可以在{1}中找到与正确信息用途一致的内容。如果正确信息的用途能在{1}中的某一条信息中找到相同的对应项，我们就认为比对成功。注意：正确信息通常用于澄清特定内容，如果{1}只是列举了多个可能的场景、物品、人物、技能等名称，那么{1}可能需要依赖正确信息进行澄清，从而可能导致不匹配的情况。".format(g_name,p_name)

        numbered_gold = [f'{i+1}. {add_punctuation(line, True)}\t\t\t\t信息用途：{add_punctuation(line_ex, True)}' for (i, line), line_ex in zip(enumerate(gold), glod_explain)]
        numbered_predict = [f'{i+1}. {add_punctuation(line, True)}' for i, line in enumerate(predict)]

        middle = '{0}：\n{1}\n\n{2}：\n{3}'.format(g_name,'\n'.join(numbered_gold),p_name,'\n'.join(numbered_predict))
        end = "仔细判断第一段{0}的每一行是否都在第二段{1}中被提及？即检查是否可以在{1}中找到与正确信息用途一致的内容。如果正确信息的用途能在{1}中的某一条信息中找到相同的对应项，我们就认为比对成功。注意：正确信息通常用于澄清特定内容，如果{1}只是列举了多个可能的场景、物品、人物、技能等名称，那么{1}可能需要依赖正确信息进行澄清，从而可能导致不匹配的情况。\n返回包含两个字段的JSON对象：一个'analysis'字段，它的值是你对此的分析字符串，以及一个'match'字段，他的值是布尔类型，当比对成功时为True。格式应如下所示：{{ 'analysis': '您的分析内容', 'match': True 或 False }}。".format(g_name,p_name)

        return '\n\n'.join([start, middle, end])
    else:

        g_name = 'Gold Information and its Purpose'
        p_name = 'Generated Information'
        start = "Below are two passages of text. The first is: {0}, and the second is: {1}. The purpose of the information in the first passage can help you understand the gold information. Since {1} lacks an explanation of these purposes, you need to analyze the purpose of {1} on your own and determine whether it includes the purpose of the gold information.\nThe method is to first analyze the purpose of {1}, then check whether you can find content in {1} that matches the purpose of the gold information. If the purpose of the gold information can be found in any part of {1}, we consider the comparison successful. Note: Gold information is typically used to clarify specific content. If {1} merely lists multiple possible scenes, objects, characters, or skill names, the {1} may need to rely on gold information for clarification, which could lead to mismatches.".format(g_name, p_name)


        numbered_gold = [f'{i+1}. {add_punctuation(line)}\t\t\t\tPurpose of Information: {add_punctuation(line_ex)}' for (i, line), line_ex in zip(enumerate(gold), glod_explain)]
        numbered_predict = [f'{i+1}. {add_punctuation(line)}' for i, line in enumerate(predict)]

        middle = '{0}：\n{1}\n\n{2}：\n{3}'.format(g_name,'\n'.join(numbered_gold),p_name,'\n'.join(numbered_predict))

        end = "Carefully determine whether each line from the {0} passage is mentioned in the {1} passage. Check whether you can find content in {1} that matches the purpose of the gold information. If the purpose of the gold information can be found in any part of {1}, we consider the comparison successful. Note: Gold information is typically used to clarify specific content. If {1} merely lists multiple possible scenes, objects, characters, or skill names, the {1} may need to rely on gold information for clarification, which could lead to mismatches.\nReturn a JSON object containing two fields: an 'analysis' field, whose value is your analytical string about this, and a 'match' field, which is a Boolean indicating whether comparison was successful. The format should be as follows: {{ 'analysis': 'Your analysis content', 'match': True or False }}.".format(g_name, p_name)

        return '\n\n'.join([start, middle, end])


def evaluate_one_multi(gold, gold_explain, predict, llm):
    gold = [s[4:].strip() if s.lower().startswith("jax:") else s.strip() for s in gold[1:]]
    predict = [s[4:].strip() if s.lower().startswith("jax:") else s.strip() for s in predict]
    easy_check = True if gold == predict else False
    if detect_language(gold[0]) != "Chinese":
        gold = [g.lower() for g in gold]
        predict = [g.lower() for g in predict]
    if easy_check:
        return 1
    else:
        easy_check = True
        for g in gold:
            if g not in predict:
                easy_check = False
                break
        
    if easy_check:
        return 1
    else:
        gold_cleaned = [sentence.rstrip('，。？！,.?!') for sentence in gold]
        predict_cleaned = [sentence.rstrip('，。？！,.?!') for sentence in predict]

        gold_diff = []
        predict_diff = []

        contained_gold = set()
        contained_gold_explain = dict()
        for p in predict_cleaned:
            for g in gold_cleaned:
                if g in p:
                    contained_gold.add(g)
                    contained_gold.add(p)
        for i_g, g in enumerate(gold_cleaned):
            contained_gold_explain[g] = gold_explain[i_g]

        gold_diff = [g for g in gold_cleaned if g not in contained_gold]
        predict_diff = [p for p in predict_cleaned if p not in contained_gold]
        glod_explain_diff = [contained_gold_explain[g] for g in gold_diff]

        if not gold_diff:
            return 1
        elif not predict_diff:
            return 0
        
        for gd, gde in zip(gold_diff, glod_explain_diff):
            prompt = data2prompt_mini([gd], [gde], predict_diff)
            resonse,_ =  llm.request(prompt, None, previous_message=None, json_format = True)
            try:
                result_dict = json.loads(resonse)
            except:
                assert False
            if 'match' in result_dict and not result_dict['match']:
                break

        if 'match' in result_dict and result_dict['match']:
            return 1
        
    return 0


def evaluate_l2l_doc():
    llm_name = sys.argv[1]
    json_file = sys.argv[2]

    if llm_name == 'llama3.1-405b':
        llm = AWSBedrockLLAMA("llama3.1-405b", 'log/llama3.1_evaluator_cache.pkl')
    else:
        llm = ChatGPT("gpt-4o-2024-05-13", 'log/llm_evaluator_cache.pkl')

    with open(json_file, 'r') as f:
        all_conv = json.load(f)

    evaluation_set = [i for i in range(26)]
    evaluate_results = []
    AQD_evaluation_results = []
    ARL_evaluation_results = []

    for i,one_type in enumerate(all_conv):
        if i not in evaluation_set:
            continue
        evaluate_results.append([])
        AQD_evaluation_results.append([])
        ARL_evaluation_results.append([])

        for j,conv in enumerate(one_type):
            evaluate_results[i].append([])
            AQD_evaluation_results[i].append([])
            ARL_evaluation_results[i].append([])

            gold_r = conv['all_response'].strip().split('\n')
            for h2l in conv['l2l']:
                if not h2l:
                    continue
                helper_response = []
                seeker_reponse = []
                for k,sent in enumerate(h2l[1:]):
                    if k % 2 == 1 and k != 1:
                        helper_response.append(sent)
                    elif k % 2 == 0:# and k != 0 and k!=len(h2l[1:])-1:
                        seeker_reponse.append(sent.strip())
                evaluate_results[i][j].append(evaluate_one_multi(gold_r, conv['all_response_exaplain'], helper_response, llm))
                AQD_evaluation_results[i][j].append(len(helper_response) +1 - len(gold_r))
                if detect_language(seeker_reponse[0]) != "Chinese":
                    ARL_evaluation_results[i][j].append(sum([s.count(' ') for s in seeker_reponse])/len(seeker_reponse))
                else:
                    ARL_evaluation_results[i][j].append(sum([len(s) for s in seeker_reponse])/len(seeker_reponse))

        print(evaluate_results[-1])
    sum_arrays = sum(sum(sum(inner) for inner in outer) for outer in evaluate_results)
    print("Success Rate: {}".format(sum_arrays/(10*len(evaluation_set))))

    sum_arrays = sum(sum(sum(inner) for inner in outer) for outer in AQD_evaluation_results)
    print("Average Query Discrepancy (AQD): {}".format(sum_arrays/(10*len(evaluation_set))))

    sum_arrays = sum(sum(sum(inner) for inner in outer) for outer in ARL_evaluation_results)
    print("Average Query Length (ARL): {}".format(sum_arrays/(10*len(evaluation_set))))




if __name__ == "__main__":
    evaluate_l2l_doc()