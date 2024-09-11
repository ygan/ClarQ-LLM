from agents.seeker_agent import player
from agents.provider_agent import helpers as general_provider
from agents.multi_info_provider_agent import helpers_m as multi_info_provider
from ALL_KEYS import *
from utils.data_loader import *
from utils.llm import ChatGPT,QianFan,LLAMA,AWSBedrockLLAMA
import argparse


def evaluate_player(task_data_path, output_path, player_llm, player_chat_mode, provider_constructor, provider_llm):
    all_conv = data_combination(read_path(task_data_path))
    evaluation_set = [i for i in range(26)]
    evaluate_results = []

    for i,one_type in enumerate(all_conv):
        if i not in evaluation_set:
            continue
        evaluate_results.append([])
        for j,conv in enumerate(one_type):

            print("{0}.{1}".format(i+1,j))
            evaluate_results[i].append([])

            gold_r = conv['all_response'].strip().split('\n')
            h = provider_constructor(gold_r, conv['background_splitted'], conv['gold_structure'], conv, provider_llm)
            p = player(conv['background_splitted'], player_llm, player_chat_mode)
            l2l_conv = []
            while True:
                l2l_conv.append(h.generate_reponse(l2l_conv))
                l2l_conv.append(p.generate_reponse(l2l_conv))
                if h.is_conv_end(l2l_conv) or len(l2l_conv) > 22:
                    break
            for c in l2l_conv:
                print(c)
                print()
            conv['l2l'][0] = l2l_conv
            print()
            print()
            print()
        if i == 26 - 1:
            break
    with open(output_path, "w") as json_file:
        json.dump(all_conv, json_file, ensure_ascii=False, indent=2)

    


def test_helper(task_data_path, provider_constructor, provider_llm):
    all_conv = data_combination(read_path(task_data_path))
    evaluation_set = [i for i in range(31)]
    evaluate_results = []

    for i, one_type in enumerate(all_conv):
        if i not in evaluation_set:
            continue
        evaluate_results.append([])
        for j, conv in enumerate(one_type):

            print("{0}.{1}".format(i+1, j))
            evaluate_results[i].append([])
            

            # if i != 1 - 1 or j != 1:
            #     continue

            print('------------------------------')
            print(conv['background'])
            print('------------------------------')
            print()
            gold_r = conv['all_response'].strip().split('\n')
            h = provider_constructor(gold_r, conv['background_splitted'], conv['gold_structure'], conv, provider_llm)

            l2l_conv = []
            while True:
                l2l_conv.append(h.generate_reponse(l2l_conv))
                for c in l2l_conv:
                    print(c)
                    print()
                user_input = input("Enter your response: ")  # User inputs the response
                l2l_conv.append(user_input)  # User input is appended instead of p.generate_response

                if h.is_conv_end(l2l_conv) or len(l2l_conv) > 24:
                    break
            print()
            print()
            print()
            exit()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run language model evaluation.')
    parser.add_argument('--seeker_agent_llm', type=str, default='gpt4o')
    parser.add_argument('--provider_agent_llm', type=str, default='gpt4o')
    parser.add_argument('--task_data_path', type=str, default='data/English', help='Path to task data')
    
    parser.add_argument('--player_chat_mode', action='store_true', help='Enable player chat mode')
    parser.add_argument('--multi_info_provider_agent', action='store_true', help='Use a multiple info provider agent instead of a general provider agent.')
    parser.add_argument('--play_around', action='store_true')
    
    args = parser.parse_args()
    
    player_chat_mode = args.player_chat_mode
    task_data_path = args.task_data_path

    provider_agent_constructor = multi_info_provider if args.multi_info_provider_agent else general_provider
    language = 'En' if task_data_path == 'data/English' else 'Ch'
    mode = 'Chat' if player_chat_mode else 'Comp'

    if args.play_around:
        test_helper(task_data_path, provider_agent_constructor, args.provider_agent_llm)
        exit()
    
    if args.seeker_agent_llm == "gpt4o":
        player_llm = ChatGPT("gpt-4o-2024-05-13", 'log/gpt4o_plyaer_cache.pkl')
        output_path = "results/l2l_gpt4o.{}.{}.json".format(mode,language)
    elif args.seeker_agent_llm == 'qianfan':
        player_llm = QianFan("ERNIE-Bot-4", 'log/qianfan_plyaer_cache.pkl')
        output_path = "results/l2l_qianfan.{}.{}.json".format(mode,language)
    elif args.seeker_agent_llm == 'llama':
        if player_chat_mode:
            player_llm = LLAMA("[your-llama-model-path] max_new_tokens:150", 'log/llama2_plyaer_cache.pkl')
        else:
            player_llm = LLAMA("[your-llama-model-path] max_new_tokens:150", 'log/llama2_plyaer_cache.pkl')
        output_path = "results/l2l_llama.{}.{}.json".format(mode,language)
    elif args.seeker_agent_llm  == 'llama3.1-405B':
        player_llm = AWSBedrockLLAMA("llama3.1-405B", 'log/llm_player_cache_llama3.1-405B.pkl')
        output_path = "results/l2l_llama3.1-405B.{}.{}.json".format(mode,language)
    else:
        player_llm = ChatGPT("gpt-3.5-turbo-0125", 'log/gpt3_plyaer_cache.pkl')
        output_path = "results/l2l_gpt3.5.{}.{}.json".format(mode,language)

    evaluate_player(task_data_path, output_path, player_llm, player_chat_mode,provider_agent_constructor,args.provider_agent_llm)


