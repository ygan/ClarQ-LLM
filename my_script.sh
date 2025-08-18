
# script for testing custom llms
# python3 l2l.py --seeker_agent_llm gpt4o --provider_agent_llm gpt4o --task_data_path data/English --multi_info_provider_agent --player_chat_mode

# script that works
python3 l2l.py --provider_agent_llm deepseek-ai/DeepSeek-V3:fireworks-ai --task_data_path data/English --play_around

# python3 l2l.py --provider_agent_llm deepseek-ai/DeepSeek-R1-Distill-Llama-8B --task_data_path data/English --play_around