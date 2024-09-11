# ============ English ============


# The GPT-4o seeker interacts with the GPT-4o provider in non-chat mode.
python l2l.py --seeker_agent_llm gpt4o --provider_agent_llm gpt4o --task_data_path data/English 

# The GPT-4o seeker interacts with the GPT-4o provider in chat mode.
python l2l.py --seeker_agent_llm gpt4o --provider_agent_llm gpt4o --task_data_path data/English --player_chat_mode



# ============ Chinese ============


# The GPT-4o seeker interacts with the GPT-4o provider in non-chat mode.
python l2l.py --seeker_agent_llm gpt4o --provider_agent_llm gpt4o --task_data_path data/Chinese 

# The GPT-4o seeker interacts with the GPT-4o provider in chat mode.
python l2l.py --seeker_agent_llm gpt4o --provider_agent_llm gpt4o --task_data_path data/Chinese --player_chat_mode