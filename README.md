
# ClarQ-LLM

This repository contains code for the paper ["ClarQ-LLM: A Benchmark for Models Clarifying and Requesting Information in Task-Oriented Dialog"](https://arxiv.org/abs/2409.06097).

If you use ClarQ-LLM in your work, please cite it as follows:
``` bibtex
@misc{gan2024clarqllmbenchmarkmodelsclarifying,
      title={ClarQ-LLM: A Benchmark for Models Clarifying and Requesting Information in Task-Oriented Dialog}, 
      author={Yujian Gan and Changling Li and Jinxia Xie and Luou Wen and Matthew Purver and Massimo Poesio},
      year={2024},
      eprint={2409.06097},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2409.06097}, 
}
```

ClarQ-LLM evaluates how well conversational agents can ask clarifying questions within task-oriented dialogues. The repository includes tasks, interaction scripts for conversational agents, and tools for model evaluation.

## Benchmark

The table below presents the benchmark results of various language models evaluated on the ClarQ-LLM benchmark for both Chinese and English task-oriented dialogue tasks. The performance metrics are defined as follows:
- **S**: Success Rate (higher is better) 
- **D**: Average Query Discrepancy (lower is better)
- **L**: Average Query Length (fewer tokens is better)

| LLMs                     | S (Chinese) | D (Chinese) | L (Chinese) | S (English) | D (English) | L (English) |
|--------------------------|-------------|-------------|-------------|-------------|-------------|-------------|
| L3.1-405B-Inst  | -           | -           | -           | 0.605       | 0.473       | 118         |
| GPT-4o        | 0.508       | 0.215       | 129.9       | 0.485       | 0.492       | 70.5        |
| GPT-4         | 0.258       | -0.72       | 94.3        | 0.296       | -0.56       | 76.4        |
| ERNIE 4.0     | 0.315       | -0.82       | 141.0       | -           | -           | -           |
| GPT-3.5       | 0           | -4.21       | 92.0        | 0.02        | -2.79       | 52.6        |

Feel free to push your results and related papers to this benchmark.



## Getting Started

### 1. Setup API Keys
Before running any scripts, you need to replace the placeholder API keys in `ALL_KEYS.py` with your own API keys.

### 2. Run Pre-Cached Interactions
We have pre-cached interaction data (saved in the `log/` directory) for GPT-4o-based interactions. Even if you don’t have an API key, you can still run the following commands to replicate the results:

```bash
sh l2l_GPT4o_seeker_GPT4o_multi_info_provider.sh
```

```bash
sh l2l_GPT4o_seeker_GPT4o_provider.sh
```

The results will be saved in the `results/` directory.

### 3. Customize and Debug
You can step through the code and debug interactions between GPT-4o agents by inspecting cached prompts and responses. The caching mechanism ensures that repeated prompts return identical responses to facilitate reproducibility and reduce API costs.

Note: In earlier versions, the terms "helper" and "player" were used for provider and seeker, respectively, so you might encounter these terms in the code.

### 4. LLAMA3.1 Interactions
We do not provide pre-cached results for LLAMA3.1 interactions. To run LLAMA3.1 interactions, update the API keys in `ALL_KEYS.py` and execute the following commands:

```bash
sh l2l_GPT4o_seeker_LLAMA3.1_provider.sh
sh l2l_LLAMA3.1_seeker_GPT4o_provider.sh
```

### 5. Evaluation
You can evaluate the performance of the models using pre-cached results with the following command:

```bash
sh evaluation.sh
```

This will evaluate the six pre-stored L2L (LLM interact with LLM) results. Evaluation is based on GPT-4o, and the results are cached for easy reproduction. If you wish to evaluate new data, you will need valid API keys. To use LLAMA3.1-405B for evaluation, modify `evaluation.sh` by replacing `gpt4o` with `llama3.1-405b`.

### 6. Dataset Split
The dataset is divided into 31 files, each containing 10 tasks:
- **Test Set**: Files 1–26
- **Development Set**: Files 25–31

All files are stored under the `data/` directory:
- `data/Chinese/` for Chinese tasks
- `data/English/` for English tasks

### 7. Running Interactions (H2L Setup)
After setting up the keys, you can interact with the agents using the following commands:

#### Chinese Task
```bash
python l2l.py --provider_agent_llm gpt4o --task_data_path data/Chinese --play_around
```

#### English Task
```bash
python l2l.py --provider_agent_llm gpt4o --task_data_path data/English --play_around
```

### 8. Developing Your Own Seeker Agent

To develop your own seeker Agent, you can extend the `player` class in the `agents/seeker_agent.py` file and implement your custom `generate_response` method. Once the development is complete, simply replace the default Seeker Agent in the `l2l.py` script with your customized agent. Here’s a step-by-step guide:

1. **Extend the Player Class:**  
   Open the `agents/seeker_agent.py` file and locate the `player` class. Create a new class that inherits from `player`, and override the `generate_response` method.

2. **Implement the generate_response Method:**  
   Inside the `generate_response` method, implement your logic for generating the seeker agent's responses. 

3. **Replace the Seeker Agent in the l2l Script:**  
   After developing your seeker agent, update the `l2l.py` script by replacing the default seeker agent with your customized version.


## Dependencies
To install required packages, run:

```bash
pip install -r requirements.txt
```


## Example Conversations

### Chinese Conversation Example：

**Conversation Background:**

你是一个游戏中的冒险者。你需要通过与Jax交谈了解如何基于你现有的物品和技能来完成建造一个临时避难所以抵御即将到来的风暴的任务。

你有一个背包，背包里面有以下物品：
- 深色的木材
- 浅色的木材
- 白色的木材
- 轻型木材

你拥有以下技能/动作：
- 建造木制避难所的技能

此外，如果没有明确限制，你可以前往下列游戏中的场景：
- 东海岸
- 西海岸
- 南海岸

你的任务是向Jax询问如何基于这些场景以及你的物品和技能/动作完成建造一个临时避难所以抵御即将到来的风暴的任务。如果Jax的回答不清楚或含糊不清，你需要提出澄清问题，以便理解他的意图，从而收集足够的信息来完成任务。记住，你是冒险者，请与Jax对话，找到完成任务的具体步骤。

**Example Conversation:**

Jax: 有什么可以帮助你？  
冒险者: 风暴即将来临，我需要建造一个避难所，你能给我一些建议吗？  
Jax: 你可以先收集一些绳子，配合坚硬的木材在相对安全的海岸建造避难所。  
冒险者: 请问我应该去哪里收集绳子？  
Jax: 你可以在西海岸找到绳子。  
冒险者: 请问哪种木材比较坚硬？  
Jax: 你应该使用白色的木材，因为这些木材的硬度较高，可以保证避难所不容易被风暴破坏。
冒险者: 请问哪个海岸相对安全且适合建设避难所？  
Jax: 这个避难所应该建在南海岸，因为南海岸有山体可以帮忙阻挡和减轻风暴对避难所的影响。



### English Conversation Example：

**Conversation Background:**

You are an adventurer in a game. You need to talk to Jax to find out how to complete the task of building a shelter to withstand the impending storm based on your existing items and skills.

You have a backpack containing the following items:
- Dark wood
- Light-colored wood
- White wood
- Lightweight wood

You can use/do the following skills/actions:
- Wooden shelter construction

Additionally, unless explicitly restricted, you can visit the following scenes in the game:
- East Coast
- West Coast
- South Coast

Your task is to ask Jax how to complete the task of building a shelter to withstand the impending storm based on these scenes, and your items and skills/actions. If Jax's answer is unclear, you need to ask clarifying questions to gather enough information. Once you figure out how to complete the task, say "goodbye" to Jax to end the conversation.

**Example Conversation:**

Jax: What can I help you with?  
Adventurer: I need to build a shelter to withstand the impending storm. How should I proceed?  
Jax: You can start by collecting some ropes and pairing them with hard wood to build a shelter on one of the safer coastlines.  
Adventurer: Which of the woods would be hard wood?  
Jax: White wood is the hardest and heaviest, so using that to construct the shelter will ensure it's not easily destroyed by the storm.  
Adventurer: Where can I find ropes?  
Jax: You can find ropes on the west coast.  
Adventurer: Which coastline would be safe for a shelter?  
Jax: The shelter should be built on the south coast. The mountains there can act as an additional shield from the storm.



## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.
