from datetime import datetime
import logging


logging.basicConfig(filename='log/'+str(datetime.now().date())+'.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def readable_log(message):
    with open('log/'+str(datetime.now().date())+'.readable', 'a') as file:
        file.write('========================\n')
        file.write('========================\n')
        file.write('========================\n')
        file.write(message)
        file.write('\n========================\n')
        file.write('========================\n')
        file.write('========================\n\n\n')



def log(message):
    logging.info(message)


def llm_log(input, output, **kwargs):
    content = "\t\tLLM input:\n"
    if type(input) == list:
        for i in input:
            content += "{0}:\n{1}\n\n".format(i['role'], i['content'])
    else:
        content = input
    content += "\n\t\tLLM output:\n{0}".format(output)
    if "gold" in kwargs:
        content += "\n\n\t\tExpect Gold Output:\n{0}".format(kwargs['gold'])
    if "model" in kwargs:
        content += "\n\n\t\tLLM Model:\t{0}".format(kwargs['model'])
    if "system_fingerprint" in kwargs:
        content += "\n\n\t\tSystem Fingerprint:\t{0}".format(kwargs['system_fingerprint'])
    if "completion_tokens" in kwargs:
        content += "\n\n\t\tCompletion Tokens:\t{0}".format(kwargs['completion_tokens'])
    if "prompt_tokens" in kwargs:
        content += "\n\n\t\tPrompt Tokens:\t{0}".format(kwargs['prompt_tokens'])
    if "total_tokens" in kwargs:
        content += "\n\n\t\tTotal Tokens:\t{0}".format(kwargs['total_tokens'])
    if "usage" in kwargs:
        content += "\n\n\t\tUsage(Prompt, Completion, Total Tokens):\t{0}".format(kwargs['usage'])
             
    readable_log(content)