import os
import json
import re

def detect_language(text):
    chinese_characters = re.findall(r'[\u4e00-\u9fa5]', text)
    
    english_char_count = sum(1 for char in text if char.isalpha() and char.isascii())
    chinese_char_count = len(chinese_characters)
    
    if english_char_count > chinese_char_count:
        return "English"
    elif chinese_char_count > english_char_count:
        return "Chinese"
    else:
        return "Chinese"

def get_gold_structure(one_type, bjx_path):
    with open(bjx_path, 'r') as f:
        data2 = json.load(f)
    new_t = []
    for (j,d),d2 in zip(enumerate(one_type), data2):    
        golds = d['all_response'].split('\n')
        d['gold_structure'] = []
        if golds and len(golds) > 2 and golds[-1][0].isdigit() :
            half = int(len(golds)/2)
            assert golds[half][0].isdigit()
            d['all_response2'] = '\n'.join(golds[:half]) 
            d['gold_structure'] = golds[half:]
            for gs in d['gold_structure']:
                if '.' in gs:
                    assert '.'.join(gs.split('.')[:-1]) in d['gold_structure']
        else:
            d['all_response2'] = '\n'.join(golds) 
            golds_bjx = d2['all_response'].split('\n')
            if golds_bjx and len(golds_bjx) > 2 and golds_bjx[-1][0].isdigit() :
                half = int(len(golds_bjx)/2)
                assert golds_bjx[half][0].isdigit()
                d['gold_structure'] = golds_bjx[half:]
                for gs in d['gold_structure']:
                    if '.' in gs:
                        assert '.'.join(gs.split('.')[:-1]) in d['gold_structure']
        new_t.append(d)
    return new_t