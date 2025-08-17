import os,json

def read_path(name):
    types = [
        '1. Gather Resources',
        '2. Escort Mission',
        '3. Stealth Mission',
        '4. Survival Challenge',
        '5. Construction Task',
        '6. Defense Mission',
        '7. Competition',
        '8. Weapon Manufacturing',
        '9. Hunting Expedition',
        '10. Rescue Mission',
        '11. Arena Battle',
        '12. Scientific Experiment',
        '13. Photography Mission',
        '14. Trade Task',
        '15. Exploration Journey',
        '16. Electrical Engineering',
        '17. Automobile Manufacturing',
        '18. Painting Task',
        '19. Repair Mission',
        '20. Training Session',
        '21. Digging Mission',
        '22. Electronic Engineering',
        '23. Planting Task',
        '24. Alliance Building',
        '25. Cooking Delicacies',
        '26. Video Production',
        '27. Animal Care',
        '28. Archaeological Excavation',
        '29. Escape Mission',
        '30. Planning Tourism',
        '31. Magic Task'
    ]
    all_data = []
    dir_path = name
    if not os.path.exists(dir_path):
        return all_data
    for t in types:
        t = t.replace(' ','_') + '.json'
        json_path = os.path.join(dir_path,t) 
        with open(json_path, 'r') as f:
            all_data.append(json.load(f)) 
    return all_data


def read_one_json(one_type):
    new_t = []
    for j,d in enumerate(one_type):
        half_len_d = int(len(one_type) / 2)
        if j >= half_len_d:
            break
        d2 = one_type[half_len_d+j]
        d['h2l'] = [d['h2l'], d2['h2l']]
        d['h2l_check'] = [d['h2l_check'], d2['h2l_check']]
        d['l2l'] = [d['l2l'], d2['l2l']]
        d['l2l_check'] = [d['l2l_check'], d2['l2l_check']]

        golds = d['all_response'].split('\n')
        d['gold_structure'] = []
        if golds and golds[-1][0].isdigit():
            half = int(len(golds)/2)
            assert golds[half][0].isdigit()
            d['all_response'] = '\n'.join(golds[:half]) 
            d['gold_structure'] = golds[half:]
            for gs in d['gold_structure']:
                if '.' in gs:
                    assert '.'.join(gs.split('.')[:-1]) in d['gold_structure']

        new_t.append(d)
    return new_t

def data_combination(all_data):
    data_after_combine = []
    for i,one_type in enumerate(all_data):
        new_t = read_one_json(one_type)  
        data_after_combine.append(new_t)
    return data_after_combine
            


