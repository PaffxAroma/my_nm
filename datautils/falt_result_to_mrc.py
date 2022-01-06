import json


filepath=r'/home/lwy/Python/NLP_Projects/mrc-for-flat-nested-ner-master (2)/datautils/data_all_finished_eval.txt'
sentence_data=[]
with open(filepath, encoding='utf-8') as f:
    one_sent = []
    for line in f.readlines():
        line = line.strip()
        if line:
            sp_line=line.split('\t')
            if len(sp_line)!=2:
                continue
            else:

                one_sent.append([sp_line[0],sp_line[1]])
        else:
            sentence_data.append(one_sent)
            one_sent = []
all_sent_span=[]
for sentence in sentence_data:
    entity_span=[]
    one_span=[]
    context=''
    wordtype=''
    for i in range(len(sentence)):
        token,label=sentence[i]
        context+=token
        if label != 'O':
            labela, type = label.split('-')
        else:
            labela, type = 'O', ''
        if labela=='B':
            if len(one_span)==1:
                one_span.append(i-1)
                span_txt=''.join([x[0] for x in sentence[one_span[0]:one_span[1]+1]])
                entity_span.append([one_span,wordtype,span_txt])
                one_span=[]
            wordtype=type
            one_span.append(i)
        if labela=='O':
            if len(one_span)==1:
                one_span.append(i-1)
                span_txt = ''.join([x[0] for x in sentence[one_span[0]:one_span[1] + 1]])
                entity_span.append([one_span, wordtype, span_txt])
            wordtype=''
            one_span=[]
    all_sent_span.append([entity_span,context])

all_js=[]
for idx,sent_span in enumerate(all_sent_span):
    if idx==0:
        continue
    span_list,context=sent_span

    type_list = {'Model': [], 'Missile': [], 'method': [], 'parameter': [], 'Math': [], 'phenomenon': [], 'process': []}
    # query_list = {'Model': '表示模型，器件，设备，系统，功能结构的名词概念',
    #               'Missile': '导弹类型、导弹武器',
    #               'method': '方法，算法，方案，技术，设计方法，控制方法，仿真方法，理论',
    #               'parameter': '参数、性能、特性、误差，物理概念包含热、物理力、角度、频率等',
    #               'Math': '与数学有关的方程，原理，概念，',
    #               'phenomenon': '现象与问题包含不良现象，故障，低品质因素，损害，危害，负面影响，负作用',
    #               'process': '飞行过程，运行过程，飞行动作，操作动作'}
    # query_list = {'Model': '模型包含物理模型、数学模型、仿真模型，可视为模型的名词概念包含系统、设备、部件、器件、功能结构',
    #               'Missile': '某一类型的导弹或武器',
    #               'method': '方法，算法，技术，控制方法，建模仿真方法，理论，设计性的方案、决策、设计方法',
    #               'parameter': '导弹参数包含性能、特性、设计参数，物理参数包含热、物理力、角度、频率等，数学计算参数，控制参数',
    #               'Math': '与数学有关的方程，原理，概念',
    #               'phenomenon': '现象与问题包含不良现象，故障，低品质因素，损害，危害，负面影响，负作用',
    #               'process': '导弹发射到命中之间的状态、动作包含飞行状态、飞行动作、发射动作、制导动作'}
    query_list = {'Model': '模型包含物理模型、数学模型、仿真模型，可视为模型的名词概念包含系统、设备、部件、器件、功能结构',
                  'Missile': '某一类型的导弹或武器',
                  'method': '方法，算法，技术，控制方法，建模仿真方法，理论，设计性的方案、决策、设计方法',
                  'parameter': '参数、性能、特性、误差，物理概念包含热、物理力、角度、频率等',
                  'Math': '与数学有关的方程、原理、概念、矩阵、算子',
                  'phenomenon': '希望改善的因素包含不良现象、故障、损害、负面影响',
                  'process': '飞行状态、飞行动作、发射动作、制导动作'}
    for one_span in span_list:
        span_range, type,span_txt = one_span
        start, end = span_range
        type_list[type].append([str(start),str(end),span_txt])
    counter = 0
    for x in type_list:
        counter += 1
        span_position = []
        start_position = []
        end_position = []
        span_context=[]
        for y in type_list[x]:
            span_position.append( y[0] + ';' + y[1] )
            start_position.append(int(y[0]))
            end_position.append(int(y[1]))
            span_context.append(y[2])
        impossible = True if len(start_position) == 0 else False
        onedata = {'context': context, 'end_position': end_position, 'entity_label': x, 'impossible': impossible,
                   'qas_id': str(idx) + '.' + str(counter), 'query': query_list[x], 'span_position': span_position,
                   'start_position': start_position,'span_context':span_context}
        all_js.append(onedata)
all_data_path='/home/lwy/Python/NLP_Projects/mrc-for-flat-nested-ner-master (2)/datautils/mrc-ner.dev'
json.dump(all_js, open(all_data_path, 'w'), ensure_ascii=False, sort_keys=True, indent=2)


print('ok')