import os
import json


def process_one_type(folder_path):

    one_folder=[]
    c1=1
    for i in range(200):
        type_list = {'Model': [], 'Missile': [], 'method': [], 'parameter': [], 'Math': [], 'phenomenon': [],
                     'process': []}
        query_list = {'Model': '模型、器件、系统和结构', 'Missile': '导弹类型', 'method': '方法、算法、方案、技术、设计方法、控制方法', 'parameter': '参数、性能、特性',
                      'Math': '数学方程、原理、概念', 'phenomenon': '不良现象',
                      'process': '飞行过程、运行过程'}
        filepath=folder_path+'/data'+str(i)+'.ann'
        txtpath=folder_path+'/data'+str(i)+'.txt'
        if os.path.exists(txtpath):

            txt_data=[]
            annotion_data=[]
            with open(txtpath, encoding='utf-8') as f:
                for line in f.readlines():
                    line=line.strip()
                    txt_data.append(line)
                    #print(line)
            if len(txt_data)!=3:
                continue
            context=txt_data[-1]
            lenc=len(context)
            span_text=[]
            # context_added=context.split(' ')
            # context_add=''.join(context_added)
            # empty_positions=[]
            # if len(context_added)>1:
            #     empty_position=0
            #     for x in context_added[0:-1]:
            #         empty_position=empty_position+len(x)+1
            #         empty_positions.append(empty_position)
            #print(context_added)
            # filepath = txtpath[:-4] + '.ann'
            with open(filepath, encoding='utf-8') as p:
                for line in p.readlines():
                    line=line.strip()
                    annotion_data.append(line)
            if context[0:8]=='多目标超视距空战':
                print('found')
            for line in annotion_data:
                id,mess,text=line.split('\t')
                type, start, end=mess.split(' ')
                headlen=len(txt_data[0])+len(txt_data[1])+2
                start=int(start)-headlen
                end=int(end)-headlen-1
                # for pos in empty_positions:
                #     if start>pos:
                #         start-=1
                #     if end>pos:
                #         end-=1
                assert text==context[start:end+1],text+'---'+context[start:end+1]
                span_text.append(text)
                if int(end)>=lenc:
                    print('error')
                type_list[type].append([str(start),str(end)])


            counter=0
            for x in type_list:
                counter+=1
                span_position=[]
                start_position=[]
                end_position=[]
                for y in type_list[x]:
                    span_position.append(y[0]+';'+y[1])
                    start_position.append(int(y[0]))
                    end_position.append(int(y[1]))
                impossible=True if len(start_position)==0 else False
                onedata={'context':context,'end_position':end_position,'entity_label':x,'impossible':impossible,'qas_id':str(c1)+'.'+str(counter),'query':query_list[x],'span_position':span_position,'start_position':start_position,'span_text':span_text}
                one_folder.append(onedata)
            c1+=1
    return one_folder


if __name__ == '__main__':
    paper_path=r'data/paper_data'
    paper_type_list=os.listdir(paper_path)
    all_data=[]
    c1=1
    all_data_path=paper_path+'/mrc_data_all.json'
    for paper_type in paper_type_list:
        if paper_type == '.stats_cache' or paper_type=='mrc_data_all.json': continue
        folder_path=os.path.join(paper_path,paper_type,'annotion_data')
        one_folder=process_one_type(folder_path)
        output_path=os.path.join(paper_path,paper_type)+'/mrc_data.json'
        json.dump(one_folder,open(output_path,'w'),ensure_ascii=False,sort_keys=True,indent=2)
        for x in one_folder:
            id=x['qas_id']
            x['qas_id']=str(c1)+'.'+id.split('.')[1]
            all_data.append(x)
            c1+=1
    json.dump(all_data, open(all_data_path, 'w'), ensure_ascii=False, sort_keys=True, indent=2)
    print('finished')