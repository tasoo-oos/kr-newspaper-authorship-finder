import pandas as pd
import json
from pathlib import Path

dataset_path = Path('../dataset')

dic = {
    'source':[],
    'title':[],
    'text':[]
}

except_count = 0

# 데이터 추출
for json_path in dataset_path.iterdir():
    with json_path.open('r', encoding='utf-8') as f:
        try:
            json_file = json.load(f)
            for instance in json_file.get('data'):
                dic['source'].append(instance['doc_source'])
                dic['title'].append(instance['doc_title'])
                paragraph = instance['paragraphs']
                if len(paragraph) > 1:
                    print('길이 2개 넘음!!')
                    print(paragraph)
                    print()
                dic['text'].append(paragraph[0].get('context',''))
        except:
            except_count +=1

# 데이터프레임으로 변환
df = pd.DataFrame(dic)

# 저장
Path('../dataset/preprocessed').mkdir(parents=True, exist_ok=True)
df.to_csv('../dataset/preprocessed/parsed_news.csv', encoding='utf-8-sig', index_label='id')

print('except_count:', except_count)