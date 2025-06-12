import pandas as pd

# 기준
min_length = 501
max_length = 1000

# parse.py로 생성한 csv 파일 불러오기
df = pd.read_csv('../dataset/preprocessed/parsed_news.csv', index_col=0)

# 길이대로 자르기 (500자 초과 1000자 이하)
df['length'] = df['text'].apply(len)
df_filtered = df[(min_length<=df['length']) & (df['length']<=max_length)]

# 기사 수 상위 10개의 언론사만 추출
top10 = df_filtered.groupby('source').count().sort_values(by='text', ascending=False).iloc[:10]
print('상위 10개 언론사')
for _, (i,c) in enumerate(zip(top10.index,top10['title']), start=1):
    print(f'{_}. {i} ({c}개)')
print('-'*50)

df_top10_filtered = df_filtered[df_filtered['source'].isin(top10.index)]
df_top10_filtered = df_top10_filtered.reset_index(drop=True) # 인덱스 0부터 재지정

# 저장 (인덱스를 id 컬럼으로 저장)
df_top10_filtered.to_csv('../dataset/preprocessed/filtered_news.csv', encoding='utf-8-sig', index_label='id')
