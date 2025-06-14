import pandas as pd
from pathlib import Path
import json
import datetime


# 불러오기

csv_path = Path('../dataset/preprocessed/filtered_news.csv')
df = pd.read_csv(csv_path, encoding='utf-8')


# 상수 지정

NEWS_NUMBER_PER_SOURCE = 100 # 한 언론사당 100개 기사 (총 10개 언론사)

MODEL_NAME = 'gpt-4.1-2025-04-14'

SYSTEM_INSTRUCTION = """
다음 두 가지 key elements를 포함하는 JSON 객체로 응답하세요:
"분석": 답변에 대한 근거가 되는 추론 과정.
"답변": 불리언(True/False) 답변.
""".strip()

NO_GUIDANCE = """
두 입력 텍스트가 동일한 저자에 의해 작성되었는지 검증하세요. 입력 텍스트 1: <[{title1}] {text1}>, 텍스트 2: <[{title2}] {text2}>
""".strip()

STYLE_GUIDANCE = """
두 입력 텍스트가 동일한 저자에 의해 작성되었는지 검증하세요. 주제와 내용의 차이는 무시하고 입력 텍스트의 문체를 분석하세요. 입력 텍스트 1: <[{title1}] {text1}>, 텍스트 2: <[{title2}] {text2}>
""".strip()

GRAMMER_GUIDANCE = """
두 입력 텍스트가 동일한 저자에 의해 작성되었는지 검증하세요. 저자성을 나타내는 문법적 스타일에 초점을 맞추세요. 입력 텍스트 1: <[{title1}] {text1}>, 텍스트 2: <[{title2}] {text2}>
""".strip()

LIP = """
두 입력 텍스트가 동일한 저자에 의해 작성되었는지 검증하세요. 주제와 내용의 차이는 무시하고 입력 텍스트의 문체를 분석하세요. 구동사, 조동사, 구두점, 희귀 단어, 접사, 수량 표현, 유머, 풍자, 오타, 철자 오류와 같은 언어학적 특징에 기반하여 추론하세요. 입력 텍스트 1: <[{title1}] {text1}>, 텍스트 2: <[{title2}] {text2}>
""".strip()

TEST_PROMPT = """
입력한 두 뉴스 기사가 동일한 언론사에 의해 작성되었는지 검증하세요. 주제와 내용의 차이는 무시하고 입력 텍스트의 문체 및 형식적 특징을 분석하세요. 기사 텍스트 1: <[{title1}] {text1}>, 기사 텍스트 2: <[{title2}] {text2}>
""".strip()

PROMPT = TEST_PROMPT

# 페어 생성 (같은 언론사 500개, 다른 언론사 500개)

same_pairs = [] # 같은 언론사로 짝지어진 페어들을 저장할 df 리스트

for source, member_df in df.groupby('source'):
    shuffled_df = member_df.sample(frac=1, random_state=42).reset_index(drop=True) # 섞기
    for idx in range(1, len(shuffled_df), 2):
        same_pairs.append([shuffled_df.iloc[idx-1], shuffled_df.iloc[idx]]) # 연속한 두 행을 페어로 하여 추가

diff_pairs = [] # 다른 언론사끼리 짝지어진 페어들을 저장할 df 리스트

first_list = [] # 페어의 좌측에 올 후보 (각 언론사별로 1개의 원소(DF))
second_list = [] # 페어의 우측에 올 후보 (각 언론사별로 1개의 원소(DF))

for source, member_df in df.groupby('source'):
    shuffled_df = member_df.sample(frac=1, random_state=43) # 섞기

    half_num = int(NEWS_NUMBER_PER_SOURCE/2) # 50개
    first_list.append(shuffled_df.iloc[:half_num]) # 절반은 first_list
    second_list.append(shuffled_df.iloc[half_num:]) # 절반은 second_list
    # -> 각 언론사별로 균등하게 포함되도록 하기 위함

remain_idx = [-1]*len(second_list)
second_idx_start = 0 # 5씩 증가할 예정 (second_list의 각 원소 df들이 서로 겹치지 않게 매칭되도록 하기 위함)
for i1, first_df in enumerate(first_list):
    first_idx = 0 # first_df의 인덱스
    for i2, second_df in enumerate(second_list):
        if i1 == i2: continue

        for second_idx in range(second_idx_start, second_idx_start+5): # second_df의 인덱스
            first_row = first_df.iloc[first_idx]
            second_row = second_df.iloc[second_idx]
            diff_pairs.append([first_row, second_row])

            first_idx += 1

    for i2 in range(i1+1, i1+6):
        i2 %= len(second_list)
        second_df = second_list[i2]
        first_row = first_df.iloc[first_idx]
        second_row = second_df.iloc[remain_idx[i2]]
        diff_pairs.append([first_row, second_row])

        first_idx += 1
        remain_idx[i2] -= 1

    second_idx_start += 5

print('same_pairs: ', len(same_pairs))
print('diff_pairs: ', len(diff_pairs))

# [검증 로직 추가]
from collections import Counter

pair_counts = Counter()
for pair in diff_pairs:
    source1 = pair[0]['source']
    source2 = pair[1]['source']
    # 순서에 상관없이 카운트하기 위해 (A,B)와 (B,A)를 동일하게 취급
    sorted_pair = tuple(sorted((source1, source2)))
    pair_counts[sorted_pair] += 1

print("\n[언론사 조합별 페어 수 검증]")
for pair, count in pair_counts.most_common():
    print(f"{pair}: {count}개") # 가장 균등


# jsonl 생성

custom_id_num = 0
json_list = []
for kind, pair_list in zip(['same', 'diff'], [same_pairs, diff_pairs]):
    for pair in pair_list:
        title1 = pair[0]['title'].replace('{','{{').replace('}','}}')
        text1 = pair[0]['text'].replace('{','{{').replace('}','}}')
        title2 = pair[1]['title'].replace('{','{{').replace('}','}}')
        text2 = pair[1]['text'].replace('{','{{').replace('}','}}')

        messages = []
        messages.append({
            'role':'system',
            'content':SYSTEM_INSTRUCTION
        })
        messages.append({
            'role':'user',
            'content':PROMPT.format(title1=title1, text1=text1, title2=title2, text2=text2)
        })

        now_datetime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        json_list.append({
            'custom_id':f'{kind}_source_pair_{now_datetime}_{custom_id_num:0>4}',
            'method':'POST',
            'url':'/v1/chat/completions',
            'body':{
                'model':MODEL_NAME,
                'messages':messages,
                'response_format':{
                    'type':'json_object'
                },
                'temperature':0.1
            }
        })
        custom_id_num += 1


# jsonl 저장

save_path = Path('../dataset/preprocessed')
file_name = 'batch.jsonl'
with open((save_path / file_name), 'w', encoding='utf-8') as f:
    for js in json_list:
        f.write(json.dumps(js, ensure_ascii=False)+'\n')

print('\njsonl 파일 저장 완료.')
