from typing import Tuple

import pandas as pd
from pathlib import Path
import json
import datetime

import argparse

from sklearn.metrics import confusion_matrix

# 상수 지정

NEWS_NUMBER_PER_SOURCE = 100 # 한 언론사당 100개 기사 (총 10개 언론사)

MODEL_NAME = 'gpt-4.1-2025-04-14'

SYSTEM_INSTRUCTION_V1 = """
다음 두 가지 key elements를 포함하는 JSON 객체로 응답하세요:
"분석": 답변에 대한 근거가 되는 추론 과정.
"답변": 불리언(True/False) 답변.
"""

SYSTEM_INSTRUCTION_V2 = """
당신은 뉴스 기사의 저자성을 검증하는 AI입니다. 주어진 두 뉴스 기사가 동일한 언론사에 의해 작성되었는지 판단해야 합니다.

다음 지침을 기반으로, 두 언론 기사가 동일한 언론사에 의해 작성되었는지 검증하세요.
- 기사의 주제(예: 정치, 연예, 경제)나 내용에 따른 자연스러운 문체 차이는 무시하세요.
- 대신, 주제와 상관없이 일관되게 나타나는 언론사 특유의 문체 및 형식적 특징 등의 '편집 스타일'에 집중하세요.

다음 두 가지 key elements를 포함하는 JSON 객체로 응답하세요:
"분석": 답변에 대한 근거가 되는 추론 과정.
"답변": 두 뉴스 기사가 동일한 언론사에 의해 작성되었는지에 대한 불리언(True/False) 답변.
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

TEST_PROMPT_V2 = """
입력한 두 뉴스 기사가 동일한 언론사에 의해 작성되었는지 검증하세요.

**지침:**
1.  기사의 주제(예: 정치, 연예, 경제)나 내용에 따른 자연스러운 문체 차이는 무시하세요.
2.  대신, 주제와 상관없이 일관되게 나타나는 언론사 고유의 '편집 스타일'에 집중하세요.
3.  아래 항목들을 중심으로 두 기사의 공통점과 차이점을 분석하세요.
    *   **형식적 특징:**
        -   제목에 대괄호 `[ ]`나 말머리(예: [포토], [단독])를 사용하는 방식
        -   기자 이름 및 이메일 주소의 표기 위치와 형식 (기사 시작, 끝, 또는 미표기)
        -   숫자(예: 1000명 vs 1천 명), 날짜(예: 14일 vs 14日), 외래어 표기법의 일관성
        -   사진이나 그래픽에 대한 설명(캡션) 형식
    *   **문체적 특징:**
        -   인용문 처리 방식 (예: "라고 말했다" vs "라고 밝혔다" vs "전했다" 등의 선호 동사)
        -   문장의 평균 길이와 리듬 (단문 위주 vs 만연체 위주)
        -   특정 부사(예: '특히', '또한', '한편')나 접속사의 사용 빈도
        -   기관이나 인물의 직책을 표기하는 방식 (예: 'A 대표는' vs 'A 대표가')

**분석 대상:**
기사 텍스트 1: <[{title1}] {text1}>
기사 텍스트 2: <[{title2}] {text2}>
"""

TEST_PROMPT_V3 = """
## 기사 텍스트 1: {title1}

```txt
{text1}
```


## 기사 텍스트 2: {title2}

```txt 
{text2}
```
""".strip()

SYSTEM_INSTRUCTION = SYSTEM_INSTRUCTION_V2
PROMPT = TEST_PROMPT_V3

def create_pairs(df: pd.DataFrame) -> Tuple[list, list]:
    """
    주어진 DataFrame에서 같은 언론사와 다른 언론사끼리의 페어를 생성합니다.
    :param df:
        DataFrame, 'source', 'title', 'text' 컬럼을 포함해야 합니다.
    :return:
        Tuple, 같은 언론사끼리의 페어 리스트와 다른 언론사끼리의 페어 리스트를 포함합니다.
    """

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

    return same_pairs, diff_pairs


def validate_pairs(same_pairs: list, diff_pairs: list):
    """
    생성된 페어 리스트의 유효성을 검증합니다.
    :param same_pairs: 같은 언론사끼리의 페어 리스트
    :param diff_pairs: 다른 언론사끼리의 페어 리스트
    :return:
    """

    news_per_source = set()
    for pair in same_pairs + diff_pairs:
        source1 = pair[0]['source']
        source2 = pair[1]['source']
        news_per_source.add(source1)
        news_per_source.add(source2)

    confusion_matrix = {}
    for source1 in news_per_source:
        confusion_matrix[source1] = {}
        for source2 in news_per_source:
            confusion_matrix[source1][source2] = 0

    for pair in diff_pairs:
        source1 = pair[0]['source']
        source2 = pair[1]['source']
        confusion_matrix[source1][source2] += 1

    for pair in same_pairs:
        source1 = pair[0]['source']
        source2 = pair[1]['source']
        confusion_matrix[source1][source2] += 1

    print("\nConfusion Matrix for Different Source Pairs:")
    print(" " * 8 + " | " + "| ".join([name[:2] for name in sorted(news_per_source)]))
    print("-" * 60)
    for source1 in sorted(news_per_source):
        row = [source1[:4]]
        for source2 in sorted(news_per_source):
            row.append(str(confusion_matrix[source1][source2]).rjust(3))
        print(" | ".join(row))
    print("\nTotal pairs: ", len(same_pairs) + len(diff_pairs))


def create_jsonl(same_pairs: list, diff_pairs: list, save_path: Path):
    """
    주어진 같은 언론사와 다른 언론사 페어 리스트를 기반으로 JSONL 형식의 요청을 생성합니다.
    :param same_pairs:
    :param diff_pairs:
    :return:
    """
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
                    'temperature':0.1,
                    'max_tokens':1024
                }
            })
            custom_id_num += 1


    # jsonl 저장
    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)
    file_name = 'batch.jsonl'
    with open((save_path / file_name), 'w', encoding='utf-8') as f:
        for js in json_list:
            f.write(json.dumps(js, ensure_ascii=False)+'\n')

    print('\njsonl 파일 저장 완료.')


if __name__ == '__main__':
    # 인자 파싱
    parser = argparse.ArgumentParser(description='Create JSONL file for batch processing.')
    parser.add_argument('--csv-path', type=str, default="../dataset/preprocessed/filtered_news.csv", help='Path to the CSV file containing news data.')
    parser.add_argument('--save-path', type=str, default="../dataset/batch", help='Path to save the generated JSONL file.')
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    save_path = Path(args.save_path)

    df = pd.read_csv(csv_path, encoding='utf-8')

    # 페어 생성
    same_pairs, diff_pairs = create_pairs(df)

    # 페어 검증
    validate_pairs(same_pairs, diff_pairs)

    # JSONL 파일 생성
    create_jsonl(same_pairs, diff_pairs, save_path)