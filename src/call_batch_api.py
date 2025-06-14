from openai import OpenAI, APIError
from pathlib import Path
import pandas as pd
import json
import time

# API key 설정 필요
# export OPENAI_API_KEY=""
# export OPENAI_ORGANIZATION=""

# openai 객체 생성
try:
    client = OpenAI()
except APIError as e:
    print(f'OpenAI 클라이언트 초기화 실패: {e}')
    exit()

# 변수·상수
jsonl_path = Path('../dataset/preprocessed/batch.jsonl')
output_jsonl_path = Path('../dataset/preprocessed/batch_output.jsonl')
output_csv_path = Path('../dataset/preprocessed/batch_output.csv')
batch_id = ''
d = {
    'custom_id': [],  # ex: "same_source_pair_0000"
    'gold_label': [],  # ["same", "diff"]
    'pred_raw': [],  # ["True", "False"] (모델 출력1)
    'pred_label': [],  # ["same", "diff", ""] (pred_raw에서 이상한 거 출력 시 -> "")
    'is_success': [],  # [True, False]
    'analysis_text': [],  # text (모델 출력2)
    'is_error': []
}

def create_batch_job():
    # 파일 업로드
    try:
        batch_input_file = client.files.create(
            file=jsonl_path.open('rb'),
            purpose='batch'
        )
        print(f'--- 파일 업로드 완료 (file id: {batch_input_file.id}')
    except:
        print('파일 업로드 중 오류 발생')
        exit()

    # 배치 API 호출
    batch_job = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint='/v1/chat/completions',
        completion_window='24h'
    )
    print(f'--- 배치 업로드 완료 (batch id: {batch_job.id}')
    return batch_job.id

def monitor_batch_job(batch_id):
    # 15초마다 현황 확인
    while True:
        batch_job = client.batches.retrieve(batch_id)
        print('현재 상황:', batch_job.state)
        if batch_job.state == 'completed':
            print('-' * 50)
            print('배치 API 수행 완료')
            print('-' * 50)
            break
        elif batch_job.state in ['failed', 'cancelled']:
            print('-' * 50)
            print('배치 API 수행 중 오류 발생 (failed or cancelled)')
            print('-' * 50)
            exit()
        time.sleep(15)

    if batch_job.state == 'completed':
        output_file_id = batch_job.output_file_id
        error_file_id = batch_job.error_file_id
        print('output_file_id:', output_file_id)
        print('error_file_id:', error_file_id)

        if output_file_id:
            # 내용 추출
            output_file_content = client.files.content(output_file_id).read()
            output_file_content = output_file_content.decode('utf-8')

            # 파일로 저장 - jsonl
            with output_jsonl_path.open('w', encoding='utf-8') as f:
                f.write(output_file_content)
            print('output file 저장 :', str(output_jsonl_path))

            # 파일로 저장 - csv
            output_json_list = output_file_content.strip().split('\n')  # 빈 줄 제거
            for idx, line in enumerate(output_json_list):
                line = json.loads(line)
                custom_id = line.get('custom_id')
                gold_label = custom_id.split('_')[0]  # 'same' or 'diff'
                response_body = line.get('response', {}).get('body', {})
                if response_body and 'choices' in response_body:
                    response_json = response_body.get('choices', [])[0].get('message', {}).get('content', '')
                    response_json = json.loads(response_json)
                    analysis = response_json.get('분석', '')
                    answer = response_json.get('답변', '')

                    # 딕셔너리에 저장 (DataFrame 용)
                    d['custom_id'].append(custom_id)
                    d['gold_label'].append(gold_label)
                    d['pred_raw'].append(answer)
                    d['analysis_text'].append(analysis)

                    if answer == 'True':
                        d['pred_label'].append('same')
                    elif answer == 'False':
                        d['pred_label'].append('diff')
                    else:
                        d['pred_label'].append('')

                    if d['gold_label'][-1] == d['pred_label'][-1]:
                        d['is_success'].append(True)
                    else:
                        d['is_success'].append(False)

                    d['is_error'].append(False)

                    # 미리보기 출력 (앞 10개만)
                    if idx < 10:
                        print('-' * 50)
                        print(f'[PREVIEW-{idx}/10]')
                        print('custom_id:', custom_id)
                        print('답변:', answer)
                        print('분석:', analysis)

                else:
                    print('오류 발생 (response에 body 또는 choices 누락) | custom_id:', custom_id)
                    print('-' * 50)
                    print(json.dumps(line))
                    print('-' * 50)

                    # 딕셔너리에 저장 (DataFrame 용)
                    d['custom_id'].append(custom_id)
                    d['gold_label'].append(gold_label)
                    d['pred_raw'].append('')
                    d['analysis_text'].append('')
                    d['pred_label'].append('')
                    d['is_success'].append(False)
                    d['is_error'].append(True)

            df = pd.DataFrame(d)
            df.to_csv(output_csv_path, encoding='utf-8-sig', index=False)
            print('-' * 50 + '\n')

        if error_file_id:
            print('-' * 50)
            print('ERROR FILE:\n')
            error_file_content = client.files.content(error_file_id).read()
            print(error_file_content.decode('utf-8'))

def main():
    batch_id = ''

    # 모드 선택
    option = ''
    while option not in ['0','1']:
        option = input(
            "모드 선택 (번호만 입력)\n"+
            "0. BATCH API 호출 (최초 호출 시)\n"+
            "1. BATCH API 현황 확인\n: "
        )
        print('\n---\n')

    # BATCH API 호출
    if option == '0':
        batch_id = create_batch_job()
        monitor_batch_job(batch_id)

    # BATCH 실시간 현황 체크
    elif option == '1':
        if not batch_id:
            batch_id = input('batch id를 입력해주세요.: ')
        monitor_batch_job(batch_id)

if __name__ == '__main__':
    main()
