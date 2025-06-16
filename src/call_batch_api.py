from openai import OpenAI, APIError
from pathlib import Path
import pandas as pd
import json
import time
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef

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

def input_batch_id(client, limit=10):
    # batch list 불러오기
    batch_list = list(client.batches.list(limit=limit))

    if not batch_list:
        print('-' * 30)
        print('BATCH API 호출 기록이 없습니다. 새로운 BATCH API를 호출해주세요.')
        print('-' * 30)
        batch_id = input('batch id를 입력해주세요.: ').strip()
    else:
        print('-'*30, '(1번이 최신)')
        for idx, batch in enumerate(batch_list, start=1):
            try:
                request_counts = batch.request_counts.total
            except:
                request_counts = '?'
            print(f'{idx}. {batch.id} ({request_counts} requests)')
        print('-' * 30)
        batch_id_input = input('batch id 또는 번호를 입력해주세요.: ').strip()
        if batch_id_input in list(map(str,range(1,len(batch_list)+1))):
            batch_idx = int(batch_id_input)-1
            batch_id = batch_list[batch_idx].id
        else:
            batch_id = batch_id_input

    return batch_id

def show_statistics(df, labels=['same','diff']):
    print(f'*에러가 아닌 케이스만 통계로 수집 (error case: {len(df[df['is_error']])})\n')
    df = df[~df['is_error']]

    print('[Confusion Matrix]')
    print('labels:', labels)
    cm = confusion_matrix(df['gold_label'], df['pred_label'], labels=labels)
    cm_df = pd.DataFrame(cm)
    cm_df.index = list(map(lambda x: 'true_'+x, labels))
    cm_df.columns = list(map(lambda x: 'pred_'+x, labels))
    cm_df['| sum'] = cm_df.sum(axis=1)
    cm_df.loc['-- sum --'] = cm_df.sum(axis=0)
    print(cm_df)
    print('-'*50)
    print('[Classification Report]')
    print(classification_report(df['gold_label'], df['pred_label'], labels=labels))
    print('-'*50)
    print('[MCC]')
    print(':', matthews_corrcoef(df['gold_label'], df['pred_label']))

def create_batch_job(jsonl_path, sample_num=0):
    # 샘플 처리
    if sample_num > 0:
        with jsonl_path.open('r', encoding='utf-8') as f:
            content = f.read().strip().split('\n')
        head = content[:int(sample_num/2)] # same
        tail = content[-int((sample_num+1)/2):] # diff

        sample_jsonl_path = jsonl_path.parent / 'sample_batch.jsonl'
        with sample_jsonl_path.open('w', encoding='utf-8') as f:
            for line in head+tail:
                f.write(line+'\n')
        jsonl_path = sample_jsonl_path

    # 파일 업로드
    try:
        batch_input_file = client.files.create(
            file=jsonl_path.open('rb'),
            purpose='batch'
        )
        print(f'--- 파일 업로드 완료 (file id: {batch_input_file.id})')
    except:
        print('파일 업로드 중 오류 발생')
        exit()

    # 배치 API 호출
    batch_job = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint='/v1/chat/completions',
        completion_window='24h'
    )
    print(f'--- 배치 업로드 완료 (batch id: {batch_job.id})')
    return batch_job.id

def monitor_batch_job(batch_id):
    # 15초마다 현황 확인
    while True:
        try: batch_job = client.batches.retrieve(batch_id)
        except:
            print('잘못된 batch id입니다.')
            return
        print('현재 상황:', batch_job.status)
        if batch_job.status == 'completed':
            print('-' * 50)
            print('배치 API 수행 완료')
            print('-' * 50)
            break
        elif batch_job.status in ['failed', 'cancelled']:
            print('-' * 50)
            print('배치 API 수행 중 오류 발생 (failed or cancelled)')
            print('-' * 50)
            exit()
        time.sleep(15)

    if batch_job.status == 'completed':
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
                for line in output_file_content.strip().split('\n'):
                    f.write(json.dumps(json.loads(line), ensure_ascii=False)+'\n')
            print('output file 저장 :', str(output_jsonl_path))

            # 파일로 저장 - csv
            output_json_list = output_file_content.strip().split('\n')  # 빈 줄 제거
            for idx, line in enumerate(output_json_list):
                is_error = False
                line = json.loads(line)
                custom_id = line.get('custom_id')
                gold_label = custom_id.split('_')[0]  # 'same' or 'diff'
                response_body = line.get('response', {}).get('body', {})
                if response_body and 'choices' in response_body:
                    response_json_str = response_body.get('choices', [])[0].get('message', {}).get('content', '')
                    try:
                        response_json = json.loads(response_json_str)
                    except json.decoder.JSONDecodeError as e:
                        print(f'모델이 답변한 JSON 파싱 중 오류 발생: {e}')
                        print('v'*30)
                        print(response_body.get('choices', [])[0].get('message', {}).get('content', ''))
                        print('^'*30)
                        response_json = {}
                        is_error = True
                    analysis = response_json.get('분석', '')
                    answer = response_json.get('답변', '')

                    # 딕셔너리에 저장 (DataFrame 용)
                    d['custom_id'].append(custom_id)
                    d['gold_label'].append(gold_label)
                    d['pred_raw'].append(answer)
                    d['analysis_text'].append(analysis)

                    if answer == True:
                        d['pred_label'].append('same')
                    elif answer == False:
                        d['pred_label'].append('diff')
                    else:
                        d['pred_label'].append('')
                        is_error = True

                    if d['gold_label'][-1] == d['pred_label'][-1]:
                        d['is_success'].append(True)
                    else:
                        d['is_success'].append(False)

                    d['is_error'].append(is_error)

                    # 미리보기 출력 (앞 10개만)
                    if idx < 10:
                        print('-' * 50)
                        print(f'[PREVIEW-{idx}]')
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
            show_statistics(df, ['same','diff'])

        if error_file_id:
            print('-' * 50)
            print('ERROR FILE:\n')
            error_file_content = client.files.content(error_file_id).read()
            print(error_file_content.decode('utf-8'))

def main():
    batch_id = ''

    # 모드 선택
    option = ''
    while option not in ['0','1','2']:
        option = input(
            "모드 선택 (번호만 입력)\n"+
            "0. BATCH API 호출 (전체)\n"+
            "1. BATCH API 호출 테스트 (샘플 n개)\n" +
            "2. BATCH API 현황 확인\n: "
        )
        print('\n---\n')

    # BATCH API 호출
    if option == '0':
        batch_id = create_batch_job(jsonl_path)
        monitor_batch_job(batch_id)

    elif option == '1':
        try:
            sample_num = int(input('테스트할 샘플 개수 입력 (기본값 10): '))
        except:
            sample_num = 10
        batch_id = create_batch_job(jsonl_path, sample_num=sample_num)
        monitor_batch_job(batch_id)

    # BATCH 실시간 현황 체크
    elif option == '2':
        if not batch_id:
            batch_id = input_batch_id(client)
            if not batch_id: return
        monitor_batch_job(batch_id)

if __name__ == '__main__':
    main()
