import pandas as pd
from pathlib import Path
import argparse
import json


def parse_news(dataset_path: Path) -> pd.DataFrame:
    """
    Parse the news dataset from JSON files into a CSV file.
    :param dataset_path:
    :return:
    """

    dic = {
        'source':[],
        'title':[],
        'text':[]
    }

    except_count = 0

    # 데이터 추출
    for json_path in dataset_path.iterdir():
        if json_path.suffix != '.json':
            continue
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

    df = pd.DataFrame(dic)
    return df


def preprocess_news(df: pd.DataFrame, min_length: int = 501, max_length: int = 1000) -> pd.DataFrame:
    """
    뉴스 기사 데이터를 전처리하여 길이에 따라 필터링하고, 상위 10개 언론사의 기사만 추출하여 저장합니다.
    :param df: 입력 데이터프레임 (columns: ['id', 'title', 'text', 'source'])
    :param min_length: 최소 기사 길이 (기본값: 501)
    :param max_length: 최대 기사 길이 (기본값: 1000)
    :return: 필터링된 뉴스 기사 데이터프레임
    """

    # 길이대로 자르기 (500자 초과 1000자 이하)
    df['length'] = df['text'].apply(len)
    df_filtered = df[(min_length <= df['length']) & (df['length'] <= max_length)]

    # 기사 수 상위 10개의 언론사만 추출
    top10 = df_filtered.groupby('source').count().sort_values(by='text', ascending=False).iloc[:10]
    print('상위 10개 언론사')
    for _, (i,c) in enumerate(zip(top10.index,top10['title']), start=1):
        print(f'{_}. {i} ({c}개)')
    print('-'*50)

    df_top10_filtered = df_filtered[df_filtered['source'].isin(top10.index)]
    df_top10_filtered = df_top10_filtered.reset_index(drop=True) # 인덱스 0부터 재지정

    return df_top10_filtered


def randomize_and_sample_news(df: pd.DataFrame, sample_size: int = 100, seed: int = 42) -> pd.DataFrame:
    """
    뉴스 기사 데이터를 무작위로 섞고, 지정된 크기만큼 샘플링합니다.
    :param df: 입력 데이터프레임
    :param sample_size: 신문사별로 샘플링할 기사 수 (기본값: 100)
    :param seed: 무작위 시드 값 (기본값: 42)
    :return: 무작위로 섞인 샘플링된 데이터프레임
    """

    concat_list = []
    for source, member_df in df.groupby('source'):
        sampled_member = member_df.sample(n=sample_size, random_state=seed)
        concat_list.append(sampled_member)

    df_sampled = pd.concat(concat_list, axis=0).reset_index(drop=True)

    # 통계
    print(f'총 {len(df_sampled)}개의 기사가 샘플링되었습니다.')
    print('샘플링된 데이터프레임의 상위 5개 행:')
    print(df_sampled.head())

    return df_sampled


if __name__ == '__main__':
    # 인자 파싱
    parser = argparse.ArgumentParser(description='Parse news dataset from JSON files to CSV and filter.')
    parser.add_argument('--min-length', type=int, default=501, help='최소 기사 길이')
    parser.add_argument('--max-length', type=int, default=1000, help='최대 기사 길이')
    parser.add_argument('--dataset-path', type=str, default="../dataset", help='Path to the dataset directory containing JSON files.')

    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"The specified dataset path does not exist: {dataset_path}")

    df = parse_news(dataset_path)

    print(f"Total records parsed: {len(df)}")

    output_dir = dataset_path / 'preprocessed'
    output_dir.mkdir(parents=True, exist_ok=True)

    parsed_file = output_dir / 'parsed_news.csv'
    df.to_csv(parsed_file, encoding='utf-8-sig', index_label='id')

    print(f"Parsed raw dataset saved to: {parsed_file}")

    sampled_file = output_dir / 'filtered_news.csv'

    filtered_df = preprocess_news(df, args.min_length, args.max_length)
    sampled_df = randomize_and_sample_news(filtered_df, sample_size=100, seed=42)

    sampled_df.to_csv(sampled_file, encoding='utf-8-sig', index_label='id')
    print(f'필터링된 뉴스 기사가 {sampled_file}에 저장되었습니다.')