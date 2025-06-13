import pandas as pd
from pathlib import Path
import argparse


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
    parser = argparse.ArgumentParser(description='뉴스 기사 전처리')
    parser.add_argument('--min-length', type=int, default=501, help='최소 기사 길이')
    parser.add_argument('--max-length', type=int, default=1000, help='최대 기사 길이')
    parser.add_argument('--input-path', type=Path, default='../dataset/preprocessed/parsed_news.csv', help='입력 파일 경로')
    args = parser.parse_args()

    input_path = Path(args.input_path)
    sampled_file = input_path.parent / 'filtered_news.csv'

    df = pd.read_csv(input_path, encoding='utf-8', index_col=0)

    filtered_df = preprocess_news(df, args.min_length, args.max_length)
    sampled_df = randomize_and_sample_news(filtered_df, sample_size=100, seed=42)

    sampled_df.to_csv(sampled_file, encoding='utf-8-sig', index_label='id')
    print(f'필터링된 뉴스 기사가 {sampled_file}에 저장되었습니다.')