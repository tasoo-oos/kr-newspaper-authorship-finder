import pandas as pd
import json
from pathlib import Path
import argparse


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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse news dataset from JSON files to CSV.')
    parser.add_argument('--dataset-path', type=str, default="../dataset", help='Path to the dataset directory containing JSON files.')

    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"The specified dataset path does not exist: {dataset_path}")

    df = parse_news(dataset_path)

    print(f"Total records parsed: {len(df)}")

    output_dir = dataset_path / 'preprocessed'
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / 'parsed_news.csv'
    df.to_csv(output_path, encoding='utf-8-sig', index_label='id')

    print(f"Parsed dataset saved to: {output_path}")