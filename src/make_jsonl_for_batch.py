import pandas as pd
from pathlib import Path
import json

# 불러오기
csv_path = Path('../dataset/preprocessed/filtered_news.csv')
df = pd.read_csv(csv_path, encoding='utf-8')

SYS