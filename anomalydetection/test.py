import requests
import pandas as pd
import json

data = requests.get('https://herokunewsproject.herokuapp.com/api/posts/')
data = data.json()
for i in data:
    i.pop('comments')
with pd.option_context('display.max_rows', None, 'display.max_columns', None, "display.expand_frame_repr", False):  # more options can be specified also
    print(pd.DataFrame.from_dict(data))
