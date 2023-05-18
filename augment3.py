import openai
import re
import pandas as pd
import csv
import pandas as pd
from tqdm import tqdm
import os
import numpy as np

## api key
OPENAI_API_KEY = "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"
openai.api_key = OPENAI_API_KEY

## api call
model = "gpt-3.5-turbo"

folder = os.listdir('/home/jihyeon/toxicity_movie/toxicity_movie_sub_after/18세관람가')
# os.rmdir('/home/jihyeon/toxicity_movie/toxicity_movie_sub/12관람가/.DS_Store')
# 확장자가csv인 파일만 다운받는편이 error 덜생성
folder = [file for file in folder if file.endswith(".csv")]  # 확장자가 csv인 것만 추출
print(folder)

path = ['전체관람가', '12세관람가', '15세관람가', '18세관람가']

for files in folder:  # files = 영화 제목
    print(files)
    # print(files)
    # df=pd.read_csv("/home/jihyeon/toxicity_movie/toxicity_movie_sub/전체관람가/" + files, index_col=0, error_bad_lines=False)
    data = pd.read_csv('/home/jihyeon/toxicity_movie/toxicity_movie_sub_after/18세관람가/' + files, sep=',')

    # 결측치 제거
    data['Subtitle'].replace('', np.nan, inplace=True)
    data.dropna(subset=['Subtitle'], inplace=True)

    ## csv reader
    texts = []
    labels = []

    for text in zip(data['Subtitle']):
        texts.append(text)

    ini = 0

    while True:
        iidx = []
        results = []

        for idx, _ in enumerate(zip(texts[ini:]), start=ini):

            text_3 = texts[idx*3]+texts[idx*3+1]+texts[idx*3+2]
            #text_5 = texts[idx*5]+texts[idx*5+1]+texts[idx*5+2]+texts[idx*5+3]+texts[idx*5+4]
            #text_7 = texts[idx*7]+texts[idx*7+1]+texts[idx*7+2]+texts[idx*7+3]+texts[idx*7+4]+texts[idx*7+5]+texts[idx*7+6]
            # 1 query = '법률 사례:' + text[idx+1]+'\n'+"사실 관계: "+'\n'+key_facts[idx+1]+'\n'+ '법률 사례' + text[idx]+'\n'+"사실 관계: "
            # 2 query = '[사례]\n' + text[idx] + '\n[주요 사실 관계]\n' + key_facts[idx] + '\n[질문]\n' + question[idx] + '\n이에 대한 답변을 법률적 쟁점을 들어 항목별로 간단히 작성하시오'
            # 3 query = '[사례]\n' + text[idx] + '\n[질문]\n' + question[idx] + '\n[주요 사실 관계]\n' + key_facts[idx] + '\n[질문]을 해결하기 위한 법률적 쟁점을 작성하고, 이에 대하여 서술하시오.'
            # query = '[텍스트]\n' + texts[idx] + ' (텍스트를 읽고 유해성이 있으면 1, 없으면 0으로 답변하세요.)'
            # query = '[텍스트]\n' + "".join(texts[idx]) + ' (Answer with 1 if there is toxicity, and 0 if there is not toxicity. Respond only with numbers. no explanation)'
            query = '[텍스트]\n' + "".join(text_3) + ' (Answer with 1 if there is strong toxicity example "씨발", and 0 if there is not toxicity. Respond only with numbers. no explanation)'


            messages = [
                {"role": "system",
                 "content": "you are a professional data monitoring agent. Only answer harmness with either 0 or 1"},
                {"role": "user", "content": query}
            ]

            try:
                response = openai.ChatCompletion.create(model=model, messages=messages)
                answer = response['choices'][0]['message']['content']
                print("index: ", idx)
                # print(query)
                print("prediction: ", answer)
                print()
                iidx.append(idx)
                results.append({"id": idx, "prompt": query, "prediction": answer})

            except:
                print(f"{idx}_error")
                # iidx.append(idx)
                # results.append({"id": idx, "prompt": query, "prediction": "error"})
                 #  break

            if idx >= len(texts)/3-6  :
              break
                
            #df = pd.DataFrame(results)
            #df.to_csv('/home/jihyeon/toxicity_movie/chat_gpt_api/toxic_3_width/전체관람가/' + files +  '.tsv' , sep='\t', index=False)

        ini = ini + len(iidx)
        if ini >= len(text):
        #if ini >= 10:
            break

    df = pd.DataFrame(results)
    df.to_csv('/home/jihyeon/toxicity_movie/chat_gpt_api/toxic_3_width/18세관람가/' + files + '.tsv', sep='\t', index=False)
