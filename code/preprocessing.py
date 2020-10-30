import MeCab
import mojimoji
import re
import pandas as pd
import numpy as np

tagger = MeCab.Tagger("-Owakati -d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd")
def make_wakati(sentence):
    # MeCabで分かち書き
    sentence = sentence.lower()
    sentence = tagger.parse(sentence)
    sentence = mojimoji.zen_to_han(sentence)
    # 記号もろもろ除去
    sentence = re.sub(r'[\．_－―─！＠＃＄％＾＆\-‐|\\＊\“（）＿■×+α※÷⇒—♬◉ᴗ͈ˬ●▲★☆⭐️⭕⚡⚠①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮♡⭐︎〇◎◆◼♦▼◇△□(：〜～＋=)／*&^%$#@!~`)♪ᴖ◡ᴖｰ{}［］↑↓←→➡⇩™･⊡…\[\]\"\'\”\’:;<>?＜＞〔〕\r＼−〈〉？、､。｡・,\./『』【】｢｣「」→←○《》≪≫\n\u3000⭕]+', "", sentence)
    # 絵文字除去
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           "]+", flags=re.UNICODE)
    sentence = emoji_pattern.sub(r'', sentence)
    return sentence

def title_torkenize(sentence):
    sentence = mojimoji.zen_to_han(sentence)
    sentence = re.sub("[\．_－―─！＠＃＄％＾＆\-‐|\\＊\“（）＿■×+α※÷⇒♬◉ᴗ͈ˬ—●▲★☆⭐️⭕⚡⚠①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮♡⭐︎〇◎◆♦▼◼◇△□(：〜～＋=)／*&^%$#@!~`)♪ᴖ◡ᴖｰ{}［］↑↓←→➡⇩™･⊡…\[\]\"\'\”\’:;<>?＜＞〔〕\r＼−〈〉？、､。｡・,\./『』【】｢｣「」→←○《》≪≫\n\u3000]", " ", sentence)
    sentence = re.sub("[あ-ん]", " ", sentence)
    sentence = re.sub("( |　)+", " ", sentence)
    sentence = sentence.lower()
    #〇〇様専用を除く
    sentence = re.sub("[^ ]*専用", "", sentence)
    sentence = re.sub("[^ ]*様", "", sentence)
    #1文字のアルファベットを除く
    sentence = re.sub(" [a-z]{1}[^(a-z)]", " ", sentence)
    # 絵文字除去
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           "]+", flags=re.UNICODE)
    sentence = emoji_pattern.sub(r'', sentence)
    sentence = sentence.strip()

    return sentence

def preprocess(df):
    df["price"] = df["price"].str.replace(r"\D", "").astype(np.float)
    
    #列ズレを修正
    pattern = re.compile(r"^(?!.*(傷や汚れあり|全体的に状態が悪い|やや傷や汚れあり|未使用に近い|目立った傷や汚れなし|新品、未使用)).+$")
    invalid = df["status"].str.match(pattern)

    df.loc[invalid, "description"] = df.loc[invalid, "description"] + "\n" + df.loc[invalid, "status"]
    df.loc[invalid, "status"]      = df.loc[invalid, "shipping"]
    df.loc[invalid, "shipping"]    = df.loc[invalid, "method"]
    df.loc[invalid, "method"]      = df.loc[invalid, "region"]
    df.loc[invalid, "period"]      = "未定"
    
    df["title"] = df["title"] + " " + df["sub_category_1"] + " " + df["sub_category_2"] + " " + df["brand"]

    df = df.drop(columns=["sub_category_1", "sub_category_2", "brand"])
    
    status_dict = {'新品、未使用': "best",
                   '未使用に近い': "Very Good",
                   '目立った傷や汚れなし': "good",
                   '傷や汚れあり': "Poor",
                   'やや傷や汚れあり': "very poor",
                   '全体的に状態が悪い': "worst"
                  }
    
    #配送負担をラベルエンコーディング
    shipping_dict = {'送料込み(出品者負担)': 0, '着払い(購入者負担)': 1}

    df["status"] = df["status"].map(status_dict)
    df["shipping"] = df["shipping"].map(shipping_dict)
    
    #トークナイズ
    df["title"] = df["title"].apply(title_torkenize)
    df["description"] = df["description"].apply(make_wakati)
    
    #不要列削除
    return_columns = ['title', 'price', 'status', 'shipping', 'description']
    return df[return_columns]