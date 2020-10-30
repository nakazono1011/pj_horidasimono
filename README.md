# Horidasu

個人プロジェクト「フリマ掘り出し物発掘AI」のレポジトリです。

# DEMO

![スクリーンショット 2020-10-30 165600](https://user-images.githubusercontent.com/66734196/97674525-05a33080-1ad1-11eb-8710-92b68c1e32c6.png)

# Features

機械学習を用いてフリマ出品中の商品の中からお得な掘り出し物をレコメンド・査定を行います。
出品中の商品の本来価値を査定し、お得な商品を表示するWebアプリです。
多種多様な商品の相場をわざわざ調べる手間を省き、良質で低価格な商品を素早く見つけ出す支援を行います。

# Requirement
* mecab-python3==1.0.1
* mojimoji==0.0.11
* tensorflow==2.3.0
* Keras==2.4.3
* flake8==3.8.3
* SQLAlchemy==1.2.19

# Installation

1. Mecab辞書のインストール
```bash
#Mecabインストール
!sudo apt install mecab libmecab-dev mecab-ipadic-utf8 git make curl xz-utils file swig
#形態素解析器辞書としてmecab-ipadic-NEologd
!sudo apt install git make curl xz-utils file
%cd /tmp
!git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git
%cd mecab-ipadic-neologd
!./bin/install-mecab-ipadic-neologd -n
# シンボリックリンクによるエラー回避
!ln -s /etc/mecabrc /usr/local/etc/mecabrc
!pip install mojimoji
!pip install mecab-python3
```

2. モジュールインストール
```bash
pip install mecab-python3
pip install mojimoji
```

# Usage

app.py を実行することでサーバーが起動します

```bash
cd webapp
python app.py
```

# Note

* 現在はメルカリ、電化製品ジャンルのみが対象です

# Author

* 作成者: Keisuke Nakazono
* メールアドレス: k.nakazono1011@gmail.com

# License
ライセンスを明示する

"Horidasu" is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).
"Horidasu" is Confidential.
