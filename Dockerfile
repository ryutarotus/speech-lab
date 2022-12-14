# python:3.8の公式 image をベースの image として設定
FROM python:3.8

# 作業ディレクトリの作成
RUN mkdir /code

# 作業ディレクトリの設定
WORKDIR /code

# カレントディレクトリにある資産をコンテナ上の指定のディレクトリにコピーする
ADD . /code

RUN apt-get update && \
    apt-get install -y build-essential cmake clang libssl-dev vim && \
    apt-get install -y --no-install-recommends build-essential gcc libsndfile1 && \
    apt-get -y install ffmpeg

# pipでrequirements.txtに指定されているパッケージを追加する
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . /code/
