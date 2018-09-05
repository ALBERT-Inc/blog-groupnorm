# Group Normalization プログラム

[ニューラルネットの新しい正規化手法 Group Normalization の高速な実装と学習実験 | ALBERT Official Blog](https://blog.albert2005.co.jp/2018/09/05/group_normalization) で用いたプログラムです。


## ライセンス

MIT ライセンスです。

ただし [xoshiro128+ のリファレンス実装](http://xoshiro.di.unimi.it/xoshiro128plus.c)を移植した
`xoshiro.py` のみ、元の実装と同じくパブリックドメインとします。

## 動作環境

* Python 3.6.6
* CUDA 9.2
* cuDNN 7.1
* Linux x64

で確認済みです。

おそらく

* Python 3.4 以降
* CUDA 9.0 以降
* cuDNN 5.0 以降

くらいであれば動作します。


## 依存ライブラリのインストール

まず virtualenv や miniconda でサンドボックス環境を作ることを勧めます。
以下のコマンドで依存ライブラリをインストール可能です。

    CC="gcc -O2 -march=native" pip install -r requirments.txt

`CC` 環境変数は筆者の環境で Pillow-SIMD を動かすために必要でした。


## 実行方法

最初に

    ./get_dataset.py


を実行してデータセットをダウンロードする必要があります。

学習は

    ./train.py --normalization=(bn|gnchainer|gnalb1|gnalb2)

で行います。`result` ディレクトリに json 形式のログファイル `log` および
npz 形式の重みファイル `model.npz` が出力されます。

`-o` オプションによって出力ディレクトリの変更ができます。GPU の指定は
`-g` オプションで行います。

学習に関するパラメータの変更を行う際は `train.py` を編集してください。


## results ディレクトリ

`results` ディレクトリ内には手元で学習を行って出力されたログファイル群および
ブログに載せる図表を生成するスクリプト `plot.py` があります。`plot.py`
の実行には matplotlib 2.2 以降が必要です。これを実行するとブログに載せたのと
同じ図表が生成されます。

`plot.py` では筆者が LinoType より購入した 'Avenir Next LT Pro' フォントを
指定しているので、多くの環境ではフォントが変わってしまいます。未確認ですが、
Mac ユーザーはフォント指定を 'Avenir Next' に変更することで OS にバンドル
されているフォントを利用できると思われます。
