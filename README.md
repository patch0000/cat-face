# cat-face
本のコードサンプルに基づいた猫顔認識のpython実装

技術評論社 データサイエンティスト要請読本 ～機械学習入門編～
特集４の３章　シンプルな画像認識を実装サンプルコードを
WindowsのAnaonda上Python3.6で動かしたものです。

・掲載コードは断片的なものに留まっている  
・Python2.X系の書式になっている  
・おそらくUnix環境前提  
のため、適宜変更及び試行錯誤。

##2018年1月21日
現時点では下記の問題が未解決 
（１）face_detect.pyが猫顔を正しく切り出せない時がある

###１）LBPで各画像から特徴量を抜き出す
get_feature.py

###２）SVMで学習
特徴量とそれが猫か否かのラベルを使いscikit-learnのLinerSVMで学習する
svm.py

###３）検証
ダミーデータを使って精度を検証する
Accuracy.py  

###４）間違った画像の確定
image_check.py

###５）猫顔部分の検出
face_detect.py

###６）参考ページ
技術評論社様のサポートページは下記です。
http://gihyo.jp/book/2015/978-4-7741-7631-4/support

著者様のgithubはおそらく下記です。
https://github.com/t-abe/cat-face-detection

