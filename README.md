# voice_conversion　　
## 使い方
1. data/male/wavというディレクトリに男声のデータを、data/female/wavというディレクトリにそれと対になる女声のデータを配置する。
2. 01feature_analysis.pyを実行する
3. 02timewarping.pyを実行する
4. 03makelist.shを実行する
5. 04train.pyを実行する
6. 05convert.pyを実行する  

* dataというフォルダには音声が5サンプル分あります。sourceをtargetに変換しようとして出力されたのがresultです。数字が同じものは同じセリフのものです。
