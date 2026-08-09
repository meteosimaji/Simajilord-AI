# Meteo 実行検証記録

更新: 2026-08-09

この文書は実行した事実だけを記録します。コード経路の受入、外部教師との接続、学習更新、
大会級棋力は別の主張です。特記しない限り外部binary、評価関数、生成棋譜、checkpointは
ローカルの一時領域またはユーザーのDownloadsにあり、リポジトリへ同梱していません。

## MLX end-to-end

次を実行しました。

```bash
uv run --project shogi_ai simajilord-shogi verify \
  <LOCAL_RUN_DIRECTORY> \
  --workers 4 --profile competition_v1
```

20 block x 256 channel ResNet、24,384,760 parametersをMLX 0.32.0/Metalで動かし、次を確認しました。

- 異なる4個の一手詰めSFENをbatched neural MCTSで同時探索
- 4局とも1手で詰みまで完遂、`max_plies`到達0局
- 全棋譜の合法手replay成功
- 同じ4局面を各200 simulationsで深再解析
- MLX automatic differentiationで8 step更新
- 固定probe loss `8.3195877075 -> 6.6896896362`、比率`0.804089`
- global gradient clip閾値1.0、観測したclip前最大norm `389.8749`
- 学習中MLX peak memory `607,717,368 bytes`
- checkpointをstep 8で保存・厳密再読込
- 再読込後の初期局面で合法手30手だけを出力
- 再読込モデルでも一手詰めを完遂

これは「終局対局を生成し、その局面を学習してlossを下げ、保存モデルを再び対局へ使える」という
基盤ゴールの証拠です。4局面8 stepの重みが強いという意味ではありません。

## 自己改善1世代

`smoke` profileと決定論的な一手詰め局面を使い、次を実際に連続実行しました。

1. batched actor 2局を詰みまで完遂
2. 同じ2局面をactorより深い探索で再解析
3. 深教師2 sampleで1 step学習
4. candidate checkpointとSHA-256を世代manifestへ保存
5. candidate/championを色替わりで2局
6. 勝点率0.5のためcandidateを棄却し、championを維持

棄却まで含めて自己改善ループです。弱くなったcandidateを自動昇格させないことも受入条件です。

## 本物の無料教師との終局対局

すべて初期局面、色替わり2局、Meteoは未学習`smoke` checkpoint・32 simulations/手、
教師は1 thread・原則10,000 nodes/手、定跡なしです。いずれも上限到達0、全着手を合法に再生し、
詰みまで終局しました。

| 教師 | Meteo結果 | 終局手数 | 用途 |
| --- | ---: | --- | --- |
| NAGISA V3.1 | 0勝2敗 | 28、21 | 強いNNUE評価・探索教師 |
| AobaNNUE v1.1 | 0勝2敗 | 26、21 | 無料NNUE異種教師 |
| 技巧2 v2.0.2 | 0勝2敗 | 28、33 | 戦術・10戦型・人間的評価の多様性教師 |
| 水匠5 | 0勝2敗 | 30、35 | 固定された再現可能baseline |

2026-08-09に現コードで技巧2との2局を再実行した結果も0勝2敗、未完0でした。この再実行は
Meteo 36手合計1,152 nodes/1.309秒/約880 NPS、技巧37手合計449,127 reported nodes/
0.0567秒/約791万reported NPSを保存しました。外部エンジンの`nodes`とMCTS simulationは同一の
仕事量定義ではないため、NPSの大小だけで棋力を比較しません。

現時点の実測は4教師すべてに0勝2敗です。「NAGISA、水匠、氷彗、Floodgate参加者へ勝てる」とは
まだ主張できません。接続・終局・記録経路が実エンジンでも動く証拠です。

## MultiPV蒸留と深度差curriculum

`reanalyse-usi`で、台帳がoutput蒸留可とする教師だけを起動しました。MultiPV=4、原則
10,000 nodes/局面で次を実動確認しています。

| 教師 | 蒸留局面 | 最善手逆転 | 備考 |
| --- | ---: | ---: | --- |
| NAGISA V3.1 | 2 | 0 | 一手詰めprobe |
| AobaNNUE v1.1 | 2 | 0 | 一手詰めprobe |
| 水匠5 | 2 | 0 | 一手詰めprobe |
| 技巧2（初回） | 24 | 6 | 戦術局面、lower-bound regret 1 |
| 技巧2（telemetry再実行） | 28 | 6 | 全局面でactor/teacher node比を保存 |

telemetry再実行の技巧28局面は教師nodeを合計347,458、reported NPSを73.5万〜2,052.1万で保存し、
全28局面で探索比を計算できました。7局面は教師がactorの64倍超、最大比は約899.59でした。

この技巧28局面へNAGISA/AobaNNUE/水匠の6 anchor局面を加えた34 sampleを8 step学習しました。

- 固定probe loss `8.7963972092 -> 8.3014965057`
- probe loss比 `0.943738`
- configured teacher policy mix `0.75`
- 深度差適用後のeffective mix `0.200045〜0.75`、平均`0.669317`
- teacher value mixも同じcurriculum scaleで縮小
- global gradient clip閾値1.0、clip前最大norm `29.3014`
- 合法手だけへのlabel smoothing `0.01`
- MLX peak memory `41,111,084 bytes`（`smoke` model）

これにより「教師が約900倍深い局面を、いきなり100%置換せず段階導入し、固定probe lossを
悪化させず更新できた」ことを実データで確認しました。大規模長期学習が崩壊しない保証ではないため、
世代ごとに同じ固定probe、held-out arena、過去championを継続監視します。

## engine・評価関数の固定hash

| 教師 | archive SHA-256 | engine SHA-256 | eval/model SHA-256 |
| --- | --- | --- | --- |
| NAGISA V3.1 | `d8912d47bc1f4466ff96daeba8d00e3f92ad7efa9c09b243eecad7e352da1041` | `f3fab18477e1be7069719f59bcb4126596fcb6eee606d74502feac313cd29799` | `e6b0b6ac99e95922ceba11633cc8e329968b152d7a79910156405f8a7ea9cdb9` |
| AobaNNUE v1.1 | `83d4798d7461e29e95ba1700f4f5bf10e670b360ab1004102e21b913c22ed979` | `9d1609ea75bd76fd90d0391048fd991db3dffb50000aa9d48b1ab38c73561b21` | `f8ee839ae8c08537036f23345dd5ed0416958b22425476fc60177942903219b5` |
| 技巧2 v2.0.2 | `3bbb667bfd77e0236b052593b5c61b7fe1827ed72b5b5f437760c60974852757` | `1170efc8fefe5754533fe8c5a4fd569823636c5cd4d9a3b7f4fbdfc3ee6f9916` | GPL配布の複数parameter file。repoへ未同梱 |
| 水匠5 | `6734e3a3d28e67b9206c3442f6d10f16148138327dff811cadedfcf581f79809` | `f4bfaee3f411e9688ebf55a585593463311a18bb2ea01b02eeb5fb0babcacb13` | `768068f0d534a0603a5d38bcd143de6bbca820d5f1c95a14d40863e5b7892d76` |

実行・出力蒸留と、重みのコピー・変換・再配布は別判定です。詳細は
[`MODEL_RIGHTS.md`](MODEL_RIGHTS.md)を参照してください。

## NPSと統合メモリ

64 GiB Apple Silicon、ランダム重み、初期局面で`bench-nps`を実行しました。

| profile | 並列root | node/root | 合計NPS | MLX peak | 1億node/root見積り |
| --- | ---: | ---: | ---: | ---: | ---: |
| ResNet 20x256 | 1 | 100 | 104.58 | 102,631,204 B | 265.60時間 |
| ResNet 20x256 | 32 | 100 | 780.88 | 215,785,632 B | 32 root完了まで1,138.32時間 |
| hybrid 40x512 | 1 | 30 | 38.07 | 754,925,376 B | 729.56時間 |

短いwarmupを含むmicrobenchmarkなので長期平均ではありません。現在のメモリsnapshotでは、
総量68,719,476,736 bytes、予約約10.31 GB、MLX上限約13.3 GB、tree予算約8.4 GB、推定tree上限
約205万nodeでした。`max_tree_nodes=1024`、1,100 simulationsの試験では、探索を停止せず
subtree recycleが1回以上発生しました。

この安全弁はOOM回避です。tree recycle後もroot visit/value統計は残りますが、深いsubtreeは
失われます。したがって「1億nodeを数えられる」と「1億nodeの深い木を有効保持できる」は
同じ検証結果として扱いません。

## Floodgate

[Floodgate](https://wdoor.c.u-tokyo.ac.jp/shogi/)へは接続していません。現在の0勝2敗x4という
実測から、参加可能な棋力とは判定していません。参加前の時間制御、CSA、耐久、棋力、公開実績の
条件は[`FLOODGATE.md`](FLOODGATE.md)へ固定しています。
