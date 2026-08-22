# Meteo 教師系譜・公開局面再利用台帳

更新: 2026-08-13

この台帳は「公開ページに置かれている」「教師が利用したと書かれている」「Meteoが安全に学習へ
投入できる」を別々に判定します。機械可読な正本は
`simajilord-shogi teacher-lineage`です。モデル自体の実行・蒸留・再配布可否は
[`MODEL_RIGHTS.md`](MODEL_RIGHTS.md)を正本とし、本書は学習データの系譜と技術互換性を扱います。
関連配布者のHugging Face全26 repository、Aoba/GCT/Qhapaq/大会棋譜、ローカルdataを含む
全件表は[`DATASET_AUDIT.md`](DATASET_AUDIT.md)です。

## 主教師3系統

| 教師 | 確認できた学習系譜 | 局面数 | Meteoでの扱い |
| --- | --- | ---: | --- |
| NAGISA V3.1 | Tatara。奏乗WCSC36が公開した教師局面集を使用 | 奏乗系の約145億局面を継承した規模 | 奏乗と誤差・データ分布が相関し得る。同じ145億を独立な2票・290億局面とは数えない |
| 奏乗 WCSC36/TSEC7系 | bullet-shogi。DL水匠で評価値を書換え、静止探索処理、重複除去 | 240億から約145億。公開unique版は正確に14,668,949,437 record | 暫定anchor候補。ただし公開データと手元TSEC7の完全同一性はhashで証明されていない |
| 水匠11 | SFNNwithoutPSQT、HalfKAv2-1024_8_64。Ryfamate最新、DL水匠、AobaZeroの3モデルによるアンサンブル評価値データ | 公開資料では不明 | 3教師利用の重要な先行例。ただし統合式・重み・校正係数は公開されていない |

NAGISAの[公式GitHub v3.1 release](https://github.com/keinoda/YaneuraOu/releases/tag/nagisa-v3.1)は、
YaneuraOu 9.60、`HalfKA_hm2 1024x16x64 / LayerStack 9`、`FV_SCALE=28`、
`progress8kpabs`、source commit `640f46561455436641b2eafb6fb75dfbeaf21f3f`を明示しています。
v3.1で変わったのはSPSA調整済み探索parameterで、評価関数とNNUE構成はv3から不変です。
2026-08-13時点で同repositoryにNAGISA v4の公開release・tag・branchは確認できないため、v4は
教師系譜へ推定登録せず、適法なexact artifact取得後のsealed対戦targetとしてのみ扱います。

水匠11の[公式追加アピール](https://www.apply.computer-shogi.org/wcsc36/appeal/Suisho/appeal2.pdf)は、
3モデル名、極端な勝率帯が3割以上あること、それを削ると弱くなったこと、Ponanza定数を大きくして
int8飽和を抑えたことまで記載しています。したがってMeteoでも、勝率1%以下・99%以上を機械的に
捨てません。exact mate・終局値は別種の証明信号として扱い、通常の有限CPへ無理に平均しません。

## 公開局面データの再利用判定

| corpus | 現在確認したidentity | 内容 | ライセンス | 現時点の結論 |
| --- | --- | --- | --- | --- |
| 奏乗 `datasets_1` | revision `4dfad115d4a808ebe20b6f65f6416ad75a69a6e7`、1,983,794,202,520 bytes、49,594,855,063 records | nodchip 3種、AobaZero、DL水匠policy展開。qsearch・value書換え・shuffle・dedup済み。256 record全Move16=0 | card licenseなし。利用者がローカル利用可能と確認 | 明示acknowledgement付きlocal-only。NAGISA型NNUEでは旧scalarを直接value教師にする。policyは存在せず、派生checkpointは公開しない |
| 奏乗 `datasets_2` | revision `b2a90707a7afc932ae05f1923219ff20c6b5d138`、4,477,806,959,080 bytes、111,945,173,977 records | `datasets_1`＋DL水匠policyの2段階展開、qsearch・shuffle・dedup | manual gated、licenseなし、binary未probe | 名目上1,000億超だが未使用。HF access、権利、probe、`datasets_1`との横断dedupが必須 |
| DL水匠15b unique | revision `5da309f4de4091cfb004eff94da97d49e3268aa2`、bin 586,757,977,480 bytes、14,668,949,437 records | qsearch・評価値書換え・重複除去済み。抽出100 recordは全てMove16=0 | repositoryにdataset card/licenseなし | 価値事前学習の時間を大きく短縮できる可能性はあるが、利用許諾とvalue-only loaderが揃うまで投入禁止。policy局面数には数えない |
| DL水匠15b precursor | revision `aefe43c547f4230f9d5d16dda671c61c7c28b796`、raw PSV 666,682,171,480 bytes、16,667,054,287 records | Hao/tanuki qsearch shuffle後、DL水匠15bでvalue書換え、Eval_Coef=600。512件全Move16=0 | card licenseなし。利用者がローカル利用可能と確認 | local-only value教師として技術的に利用可。ただし`datasets_1`と祖先重複が大きいためprimaryへ加算せず予備にする |
| AobaZero公開自己対局棋譜 | 2026-08-11 21:25時点 `w4753`、78,685,549局 | 世代別CSA。近年は探索・NNのpolicy/valueコメントを含むが、初期121,032局はNN未使用 | GitHub内コードのGPL/Public Domain表記は確認できるが、外部Drive棋譜への明示適用は未確認 | 権利確認後、世代ごとにparseして局面源へ使用。現3教師で再解析し、直接policy教師にはしない |
| tanuki-.nnue-pytorch-2024-07-30.1 | revision `59fd246d3f85d51707a62c89531564a1a8aeb793`、8,000,057,305 records | 深さ9、未shuffle、qsearch leaf未置換 | MIT | 局面生成は省ける。qsearch、重複除去、split、現3教師による再labelが必要 |
| shogi_hao_depth9 | revision `7bc19a9e880ea307a52c57f57ea6c752301b25bc`、8,000,074,486 records | 深さ9、未shuffle、qsearch leaf未置換 | MIT | 同上。奏乗系の祖先でもあるため、独立な新規局面として二重計上しない |

現在のローカル空きは約87GiBで、unique版だけでも約586.8GBです。全量を内蔵SSDへ取得する計画は
開始しません。利用許諾を確認できた場合も、外部ストレージまたはimmutableなremote shardから
選択範囲をstreamし、revision・filename・range・SHA-256・record countをreceiptへ残します。

最重要の形式差は、公開value corpusの抽出recordで`move16=0`だったことです。旧
`PackedSfenValueDataset`はpolicy教師なので、この形式を明示的に拒否します。合法手を捏造して
one-hot policyを作ってはいけません。現productionは次の専用経路です。

```text
public value-only PSV shard
  -> user-attested local-only receipt
  -> revision・size・LFS SHA-256固定
  -> 最終shard tail 1,000,000局面をheld-outへ先に固定
  -> Move16=0、score、game_result、手番、record alignmentを検証
  -> policy headを持たないTatara NNUEへscalar valueを直接入力
  -> NAGISA互換SFNNをランダム初期値から学習
  -> やねうら王互換nn.binへ変換
```

つまり公開scalarはNAGISA型のvalue学習を直接省力化しますが、「全合法手の最善・次善手分布」を
代替しません。bootstrap後にpolicy型を別途研究する場合は、NAGISA・水匠11Plus・奏乗TSEC7が候補を
提案し、単一scorerが等予算で各候補を再探索した新しいscore matrixから生成します。

AobaZero棋譜は局面多様性と自己対局由来の攻防を得る種として有望です。ただし公式の学習履歴だけでも、
`w4201`からNNの生Value/Policyを記録、`w4659`から同一盤面の手順前後を区別しない変更、`w4747`から
全局24点法、という契約変更があります。選択archiveのweight範囲をreceiptへ固定し、完全な1局をsplitの
最小単位にし、千日手・連続王手・入玉はその世代の規則で再構成します。78,685,549は「棋譜数」であり、
Meteoの一意学習局面数や高信頼label数へそのまま加算しません。

## 水匠11の3モデル・アンサンブルをどう参考にするか

水匠11の正確な式は非公開なので、同じものを再現したとは主張しません。公開されている
[INUGAMI詳細アピール](https://www.apply.computer-shogi.org/wcsc35/appeal/INUGAMI/INUGAMI_appeal_detail.pdf)
には、各GPU教師の勝率を評価値へ変換し、複数評価値の相加平均で学習labelを置換する手順が
明記されています。これをRound Dの比較armとして次のように固定します。

教師`i`の手番側期待勝率を`p_i`、校正後の信頼度重みを`alpha_i`、Ponanza係数を`C`とすると、

```text
e_i = C * log(p_i / (1 - p_i))
e_score_mean = sum(alpha_i * e_i)
p_score_mean = sigmoid(e_score_mean / C)
```

です。これは生の異尺度CP平均ではなく、同じ確率規約から共通尺度へ変換した後の平均です。比較対象に、

```text
p_probability_mean = sum(alpha_i * p_i)
```

も置きます。両者は一般に同じ値になりません。実装は
`teacher_value_ensemble.py`で別modeとしてreceiptへ残します。`p=0/1`だけは有限CPへ変換できないため、
明示epsilonで数値clipし、clipした教師IDを記録します。0〜1%・99〜100%帯を丸ごと削除する処理では
ありません。

具体的な再現可能手順は次です。

1. 同じ局面集合と手番視点を3教師へ与え、各教師のraw出力、モデルhash、探索条件を別々に保存する。
2. mate、終局、`lowerbound`、`upperbound`を有限CPとは分離し、通常スカラー合成へ入れない。
3. 教師ごと・探索予算ごとに、calibration splitだけで`p_i = P(win)+0.5P(draw)`へ単調校正する。
4. 上記2方式を別armで計算し、重みはまず等重み、その後は独立した深探索regretからfitする。
5. 変換後の極端値は削除せず、clip件数、W4飽和率、FV_SCALE、Ponanza係数をreceiptへ残す。
6. 同一初期重み・batch順で単一教師、単一scorer、2種の合成を比較し、fixed-time棋力まで勝った方式だけ昇格する。

水匠11公式資料から確定できるのは「3モデルでアンサンブルした評価値データを使った」ことまでで、
この順序、平均空間、等重みか否かは推定です。Meteoの比較armは追試可能な独自方式であり、
水匠11の複製とは表記しません。

この比較はbootstrap後の教師再解析研究です。現在のvalue-only productionは三教師値を平均せず、
公開PSVに保存された単一DL水匠scalarを学習します。

1. A-N/A-W/A-Sで各教師単独のlabel品質を測る。
2. B-N/B-W/B-Sで3教師が候補を出し、1教師だけが全候補を共通尺度で採点する。
3. Round Dで「共通尺度評価値平均」「勝率平均」「既存interval robust」を比較する。
4. 同一局面・seed・step・モデルで、held-out regret、fixed-node/time、worst-group、arenaを比較する。
5. Round DがA/Bを実測で超えた場合だけ、合成を主経路候補へ昇格する。

NAGISAと奏乗は学習データの祖先が重なるため、常時各1/3という重みにはしません。水匠11も
DL水匠を祖先に持つため完全独立ではありません。重みを使う場合はモデル名で決めず、独立した
calibration splitでの深探索regret、探索安定性、mate事故率、局面群別校正誤差からfitし、最終
held-outには触れません。

## ひらがな水匠コードの位置づけ

`tayayan/HiraganaSuisho`のrelease `230906`（tag commit `2bfb0182...`、確認時HEAD
`744ddff1...`）も実コードを確認しました。MITで、`makebook.py`はUSI自己対局と定跡木のminimax更新、
千日手の引き分け処理を行います。一方でNNUE学習器、3教師の評価値校正・合成、policy/value蒸留は
実装していません。したがって定跡生成・USIプロセス管理の参考にはできますが、水匠11型アンサンブルの
根拠や学習コードとしては流用しません。

## 実行確認

```bash
uv run simajilord-shogi teacher-lineage
uv run simajilord-shogi learning-strategy
uv run simajilord-nnue index-corpus \
  --allow-user-attested-local-only
uv run simajilord-nnue prepare \
  artifacts/runs/meteo-nagisa-nnue-20260812-v1 \
  --nagisa-archive /path/to/NAGISA_V3.1-release.zip \
  --allow-user-attested-local-only
uv run simajilord-shogi fetch-public-psv-seeds \
  nodchip-shogi-hao-depth9 artifacts/public-seeds/hao-v1 \
  --file-count 8 --records-per-file 4096 --seed meteo-bootstrap-v1
```

最後のコマンドはMIT確認済みcorpusだけを許可し、HTTP `206`、固定revision、ファイル名、総bytes、
record-aligned range、ETag、range SHA-256、出力SHA-256をreceiptへ保存します。出力は
`optimizer_input_allowed=false`の局面種であり、元のdepth-9指し手・評価値・勝敗を学習labelにしません。
CPU教師で現行score matrixを付けた後、GPU上のMeteo学習へ移します。

`unique_source_positions`、`unique_labelled_positions`、`unique_retained_training_positions`、
`cumulative_positions_seen`を分離し、同じ祖先corpusやepoch反復で1,000億目標を水増ししません。
