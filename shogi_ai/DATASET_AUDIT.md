# Meteo 公開・ローカル学習データ総監査

更新: 2026-08-12 JST

## 結論

今回の監査範囲は、利用者から指定された配布物だけでなく、次までです。

1. Hugging Face の `sojoteam`、`penguinkumimanu`、`washiun`、`nodchip` が公開する
   将棋データ全26 repository
2. 水匠の公開資料で教師源として列挙されたAobaZero、GCT、水匠公開PSV、Qhapaq、
   Floodgate、電竜戦・WCSC棋譜、書籍付属データ
3. ローカルの生成済みJSONL、教師成果物、定跡データ

Meteoの現NAGISA型本学習に使う主sourceと、将来の再解析候補は次の二群です。

- 利用者がローカル利用可能と確認した奏乗 `datasets_1` 495.95億局面と旧DL水匠
  166.67億record。`datasets_1`をprimaryにし、`--allow-user-attested-local-only`を明示した
  ローカル限定NNUE経路で旧DL水匠scoreを直接value教師にする。Move16=0はpolicyではない。
- MITが明記されたnodchipの4 corpus。将来のPolicy+Value/自己対局経路では過去の指し手・評価値を
  捨て、現在のNAGISA・水匠11Plus・奏乗TSEC7で再解析する。

`datasets_2` は111,945,173,977 recordと、名目上は1,000億目標を単独で超えます。しかし、
binaryはHugging Faceのmanual gateでHTTP 401となり、license欄もなく、`datasets_1`を丸ごと
含む派生物です。アクセス承認、利用条件、binary probe、祖先横断dedupが終わるまでは、
「確認済みの候補」であって「使用可能な1,119.45億教師label」ではありません。

また、公開corpusを足し算してはいけません。同じnodchip局面が、DL水匠、奏乗、NAGISA、
Kanade、Fuka、JC26、NAGI、ponkotsuの各派生版に繰り返し現れます。Meteoは次を別々に数えます。

- `unique_source_positions`: 祖先横断で重複除去した元局面
- `unique_value_source_positions`: immutable PSVのvalue教師を持つ一意source局面
- `unique_search_labelled_positions`: 将来、現在の3教師で再解析が完了した一意局面
- `unique_retained_training_positions`: 品質gateを通ってreplay bufferへ残した局面
- `cumulative_positions_seen`: epoch反復を含むGPUが実際に見た累積局面

1,000億目標は最後の累積値です。未取得・未解析・重複した公開recordで水増ししません。

## 監査方法

Hugging Faceは2026-08-11時点のHub APIから、author配下のrepositoryを全件列挙しました。
各repositoryについて、`revision`をSHAに固定し、再帰treeを最後のcursorまで取得して、
ファイル数とbytesを集計しています。Dataset Viewer APIの`/is-valid`も確認しましたが、PSV/HCPE
binaryは4つの代表例すべてでpreview、viewer、search、filter、statisticsが非対応でした。
したがってViewerの推定行数は使わず、PackedSfenValueだけはpayload bytesを40で割った
record数を使用しています。

Range probeでは、盤面decode、Move16、合法性、score、game resultを確認しました。これは
「historical Move16をpolicyとして使用してよい」という確認ではありません。現NAGISA型NNUEでは、
review済みMove16=0 corpusのscalar scoreだけをvalue教師として使います。将来のPolicy+Value経路では
盤面だけを採用し、現3教師の共通条件探索から新しいpolicy/valueを作ります。

license欄が空のrepositoryは、公開downloadできるだけでは再配布許諾とみなしません。利用者が
確認した2 corpusはローカル限定で扱い、そこから作るcheckpointも公開・販売しません。公開する
場合は、元配布者から派生成果物を含む許諾を別途得る必要があります。

## Hugging Face全26 repository

### sojoteam: 2/2確認

| corpus | 固定revisionと実測規模 | 系譜・形式probe | 権利・アクセス | Meteoでの判定 |
| --- | --- | --- | --- | --- |
| [`sojoteam/datasets_1`](https://huggingface.co/datasets/sojoteam/datasets_1) | `4dfad115…`、102 entries、1,983,794,202,520 PSV bytes、49,594,855,063 records | nodchip 3種＋AobaZero＋DL水匠policy 10%以上の展開局面。qsearch、DL水匠でvalue書換え、shuffle、dedup済み。256件probeは全て盤面正常、Move16=0 | card licenseなし。利用者がローカル利用可と確認 | 現NAGISA型本学習のprimary value source。旧scalarを直接使うがpolicy labelは0件。local-only checkpointのみ |
| [`sojoteam/datasets_2`](https://huggingface.co/datasets/sojoteam/datasets_2) | `b2a90707…`、226 entries、4,477,806,959,080 PSV bytes、111,945,173,977 records | `datasets_1`に、DL水匠policy 10%以上を2段階展開してqsearchした局面を追加し、shuffle・dedup | manual gated、licenseなし。READMEとtree metadataは公開、binary RangeはHTTP 401 `GatedRepo` | 未使用。HF access、明示条件、Move16/score probe、`datasets_1`との重複receipt完了後に再判定 |

### penguinkumimanu: 18/18確認

| corpus | 固定revision、payload、record | 系譜・probe | 判定 |
| --- | --- | --- | --- |
| [`sample_Knowledge_distilled_dataset_by_Kanade`](https://huggingface.co/datasets/penguinkumimanu/sample_Knowledge_distilled_dataset_by_Kanade) | `197c00f5…`、PSV 8,802,292,200 bytes＝220,057,305 records、HCPE 4.18GB | 水匠5由来の同じ約1.1億局面をqsearch PSV、非qsearch PSV、HCPEで重複表現。Kanade、`Eval_Coef=285`。qsearch sampleはMove16=0、非qsearch sampleは64件中42件のmoveが保存盤面に非合法 | 小型変換テストには有用。3表現を別局面として加算禁止。現policyには不使用 |
| [`Knowledge_distilled_dataset_by_Kanade_20250805`](https://huggingface.co/datasets/penguinkumimanu/Knowledge_distilled_dataset_by_Kanade_20250805) | `5405f058…`、320,000,000,000 bytes＝8,000,000,000 records | tanuki由来、非公開Kanadeで指し手・値を書換え、`C=285`。probe 128件中44件のmoveが保存盤面に非合法 | licenseなし。旧move/valueは使わず、祖先tanuki MIT版を優先 |
| [`ensenble_Kanade_Origine`](https://huggingface.co/datasets/penguinkumimanu/ensenble_Kanade_Origine) | `9152f4c5…`、324,001,802,160 bytes＝8,100,045,054 records | READMEなし。sampleでは64件中2件のmoveが非合法 | provenance・license不足。使用しない |
| [`tanuki-.nnue-pytorch-2024-07-30.1_Shuffled_qsearch`](https://huggingface.co/datasets/penguinkumimanu/tanuki-.nnue-pytorch-2024-07-30.1_Shuffled_qsearch) | `e4fe45c8…`、340,002,292,200 bytes＝8,500,057,305 records | tanuki祖先のqsearch/shuffle派生と名称から推定。READMEなし、sample Move16=0 | licenseなし。MITの元tanukiをMeteo側でqsearchする方を優先 |
| [`Knowledge_distilled_dataset_by_Fuka2025Q2-40b_qsearch`](https://huggingface.co/datasets/penguinkumimanu/Knowledge_distilled_dataset_by_Fuka2025Q2-40b_qsearch) | `03f5955e…`、340,002,292,200 bytes＝8,500,057,305 records | tanukiをHaoでqsearch/shuffle後、Fuka2025Q2-40bでvalue書換え、`C=600`。Move16=0 | licenseなし。position overlap比較用。現在の3教師labelを優先 |
| [`shogi_hao_depth9_Shuffled_qsearch`](https://huggingface.co/datasets/penguinkumimanu/shogi_hao_depth9_Shuffled_qsearch) | `fa24a749…`、324,002,979,440 bytes＝8,100,074,486 records | READMEなし。Hao祖先のqsearch/shuffle派生、Move16=0 | licenseなし。MITの元HaoをMeteo側で処理 |
| [`Knowledge_distilled_dataset_by_DLSuisho15b`](https://huggingface.co/datasets/penguinkumimanu/Knowledge_distilled_dataset_by_DLSuisho15b) | `aefe43c5…`、raw PSV 666,682,171,480 bytes＝16,667,054,287 records。repository 855.10GBのうち188.41GBは水匠5 source archive | Hao/tanukiをqsearch/shuffle、DL水匠15bでvalue書換え、`C=600`。512件全Move16=0 | 利用者確認済みのローカル限定value source。`datasets_1`と祖先重複するためprimaryへ加算せず予備。bundled source archiveも追加局面に数えない |
| [`Knowledge_distilled_dataset_by_JC26`](https://huggingface.co/datasets/penguinkumimanu/Knowledge_distilled_dataset_by_JC26) | `040667c1…`、340,002,292,200 bytes＝8,500,057,305 records | tanuki qsearch後、Just Counter 26歩でvalue書換え、`C=600`。Move16=0 | licenseなし。同じtanuki局面の別valueであり、独立局面ではない |
| [`Knowledge_distilled_dataset_by_NAGI`](https://huggingface.co/datasets/penguinkumimanu/Knowledge_distilled_dataset_by_NAGI) | `32b58ed7…`、340,002,292,200 bytes＝8,500,057,305 records | tanuki qsearch後、非公開NAGIでvalue書換え、`C=600`。Move16=0 | licenseなし。同じ祖先のablation候補だけに留める |
| [`nnue-data_16B_shuffled`](https://huggingface.co/datasets/penguinkumimanu/nnue-data_16B_shuffled) | `34d81d0f…`、658,457,098,080 bytes＝16,461,427,452 records | READMEなし、sample Move16=0 | 系譜・license不足。使用しない |
| [`Knowledge_distilled_dataset_Same_sign`](https://huggingface.co/datasets/penguinkumimanu/Knowledge_distilled_dataset_Same_sign) | `5ee6beed…`、285,376,545,560 bytes＝7,134,413,639 records | READMEなし、sample Move16=0 | 選別規則・license不足。使用しない |
| [`Knowledge_distilled_dataset_by_RyfamateTSEC4`](https://huggingface.co/datasets/penguinkumimanu/Knowledge_distilled_dataset_by_RyfamateTSEC4) | `59f33fb7…`、`.gitattributes` 2,461 bytesだけ | data、READMEともになし | 空repository。水匠11のRyfamate教師データの代替にはならない |
| [`aoba_psv`](https://huggingface.co/datasets/penguinkumimanu/aoba_psv) | `e620bee3…`、117,434,634,960 bytes＝2,935,865,874 records | AobaZero archive 0～7673をPSV化、qsearch/shuffle、publisher申告で2～3%重複、未蒸留。Move16=0 | licenseなし。AobaZeroと`datasets_1`の祖先重複あり。世代metadataが必要 |
| [`Knowledge_distilled_dataset_by_ponkotsuWCSC35_unique`](https://huggingface.co/datasets/penguinkumimanu/Knowledge_distilled_dataset_by_ponkotsuWCSC35_unique) | `e75a5807…`、195,585,992,400 bytes＝4,889,649,810 records | nodchip 3種をqsearch/dedup後、ponkotsu WCSC35でvalue書換え、`C=600`。Move16=0 | licenseなし。局面源としてはMIT祖先を優先 |
| [`dlsuisho_unique_expand_psv_from_policy_qsearch_dedup`](https://huggingface.co/datasets/penguinkumimanu/dlsuisho_unique_expand_psv_from_policy_qsearch_dedup) | `9691ad7e…`、1,342,731,791,280 bytes＝33,568,294,782 records | DL水匠uniqueからpolicy 10%以上の手を展開、qsearch、dedup。sampleはMove16・score・resultが全て0 | labelなしのposition expansion。`datasets_1`の主要祖先であり、両方を足さない |
| [`generic_ponkostu_wcsc36_Pre-learning`](https://huggingface.co/datasets/penguinkumimanu/generic_ponkostu_wcsc36_Pre-learning) | `2c4a70b5…`、HCPE 543,656,003,150 bytes | AobaZero、Hao、tanuki、水匠5入玉をmerge、dedup、shuffle。ponkotsu資料の実使用は約27億だが選別規則は不明 | full mergeを「ponkotsu実使用27億」と呼ばない。ancestor単位でsplit/dedupが必要 |
| [`generic_ponkostu_wcsc36_Pre-learning_Raw_data`](https://huggingface.co/datasets/penguinkumimanu/generic_ponkostu_wcsc36_Pre-learning_Raw_data) | `2be7b0c4…`、585,466,719,840 PSV bytes＝14,636,667,996 records | 同祖先をdedupのみ、未shuffle。READMEはdownload/use自由と明記。Aoba sampleはscore=0、他sampleには合法move | 公開利用文はあるが上流系譜を保持。元MIT/Aoba世代別sourceを優先し、比較用に限定 |
| [`Knowledge_distilled_dataset_by_ponkotsuWCSC36`](https://huggingface.co/datasets/penguinkumimanu/Knowledge_distilled_dataset_by_ponkotsuWCSC36) | `526bd42a…`、`.bin` 567,199,378,240 bytes＝14,179,984,456 records | READMEは約145億とする一方、名称はWCSC36、本文はponkotsu WCSC35で書換えと記載。sampleはWCSC35 unique sampleと一致 | identity説明に矛盾、licenseなし。使用しない |

### washiun: 2/2確認

| corpus | 固定revisionと実測規模 | 系譜・probe | 判定 |
| --- | --- | --- | --- |
| [`Knowledge_distilled_dataset_by_DLSuisho15b_unique`](https://huggingface.co/datasets/washiun/Knowledge_distilled_dataset_by_DLSuisho15b_unique) | `5da309f4…`、586,757,977,480 bytes＝14,668,949,437 records | 奏乗資料の240億→dedup約145億と整合。100件全Move16=0 | README/licenseなし。利用許諾を別途得るまでblocked。`datasets_1`・旧160億と独立加算禁止 |
| [`Knowledge_distilled_by_DLSuisho15b_add_aobazero_unique`](https://huggingface.co/datasets/washiun/Knowledge_distilled_by_DLSuisho15b_add_aobazero_unique) | `f2e98309…`、679,399,716,760 bytes＝16,984,992,919 records | 上記にAobaZeroを加えた名称。READMEなし、sample Move16=0 | license・世代範囲不明。`datasets_1`や`aoba_psv`と強く重複するため使用しない |

### nodchip: 4/4確認

| corpus | 固定revisionと実測規模 | 公開契約 | Meteoでの判定 |
| --- | --- | --- | --- |
| [`tanuki-.nnue-pytorch-2024-07-30.1`](https://huggingface.co/datasets/nodchip/tanuki-.nnue-pytorch-2024-07-30.1) | `59fd246d…`、320,002,292,200 bytes＝8,000,057,305 records | MIT。depth 9、未shuffle、非qsearch。sampleは合法Move16 | 直接Range sampling可。board split→qsearch→dedup→現3教師relabel |
| [`shogi_hao_depth9`](https://huggingface.co/datasets/nodchip/shogi_hao_depth9) | `7bc19a9e…`、320,002,979,440 bytes＝8,000,074,486 records | MIT。depth 9、未shuffle、非qsearch。sampleは合法Move16 | 同上。旧move/score/resultは捨てる |
| [`shogi_suisho5_depth9`](https://huggingface.co/datasets/nodchip/shogi_suisho5_depth9) | `a399f456…`、repository 184,925,809,788 bytes。直接読めるvalidation `shuffled.bin`は289,626,920 bytes＝7,240,673 records | MIT。qsearch PV leaf・shuffle済み。validation sampleはMove16=0 | P0の低コストqsearch済み局面源。full multipart archiveは外部storageがある場合だけ |
| [`shogi_suisho5_depth9_entering_king`](https://huggingface.co/datasets/nodchip/shogi_suisho5_depth9_entering_king) | `441f1492…`、20,000,241,760 bytes＝500,006,044 records | MIT。Floodgate 2015～2024の入玉開始局面、未shuffle・非qsearch。sampleは合法Move16 | 入玉worst-groupを意図的にoversample。rule/history groupを保って現3教師relabel |

注: 入玉corpusの正確値は全129 entryをcursor最終ページまで再集計したものです。以前の
500,061,044という値は誤りで、500,006,044へ訂正しました。

## Hugging Face以外の全関連源

| source | 現在確認できる内容 | 権利・品質上の注意 | Meteoでの用途 |
| --- | --- | --- | --- |
| [AobaZero](http://www.yss-aya.com/aobazero/) | 2026-08-11 21:25時点78,685,549局、w4753。121,032局目からNN使用。w4201以降は生policy/valueも記録。w4659、w4747、w4750で履歴・入玉規則・draw valueが変更 | archiveへの明示licenseは未確認。root noiseにより実際の着手は最善とは限らない | weight世代・規則ごとの局面多様性、RL比較。現教師で再解析 |
| [Aoba駒落ち](http://www.yss-aya.com/komaochi/index.html) | 13,002,907局、w1250。平手を含む7手合い。500,008局目からNN、publisherはw40/90万局以降を推奨。w745～w761相当は既知bugで削除 | standard Meteoに駒落ちを混ぜない。archive license未確認 | 平手だけを通常候補。駒落ちは将来のvariant/rule test用 |
| [Aoba振り飛車](http://www.yss-aya.com/furibisha/) | 22,028,029局、w2195。100,029局目からNN。希望する飛車筋とstrategy bonusを持つ。publisherは「最善手だけなら最大訪問数を使う」と明記 | played moveをbest moveにしない。bonus込みvalueを通常勝率と同一視しない | 振り飛車・対抗形の希少groupと候補発見。現3教師で再解析 |
| [GCT WCSC31公開データ](https://tadaoyamaoka.hatenablog.com/entry/2021/05/06/223701) | HCPE/HCPE3。1600～2400 playoutの世代別self-play、Floodgate26手開始、互角局面、AobaZero、入玉、電竜戦・local gameを含む | mutable Drive。過去世代を常時混ぜたと記載。mate除外版もある。全shard hashとgeneration splitが必要 | HCPE3 visit分布の比較、RL diversity、入玉・実戦失敗局面。現在のalpha-beta nodesとは混同しない |
| 水匠 `Suishopsv-150m.rar` | 水匠5、1手200万nodeの棋譜由来、約1.5億局面 | Driveのexact file/hash、Move16、qsearch、重複率、利用条件が未監査 | 深い旧探索局面の候補。取得後も現教師relabel |
| 水匠 `Suisho10Mn_psv.7z` | 水匠5、1000万node×約1億局面 | mutable folder。exact assetとlicenseを固定できるまでdocumentation-only | 高品質holdout/候補源。全量よりstratified samplingを優先 |
| [Qhapaq Pretty Daabi](https://github.com/qhapaq-49/qhapaq-bin/releases/tag/dataset) | asset 11,304,930 bytes、SHA-256 `8887ae22…a6ae`。payload 36,608,160 bytes＝915,204 PSV records。旧学習では20回反復 | archive内規約は、大会・評価関数公開・商用service時の利用明記を条件に自由利用。20反復を18,304,080 uniqueとは数えない | 低weightの振り飛車strategy seed。board dedup後に現教師relabel |
| [Floodgate](https://wdoor.c.u-tokyo.ac.jp/) | 年・日別の完全CSA、rating履歴。長期の多様な実戦・入玉・千日手 | archive公開は確認、包括的な再配布licenseは未確認。壊れた棋譜、弱い/特殊engine、途中切断をfilter | 両者rating・complete terminal・rule別に抽出し、game単位split。実際の手はtrajectoryだけ |
| [電竜戦棋譜](https://denryu-sen.jp/denryusen/dr5_hardware3/dr1_live.php) | event別にdownload可能。該当eventページは棋譜利用に制限なしと明記。第4回ruleも自由利用を明記 | 大会・rule・開始局面を固定し、evaluation splitと学習splitを交差させない | 強豪同士の失敗局面・戦型分布・held-out候補。現教師で再解析 |
| WCSC棋譜 | CSA大会サイトに対局記録あり | 今回、包括的な学習・再配布条件は確認できず。電竜戦の条件を自動適用しない | 条件確認までは評価・研究用provenanceだけ |
| 書籍「強い将棋ソフトの創りかた」付属データ | dlshogi系で広く使用される有償付属物 | ローカル所持を確認できず、書籍購入者向け条件がある | 未使用。所有・付属license確認後に別ledgerへ追加 |

## 定跡・終盤表は教師局面と別管理

定跡bookは開始局面分布を作る資料であり、book moveを正解policyとしてそのまま学習しません。

| local source | identity | 用途 |
| --- | --- | --- |
| 新ペタショック | `artifacts/inputs/petashock-15018656-20260811/user_book1.ybb`、1,402,747,500 bytes、15,018,656局面版 | book hit中、離脱直後、再突入、book外を分けて開始局面を採取。水匠11Plus搭載相手とのarena用 |
| 対右玉雁木 | `.data/shogi_ai/books/right-king-gangi-478-20260808/`、368,456局面版3形式 | 右玉雁木worst-groupの開始局面。新ペタショックとの重複をboard hashで除去 |
| Finny tables | 現在は記事・実装調査対象で、ローカル学習corpusではない | exact endgame/rule oracleとして実装とtable identityを確認できた場合だけ別channelで使用 |

## ローカル全データ棚卸し

内蔵SSDの空きは2026-08-12確認時点で約81GiBで、巨大公開PSV/HCPEの全量copyはありません。
確認できたローカル学習物は
次だけです。

- `datasets/`: 約876MiB。旧ensemble実験10組で、1組11局、残りは各24局。large corpusではなく、
  旧比較成果物です。
- `artifacts/runs/meteo-clean-reset-20260811-v1/data/`: rules-clean済みJSONL。
  train系は各24 game/約2,948～2,985 sample、validation系は各16 game/約1,620～1,632 sample。
- 現3教師委員会benchmark: 6局と各手の深いscore matrix。これはRound Sのpipeline smoke候補であり、
  数億局面bootstrapではありません。
- `.data/shogi_ai/models/soujou-tsec7/`: 奏乗の評価関数と進行度係数。モデルであって局面corpusでは
  ありません。
- 新ペタショックと対右玉雁木book。定跡開始局面源でありvalue/policy labelではありません。

現在のNAGISA型本学習では、495.95億版を固定revisionから1 shardずつstreamし、sizeとLFS SHA-256を
照合して直接value教師として使います。同時に保持するcacheは1 shardだけで、optimizer/checkpointの
完全保存後に削除します。166.67億版は主要ancestorが重複するため、primaryへ加算せず予備corpusと
します。4.48TBの`datasets_2`を内蔵SSDへ取得する操作は行いません。

## 祖先重複図

```text
nodchip tanuki / Hao / Suisho5
  ├─ qsearch・shuffle派生
  │   ├─ Kanade / Fuka / JC26 / NAGI / ponkotsu の別value版
  │   └─ DL水匠15b precursor 約166.67億
  │       └─ dedup unique 約146.69億
  │           ├─ 奏乗WCSC36/TSEC7系
  │           ├─ NAGISA系の主要祖先
  │           └─ DL水匠policy 10%以上の展開 約335.68億
  ├─ AobaZero由来局面 ─ aoba_psv / GCT / ponkotsu pretraining
  └─ DL水匠展開＋AobaZero＋dedup ─ datasets_1 約495.95億
      └─ さらに2段階展開＋dedup ─ datasets_2 約1,119.45億
```

水匠11もDL水匠とAobaZeroを教師に含むため、3主教師は完全独立な3票ではありません。この相関は、
値を平均しないmulti-proposer/single-scorer方式を採る理由の一つです。

## 現在の採用順序

1. NAGISA型value-only NNUEをランダム初期化し、`datasets_1`の旧DL水匠scalarを直接streamする。
   policy head/lossはなく、Move16=0から着手教師を捏造しない。
2. 最終shardのtail 1,000,000局面をoptimizer開始前にheld-outへ固定する。全shardを2周し、
   3周目の端数を加えて累計1000億提示へ到達する。一意局面は49,594,855,063と別記録する。
3. 各shardでloss、局面/秒、source hash、checkpoint保存・再開を検証し、最新と直前だけ保持する。
   ±32000のmate近傍scoreもvalue教師として保持し、教師target
   `sigmoid(score/600)`、WRM `nnue2score=508`、推論時`FV_SCALE=16`をreceiptへ残す。
   量子化前後は同じheld-out局面でscore分布、確率分布、相関、回帰傾き、符号不一致を監査する。
4. NNUE bootstrap後、NAGISA・水匠11Plus・奏乗TSEC7とのfixed-node/time対局と弱点再解析を行う。
5. その後にleague self-play、教師対局の失敗局面、mate/千日手/入玉のrule oracleを反復追加する。
6. 旧Policy+Value経路を再開する場合だけ、A単一教師、B複数提案＋単一scorer、C全会一致、
   D3教師合成を同じ局面・seedで比較する。この経路では公開PSVは局面種であり、policyは再生成する。
7. `datasets_2`はアクセス・権利・probe・overlap auditが完了した時点で、全量ではなく
   stratified reservoirとして追加する。

巨大corpusはNAGISA型value-only bootstrapでは局面生成と教師探索を省けます。ただし最善・次善手の
policy分布を含まないため、将来のPolicy+Value学習や指し手比較では3教師の現在探索を省略できません。
