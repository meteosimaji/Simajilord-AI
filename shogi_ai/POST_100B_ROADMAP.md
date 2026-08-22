# Meteo 1,000億局面bootstrap完走後の強化ロードマップ

更新: 2026-08-13

この文書は、現在のvalue-only NNUE学習完了後に、MeteoをNAGISA V3.1、水匠11Plus、
奏乗TSEC7、公式artifact入手後の氷彗、さらに未公開NAGISA v4級と比較できる完成エンジンへ
進めるための正本です。
機械可読契約は次で出力できます。

```bash
uv run --project shogi_ai simajilord-shogi post-bootstrap-roadmap
```

「実装済み」「計測済み」「設計契約」「未実装」を混ぜません。現在の1,000億runは広いscalar
value評価関数のbootstrapであり、上位エンジン超えを証明するarenaではありません。

## 結論

主経路は、三教師の生評価値平均でも、実際に指した手の模倣でもありません。

```text
1,000億局面value-only NNUE bootstrap
  -> QAT/native exportの数値一致、再load、完全receipt
  -> 自己対局、最新vs直前、Meteo vs 3教師、3教師相互対局
  -> 実着手は局面源としてのみ使用
  -> 3教師とMeteoが候補手を提案
  -> 実験armごと1教師の尺度に固定し、全候補を等nodeで個別再探索
  -> 相手の応手候補も統合してmin/negamax backup
  -> qsearch leafを再評価し、exact・bound・mate・終局を分離保存
  -> child scalar value、ranking、区間制約、完全対局WDLをreplay学習
  -> candidate vs championと教師別非劣化gate
  -> 固定node、固定時間、解析、特殊局面を通過した候補だけ昇格
```

これはalpha-beta探索をpolicy improverとする、教師安定化付きfitted value / Expert Iterationです。
Policy Gradient、PPO、DPOをvalue-only NNUEに無理に当てはめません。

## 現行runの入口gate

2026-08-13の監査snapshotでは、Superbatch 5、約20.9億/1,000億presentations、直近約8.18万局面/秒、
直近loss約0.05066、残りETA約13.86日でした。これは時点値であり、現在値は次で再読みします。

```bash
uv run --project shogi_ai simajilord-nnue status-mlx shogi_ai/artifacts/runs/meteo-nagisa-nnue-20260812-v4
```

完走とみなす条件は次のすべてです。

- 100,000,000,000 presentationsをdurable coordinateで到達。
- source shard、held-out、数値phase、QAT、checkpointのreceiptが連続。
- 最新と直前のmodel・optimizerがhash検証後に再load可能。
- `nn.bin`、`progress.bin`、`eval_options.txt`を対応やねうら王が読み込める。
- native evaluatorとQAT forwardの許容誤差gateを通過。
- 最後のstepを自動的に最強とせず、arenaでchampionを選択。

学習中はbatch、optimizer、warm-up、QAT、cache、checkpoint間隔を変えません。中途変更は別recipeの学習になります。

## 1世代の対局配分

最初は2,400局/世代とし、すべて先後反転ペアを作れる偶数へ固定します。配分はablationする初期値です。

| 対局family | 比率 | 局数 | 使い方 |
| --- | ---: | ---: | --- |
| Meteo現行自己対局 | 25% | 600 | on-policy局面、評価急落、未知の弱点 |
| 最新candidate vs直前champion | 20% | 480 | 世代間の回帰と対策循環 |
| Meteo vs NAGISA/水匠11Plus/奏乗 | 30% | 720 | 各教師240局で固有の弱点を採掘 |
| 3教師相互対局 | 15% | 360 | 自己組合せを除く3ペアに120局ずつ |
| Meteo vs残り6登録モデル | 5% | 120 | 同一系譜外の盲点、技巧等の異質な局面 |
| 特殊開始局面 | 5% | 120 | TSEC定跡離脱、mate、千日手、入玉、対人解析 |

投了なし、評価値adjudicationなしで、詰み、ルール上の千日手結果、入玉宣言、その他の合法な終局まで進めます。
安全用手数上限で切れた局は強い終局labelにしません。

現行9モデルは全員を毎世代で使います。ただし役割は同一ではありません。NAGISA・水匠11Plus・奏乗は
完全対局相手、候補提案者、単一尺度anchorの比較armです。AobaNNUE・水匠5・振電3・技巧2・Háo・
tanuki-Lí-VENGEは先後反転の完全対局相手とoff-lineage弱点発見器であり、その異論手を別ablationの
proposal-only候補へ追加できます。6モデルのraw CPを主教師anchor値へ混ぜず、実着手も正解labelにせず、
候補は選択した主教師が同一条件で再評価します。9 rights IDすべてが対局receiptに現れなければ、その世代を
収集完了扱いにしません。同祖先の一致はconfidenceには使えても、独立票や局面数へ水増ししません。

対局速度は「NPS」一つでは管理しません。1局ごとのengine NPS、中央/p95所要時間、完了局/時、
保存局面/秒、教師再解析局面/時、CPU・メモリ・thermal・ディスク圧力を同時記録します。並列度
`1 / 2 / 4 / 8`を比較し、同じ探索budgetと正しい終局を守れる範囲で完了局/時を最大化します。
局譜生成本体は`MultiPV=1`とし、3教師の深い再解析は失敗・不一致・特殊局面だけへ集中します。
2026-08-13にNNUE runtime adapter、完全trajectory generator、速度receiptを実装し、学習中の
Meteo exportを対応やねうら王へ読み込ませて小標本を計測しました。1手20,000node、1 thread/engine、
bookなし、`MultiPV=1`、ルール終局までの条件です。

| 並列度 | 完了局 | 終局 | 完了局/時 | 保存局面/秒 | actor合計nodes/探索秒 |
| ---: | ---: | --- | ---: | ---: | ---: |
| 1 | 8 | 8/8詰み | 1,434.54 | 44.98 | 898,122 |
| 2 | 8 | 8/8詰み | 2,127.55 | 69.29 | 777,342 |
| 4 | 8 | 8/8詰み | 2,892.17 | 106.55 | 691,738 |
| 8 | 16 | 16/16詰み | 5,116.59 | 173.75 | 592,411 |

これは極小opening集の速度smokeであり、棋力、Elo、長時間thermal安定性の証明ではありません。
また3教師の深い全候補再解析、qsearch再評価、学習I/Oは含みません。2,400局を並列8の短期速度だけで割ると
局譜生成単体の約56分ですが、production全工程ETAとはしません。

参考として、既存の教師対局receiptを再集計すると、定跡なし・`MultiPV=1`・1手20,000ノードでは、
水匠11Plus自己対局16局が中央値1.975秒/局・p95 3.005秒/局・探索時間だけの直列上限1,737.5局/時、
水匠11Plus対NAGISA 16局が中央値2.143秒/局・p95 3.152秒/局・同1,648.2局/時でした。
この小標本はorchestration、保存I/O、長時間時のthermal制限を除き、Meteoの測定でもありません。
同じ上限を機械的に2,400局へ掛けると探索時間だけなら約1.38〜1.46時間ですが、本番ETAには使いません。
深い全候補再解析は別queue・別throughputとして測り、対局生成の値に混ぜません。

過去weightは最新と直前の2件だけを対局可能にします。古い世代はhash、対局結果、学習条件metadataだけを保存し、
学習weightを残しません。arenaを通過したcompact championは1件だけを配備用registryに置きますが、必ずこの2件の
どちらかへのcontent-addressed参照または同一内容のoptimizerなしprojectionとし、第3の独立した完全checkpointにはしません。

## 3教師の役割と再解析

NAGISAと奏乗は学習データの系譜が相関し得るため、独立な3票とは数えません。教師は値を平均する者ではなく、
候補手の提案者と反証者です。

1. NAGISA、水匠11Plus、奏乗、Meteoの上位候補を和集合にする。
2. 王手、駒取り、成り、王手回避、入玉関連、浅い全合法手screeningの上位を追加。
3. A-N/A-W/A-Sは各1教師の候補と評価、B-N/B-W/B-Sは全教師の候補を単1教師が評価。
4. 同一やねうら王、同一オプション、`MultiPV=1`、同一要求node、候補別`searchmoves`で探索。
5. 最上位候補後の応手を再び和集合にし、相手の最善応手でbackup。
6. qsearch PV leafへ移動後は古いroot scoreを使わずleaf自体を再評価。
7. `exact`、`lowerbound`、`upperbound`、`mate`、証明済み終局を分離し、生score平均は作らない。
8. 不一致、符号反転、mate衝突、1x/10xでtop-1変更、応手backupで逆転する局面へ追加計算を集中。

`LB(best) > max(UB(other)) + margin`なら全候補がexactでなくても順位を証明できます。重なる区間は中点に丸めず、
追加探索またはinterval lossへ回します。教師の`mate`scoreは通常valueと混ぜず、内部DFPN等の証明を要求します。

## value-onlyで行う強化学習

学習対象は手のIDではなく、各手後の子局面を正しい視点へ符号変換したqsearch leafのscalar valueです。
複数手の値が分かれば、policy headなしでranking lossを追加できます。

完全対局WDLは教師探索値を置き換えるものではなく、新しい真の信号です。終局から遠い中盤へ一様に強く伝播させると
分散が大きくなるため、WDL混合係数は`0.0 / 0.10 / 0.25`を同一データで比較し、held-outとEloで選びます。

| replay源 | 初期比率 | 目的 |
| --- | ---: | --- |
| 1,000億bootstrapの広域anchor | 50% | forgettingと全体calibrationを防ぐ |
| Meteo自己対局の失敗 | 15% | on-policyの新しい弱点 |
| Meteo vs教師の失敗 | 15% | 教師別の反例 |
| 教師不一致・相互対局 | 10% | 難しい戦略境界 |
| mate・終局・千日手・入玉 | 5% | 希少で正確さが必要な局面 |
| TSEC定跡離脱・対人解析 | 5% | 既知定跡対策と解析用途 |

candidateが棄却されても、教師再解析で有用と判定された反例局面はdatasetへ残します。
weightの昇格とdatasetの成長は別の状態遷移です。

## 段階と停止gate

| 段階 | 内容 | 出口条件 |
| --- | --- | --- |
| B0 | 1,000億baselineをfreeze・数値監査 | receipt完全、native/QAT一致、再load可能 |
| B1 | 三軸baseline | 固定node、固定時間、完成エンジ対局を分離記録 |
| B2 | 単一scorerパイロット | A-N/A-W/A-SとB-N/B-W/B-Sを同一局面で比較 |
| B3 | 完全対局の失敗採掘・深い再解析 | split漏洩なし、重複除去済みPSV、score-matrix receipt |
| B4 | value-only replay学習 | 複数seed、QAT、worst-group、held-out gate |
| B5 | 構造・探索大会 | NPS低下込みfixed-time Eloで構造選択、SPSA/time manager調整 |
| B6 | 継続league | 停止指示まで各世代をpromotion gateで反復 |

B2で複数提案が単一教師baselineを改善するかを先に測ります。3教師のrobust合成、全会一致のみ、局面群別routerは
対照群であり、A/Bを実測で超えるまで主経路へ入れません。

## 構造、量子化、高速化

同一label、seed、step、数値契約で、現1024 LayerStack9、水匠11型HalfKAv2-1024、ローカル奏乗TSEC7で確認した
HalfKA_hm2-2048 LayerStack9、共有FT+進行度stackを比較します。氷彗は公開された特徴量・LayerStack的分離・量子化の
方向だけをablationし、非公開topologyを推定しません。構造はNPSコスト後のfixed-time Eloで選びます。

現runの配備合否はfloat masterとの近さではなく、QAT forwardとnative exportの一致で決めます。監査時のnative/QAT勝率差は
MAE約`2.6e-9`、最大約`1.3e-8`でした。一方、float masterから配備graphへの差はMAE `0.02853`、p99 `0.13365`、
最大 `0.31461`でした。これはexportバグではなく、連続masterと量子化gridの診断差です。

公式NAGISA v3.1の`FV_SCALE=28`に対し、現Meteo bootstrapは自身のlabel校正・QAT契約として
`FV_SCALE=16`を固定しています。構造互換は数値尺度の同一を意味しません。稼働中に28へ変更せず、完走checkpointの
複製から16/28と再校正を別armで比較し、native/QAT誤差、held-out calibration、fixed-time棋力で選びます。

追加したread-only weight-range監査では75,231,385要素の範囲外が0件でした。現在の差をclip飽和だけで説明できません。
完走後の複製checkpointで、exact QAT、範囲projection、master/deployment consistency、`FV_SCALE`、batch、prefetch、cacheを
一項目ずつablationします。大幅な高速化の本命はdense 73,305 x 1,024 FT更新のsparse optimizer/gradient化とMetal fusionです。
さらにRAMを予約するだけでは演算量は減らず、swapの危険があります。

## 評価、解析、対人、昇格

評価関数単体の固定node、実用棋力の固定時間、各モデル本来の探索・定跡・設定を含む完成エンジの三系列を分けます。
formal gateは相手ごと最低500 opening pairs = 1,000局、先後500局ずつです。proxy、STC SPRT、LTC SPRT、長時間解析の順で
淘汰します。

候補は、直前championへ勝ち越し、NAGISA・水匠11Plus・奏乗それぞれへの非劣化、fixed-node/time、戦型・mate・千日手・
入玉・定跡離脱のworst-group、数値/QAT gateのすべてを通します。全主教師超えの宣言は、各教師へのスコアの95%信頼下限が
0.5を超え、版、hash、hardware、時間、bookを報告できる場合だけです。

NAGISA v4は2026-08-13時点で公式release、公開branch、再現可能な測定値を確認できないsealed targetです。
存在や強さを推測して教師台帳へ登録せず、v3.1への固定node・STC・LTCすべてでpaired scoreの95%信頼下限
`0.55`を暫定reserve gateにします。これはv4超えの証明ではありません。適法なv4 artifactまたは明示された
対局serviceが得られた時点で、実行ファイル・評価関数・progress・options・bookをhash固定し、同一opening、
同一hardware、同一持ち時間、先後反転で95%信頼下限が`0.5`を超えることを直接要求します。

Meteo NNUEは対応やねうら王へ載せます。strongest-playは`MultiPV=1`と検証済みtime manager、analysisはPV1の計算量を保護した
追加PV個別再探索、human-playは同じchampion weightで時間・nodes・book範囲・駒落ちにより手合いを調整します。
LLMはPV、候補差、ルールを説明するだけで、最善手を独自に選びません。

## 権利、氷彗、実装の現在地

現在の権利台帳はたけわらべを除く9モデルです。ローカル限定の水匠11Plusと奏乗TSEC7はraw label、元weight、派生checkpointを
許諾なしに公開しません。NAGISAの`nn.bin`もコピーしません。

氷彗と未公開NAGISA v4はsealed external targetで、現在はローカルartifactがありません。学習教師、自動対局相手、勝ち越し宣言に
使いません。適法なartifactまたは明示的なservice利用許可を得た後、binary/eval/progress/book/optionsのhash receiptを作って有効化します。

実装の現在地は次の通りです。

1. NNUE exportを対応やねうら王へatomic stageし、独立processで再loadするruntime/profileとcompact registryは実装済み。
2. NNUE同士を先後反転し、投了・評価値adjudicationなしでルール終局まで進める完全trajectory generatorと速度receiptは実装済み。
3. 3教師・Meteo・戦術手の候補和集合、候補別等node `searchmoves`、応手backup、bound/mateを保持するscore matrixは実装済み。実教師optionと探索独立性を最終補強中。
4. qsearch leaf変換後に旧scoreを捨て、単一anchorで再評価するlocal-only rescore bundleは実装済み。
5. exact scalarをincremental PSVへ変換し、1000億checkpointのoptimizer/QATを継続するMLX経路は実装中。calibration split、尺度固定、global 2世代保持の最終gateが残る。
6. 完全trajectoryから失敗局面を抽出し、score matrix、qsearch、replay、継続学習を一括実行するproduction executorは未実装。
7. external-vs-external結果を既存500 opening-pair promotion gateへ自動接続するCLIは未実装。
8. analysis用とhuman-play用のtime/ponder/book profileとSPSAは完走後の実測段階。

## 主な一次資料

- [氷彗 WCSC36 詳細アピール](https://www.apply.computer-shogi.org/wcsc36/appeal/hisui/hisui_detail.pdf)
- [WCSC36 決勝結果](https://www.computer-shogi.org/wcsc36/final.html)
- [水匠 WCSC36 アピール](https://www.apply.computer-shogi.org/wcsc36/appeal/Suisho/appeal.pdf)
- [水匠 WCSC36 第2資料](https://www.apply.computer-shogi.org/wcsc36/appeal/appeal_round2_260503.pdf)
- [奏乗 WCSC36 アピール](https://www.apply.computer-shogi.org/wcsc36/appeal/sojo/sojo_WCSC36_appeal.pdf)
- [NAGISA V3.1 GitHub release](https://github.com/keinoda/YaneuraOu/releases/tag/nagisa-v3.1)
- [NAGISA V3.1 公式BOOTH](https://booth.pm/ja/items/8639574)
- [Stockfish NNUE 公式資料](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html)
- [Expert Iteration](https://arxiv.org/abs/1705.08439)
- [AlphaZero](https://arxiv.org/abs/1712.01815)

公開文書が方向だけを説明し、数値、統合式、完全recipeを開示していない場合は、Meteoの仮説と公式事実を分離します。
特に氷彗の非公開topology、水匠11の3教師統合式、NAGISAの完全学習recipeは推測しません。
