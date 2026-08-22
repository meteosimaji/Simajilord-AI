# めてお - Meteo Shogi AI

**めてお**（英字表記: **Meteo**）は、独立して開発・配布するオリジナル将棋AIです。
このリポジトリだけで将棋エンジン、自己対局、学習、評価を管理し、Discord Botや
Simajilord-AIへの実行時依存はありません。CLIとUSIで単独利用でき、外部アプリケーションとは
公開されたプロセス境界を介して接続します。USIとcheckpoint metadataにも公開名「めてお」を
保存します。

たややん氏／水匠、やねうら王、WCSC公式資料の調査とMeteoへの反映方針は
[`RESEARCH.md`](RESEARCH.md) に分離して記録します。現在の学習方式、比較arm、receipt gate、
1,000億局面の累計提示目標は[`LEARNING_STRATEGY.md`](LEARNING_STRATEGY.md)、
完走後の自己対局・3教師対局・再解析・昇格・解析/対人配備は
[`POST_100B_ROADMAP.md`](POST_100B_ROADMAP.md)、
教師の祖先関係・公開局面の再利用可否・水匠11型アンサンブル比較は
[`TEACHER_LINEAGE.md`](TEACHER_LINEAGE.md)、Hugging Face同一配布者を含む全公開・
ローカルcorpusの固定revision、bytes、形式probe、重複系譜、利用順は
[`DATASET_AUDIT.md`](DATASET_AUDIT.md)に固定します。
MLXの量子化学習、統合メモリcache、非同期dataset prefetch、transactional checkpointから得た
CNN/Transformer/LLMにも再利用できる知見は
[`NUMERICAL_TRAINING_GUIDE.md`](NUMERICAL_TRAINING_GUIDE.md)へ分離しています。
外部モデルの利用可否は[`MODEL_RIGHTS.md`](MODEL_RIGHTS.md)、直接依存のライセンスは
[`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md)、実行済みの受入試験と
4種の実エンジン対局・蒸留は[`VALIDATION.md`](VALIDATION.md)、Floodgate参加前の条件は
[`FLOODGATE.md`](FLOODGATE.md)に記録します。

## 現在の本学習経路: NAGISA型value-only NNUE

現在のMeteo本学習は、旧MLX Policy+Value/MCTSモデルではなく、NAGISA V3.1と同じ公開構造の
value-only NNUE/SFNNです。学生重みはランダム初期化し、NAGISAからコピーするのは局面進行度を
9 bucketへ振り分ける`progress.bin`だけです。NAGISAの`nn.bin`はコピーしません。

```text
Soujou datasets_1（49,594,855,063 qsearch済み局面、Move16=0）
  -> 固定revision・各shardのsize/LFS SHA-256を検証
  -> 先に固定した1,000,000局面のheld-out tailを勾配から除外
  -> DL水匠scalar scoreだけをvalue教師としてstream
  -> 教師分布sigmoid(score/600)を保つscale-aligned WRM value loss
  -> ±32000のmate近傍scoreもvalue教師に保持（policy headは存在しない）
  -> CPUでTataraと同じPSV/特徴をdecodeし、local Apple GPU/MLXで学習
  -> HalfKA_hm2(Friend) 73305 -> 1024x2 -> 15 -> 64、LayerStack 9
  -> やねうら王用nn.bin + progress.bin + eval_options.txtへ変換
```

policy head、policy loss、MCTS、三教師探索はこの初回bootstrapにはありません。1000億目標は
`cumulative_presentations=100,000,000,000`で、約495.95億の一意sourceを2周し、残りを3周目から
読む計画です。同じ局面を2回読んでも一意局面を1000億とは報告しません。完了shardはcheckpointと
optimizerの完全保存・再読込確認後に削除し、raw checkpoint、Tatara量子化bin、やねうら王exportは
それぞれ最新と直前の2世代だけ保持します。

`Eval_Coef=600`は教師側の勝率分布`sigmoid(score/600)`にだけ使います。ネット出力は
`score/508`へ収束させます。`508=(127*64)/16`なので、量子化後のraw出力をやねうら王が
`FV_SCALE=16`で割ると教師scoreの尺度に正確に戻ります。`FV_SCALE=14`と`nnue2score=600`の組合せで
生じる約3.2%の系統的な縮小はありません。最初の1,024 optimizer stepはFTをFP16、denseをFP32で
立ち上げ、その後はTatara/やねうら王の整数round・shift・factorizer foldingと同じforward値を使う
STE-QATへ切り替えます。float masterとの差は診断として残し、QAT reference、速度用FP16-FT emulation、
実際にexportしたnetworkのnative推論が一致することをsmokeと全世代の合否条件にします。

本番はクラウドGPUを使わず、Apple M4 ProのMLX混合精度/QAT backendを使います。CPU companionは
pinしたTataraのPackedSfenValue decode、HalfKA_hm2特徴、progress8kpabs bucket、量子化、
やねうら王serializationを使います。MLX cacheは実測再利用量を保持できる24 GiBとし、現在shardの
学習中に次の1 shardを空き容量gate付きでprefetchします。ETAはcompute-only probeではなく、
実PSV・download・checkpointを含む`status-mlx`の実測throughputから再計算します。

```bash
uv run simajilord-nnue prepare artifacts/runs/meteo-nagisa-nnue-20260812-v1 \
  --nagisa-archive /path/to/NAGISA_V3.1-release.zip \
  --allow-user-attested-local-only

# Apple Silicon Mac上
uv run simajilord-nnue prepare-mlx artifacts/runs/meteo-nagisa-nnue-20260812-v1
uv run simajilord-nnue smoke-mlx artifacts/runs/meteo-nagisa-nnue-20260812-v1
uv run simajilord-nnue run-mlx artifacts/runs/meteo-nagisa-nnue-20260812-v1
uv run simajilord-nnue status-mlx artifacts/runs/meteo-nagisa-nnue-20260812-v1
uv run simajilord-nnue audit-mlx-weights artifacts/runs/meteo-nagisa-nnue-20260812-v1/checkpoints/step-XXXXXXXX
```

旧Policy+Valueコードは比較・対局・将来の自己対局研究用の互換経路として残しますが、その重みと
世代成果物は現Meteoの本学習checkpointではありません。
現NNUE完走後の主経路は旧Policy+Value checkpointへ戻すのではなく、やねうら王探索で作った
完全対局と深い子局面valueを現value-only NNUEへ戻す経路です。

## 1,000億bootstrap完走後のNNUE実行経路

完走exportは対応やねうら王と一緒にcontent-addressed runtimeへstageし、別プロセスで3回読み込み
smokeを行います。対局生成は必ず先後反転ペア、`MultiPV=1`、投了・評価値adjudicationなしで、
詰み・ルール上の千日手・CSARule27入玉宣言まで進めます。`max_plies`打切りは局面源にはできますが、
強いWDLラベルにはできません。

```bash
# exportをprivate runtimeへatomic stageし、hashとUSI optionを含めて検証
uv run simajilord-nnue stage-runtime /path/to/export /path/to/YaneuraOu \
  /path/to/runtime --profile-id meteo-candidate --threads 1 --hash-mb 64
uv run simajilord-nnue verify-runtime /path/to/runtime --repeat-load-smoke

# strict JSON configから完全trajectoryと速度receiptをcreate-only生成
uv run simajilord-nnue-games /path/to/games-config.json /path/to/game-bundle

# NAGISA・水匠11Plus・奏乗の候補和集を、各単一scorer尺度で個別探索
uv run simajilord-score-matrix /path/to/matrix-config.json /path/to/matrix-receipt.json

# qsearch leaf変換。この時点では古いscoreなので学習不可
uv run simajilord-nnue qsearch-leaves SOURCE.psv ENGINE ENGINE_CWD QSEARCH_OUTPUT
```

qsearch後は`rescore_qsearch_leaves()`で単一の固定anchorを使ってleaf自体を再評価します。exactな
centipawnだけを`ScalarValueLabel`にし、mate・bound・node不足は未解決queueへ分離します。
`build-incremental-psv`で`Move16=0`のvalue-only PSVにし、完走1,000億checkpointのmodelとRanger optimizerを
`prepare_incremental_mlx_run()` / `run_incremental_mlx()`で継続します。追加学習はQATフェーズを解除せず、
広域anchorとexact hard caseを不変の比率で混合します。これらのraw receiptは絶対パスや限定教師の
出力証拠を含むため`local_only=true` / `publication_allowed=false`です。

## 旧Policy+Value互換経路でできること

- `rsshogi` による標準将棋の合法手、二歩・打ち歩詰め、千日手、入玉宣言、詰み判定
- cshogi 実装に合わせた dlshogi 標準入力 `(62 + 57) x 9 x 9` と2187方策ラベル
- MLX/Metal 上の20ブロック256チャネルResNetと40ブロック512チャネルhybrid
- 全合法手の最低探索保証、prior floor、Dirichlet noiseを備えたPUCT MCTS
- 複数対局のleafを中央GPUへまとめるbatched MCTS自己対局
- 浅い自己対局局面を同じモデルで深く再解析するasymmetric teacher
- 深探索で最善手が逆転した局面、選択手、best Q、regret、発見simulationの保存
- actorとの探索差が大きすぎる教師をpolicy/valueとも段階導入し、逆転・大悪手を優先するMLX学習
- 固定probe loss、合法手内label smoothing、勾配clipによる学習崩壊の検出と拒否
- 40-byte PSVを全件RAMへ載せないmemory-map学習と教師源provenance
- checkpoint保存、厳密再読込、終局棋譜の全合法手replay検証
- 世代manifest、色替わりarena、95%信頼下限によるcandidate昇格・棄却
- 人間との端末対局、USIエンジン、別USI教師（水匠等）のプロセス接続
- USI/MCTS/外部教師のnodes、time、NPSと、MLX peak memory・探索木回収回数の記録
- 現在の空き統合メモリからMLX cacheと探索木上限を決め、圧迫前にsubtreeを回収する安全弁
- 相手ごとの局面・序盤prefix・次手傾向・深教師から見た弱点の永続profile
- 1/3/5手等の標準詰将棋の完全証明と、自己対局局面からの一意詰み採掘

`verify` は複数のニューラル対局を詰みまで完遂し、深再解析、学習更新、保存、再読込、
再度の詰みまでを一度に検証します。

重要: コード経路が完成していることと、重みが大会級に学習済みであることは別です。
リポジトリに同梱する初期checkpointはランダム重みです。水匠やWCSC上位へ勝つという主張は、
強い教師データで学習し、同一条件の多数局arenaで確認するまでは行いません。

旧Policy+Value研究経路の第一比較候補は`multi_proposer_single_scorer`です。NAGISA V3.1、
水匠11Plus、奏乗TSEC7が候補手と
応手候補を提案しますが、1局面の最終Q値は選択した1教師だけが同一条件・同一尺度で全候補を
`searchmoves`再探索して付けます。教師値の平均、多数決、alpha-betaのroot node配分のpolicy化は
行いません。暫定anchorは奏乗TSEC7ですが恒久固定ではありません。奏乗・NAGISA・水匠それぞれを
単独scorerにしたRound A/Bを同一局面・seed・budgetで作り、深いregret、10倍budget安定性、
mate見落とし、上位1%の大事故、校正、worst-group、fixed-node/time対局で選びます。

全候補がexactなら、等しい要求node budgetのQ値からsoftmax分布を作り、最善手だけでなく次善手との
差も学びます。`lowerbound`/`upperbound`は点へ丸めず区間のまま保持します。候補`a*`について
`LB(a*) > max UB(other) + margin`なら、全候補exactでなくても一意最善手を証明できます。
区間が重なるときはpolicy lossを無効にして追加探索へ戻し、値の区間だけ信頼できる場合は区間内で
損失0のhinge value lossだけを許します。教師報告のmateは再探索要求であり、内部solverが証明した
mate集合だけを強いset-valued policy/value targetにします。

既存`canonical-target-contract-v2`の三教師worst-regret委員会はRound Dの比較実験として残します。
全会一致局面はRound Cの高純度対照群、厳格な20局面は2x32 `smoke`モデルの保存・再開・符号・lossを
確認するだけの非昇格checkpointです。Round A（各単一教師）、Round B（全教師提案＋単一教師採点）、
Round C（全会一致）、Round D（頑健合成）を同一条件で比較し、実測で勝った方式だけをproductionへ
昇格します。契約は`simajilord-shogi learning-strategy`で機械可読JSONとして確認できます。

完了した3教師committee benchmarkからRound Sを作るときは
`simajilord-shogi build-unanimous-smoke BENCHMARK_ROOT OUTPUT`を使います。このbuilderはpair reportと
score-matrixのSHA-256を照合し、root・複数budget・応手再解析が全てexactで3教師の一意最善手が同じ
局面だけを採用します。実際に対局で選ばれた手、boundの中点、alpha-beta node配分、即詰み特殊値は
labelにしません。split・score matrix・校正・権利sidecarをcreate-onlyで出力しますが、単一定跡root、
履歴欠落、全合法手未網羅をreceiptにblockerとして残すため、生成checkpointは昇格できません。

入力は盤面119面だけでなく、直近8手、反復回数、連続王手、入玉宣言可否、絶対手数、履歴完全性の
46面を追加する`history-input-v2`です。同一SFENでも履歴が違えばtensorが異なります。既存stemは
そのまま移植し、新しい履歴stemをゼロ初期化するため、upgrade直後のplay出力は旧checkpointと
数値一致します。training-eligible v2 sidecarを実データから生成するproduction score-matrix builder、
全合法候補familyの強制再採点、proof certificate、独立held-out重複除去はまだ未実装です。上記の
全会一致builderはRound S専用であり、このproduction gateを解除しません。したがって現在のraw
NAGISA／水匠／奏乗出力やv1 sidecarから20x256本学習を再開せず、2x32の非昇格スモークだけを許します。
`train --canonical-teacher-only`もproduction builder・校正artifact再検証・独立held-out receiptが
実装されるまで常にfail closedし、手書きsidecarでゲートを越えられません。

応手gateは「相手に悪い応手もある」ことを不安定とは数えません。各候補で三教師が独立提案した
principal replyの和集合をそのarmの単一scorerが全件採点し、相手が選ぶ最小Qをbudgetごとにbackupした系列の
変動だけをreply instabilityとします。これにより、全合法応手の総当たりを避けながら、教師の
一教師だけが発見した強い応手も同じ尺度で比較できます。相手の疑問手で高くなる枝を理由に健全な
最悪応手評価を捨てず、同時に最善応手への脆弱性を平均で隠しません。

## 旧Policy+Valueモデル

| profile | 構成 | パラメータ | M4 Pro実測 | 用途 |
| --- | --- | ---: | ---: | --- |
| `smoke` | ResNet 2x32 | 114,556 | — | CI、受入確認 |
| `development` | ResNet 10x128 | — | — | 高速な実験 |
| `competition_v1` | ResNet 20x256 | 24,384,760 | 学習batch 32: peak約1.23GB | 公開dlshogiと同形のbaseline |
| `competition_v2` | hybrid 40x512 | 182,930,748 | 推論約0.74GB、学習batch 8: peak約3.43GB | 主力候補 |

`competition_v2` は residual convolution、SE、81-square attention、absolute/relative
position parameter、SwiGLU を組み合わせた本プロジェクトの実験モデルです。最新版公式
dlshogiの非公開重みと互換である、または同等棋力であるという意味ではありません。

`competition_v1` は公式dlshogi ResNetと同じ、62面の3x3・62面の1x1・57面の
1x1を加算するstem、盤上2187要素のpolicy bias、27チャネルvalue headへ揃えています。
ただし、PyTorch/ONNXからMLXへの重み変換器と実際の公開重みはまだ同梱していません。
公開checkpointの形式とライセンスを固定した後、convolutionのOIHW↔OHWI変換、
BatchNormの統計量、全層shape、同一SFENのpolicy/value数値一致を検証して取り込みます。

既存モデルの改造は次の三経路を使い分けます。

- 同一構造: 利用・再配布条件を確認した公開dlshogi ResNet重みをv1へ変換し、深教師データで継続学習
- 構造拡張: v1の対応するconvolution層をv2へ移植し、追加Transformer層は段階的に解凍
- 非互換: AobaNNUE・公開水匠・探索エンジンは重みをコピーせず、USI深解析から蒸留

価格を許諾の代用にはしません。公開checkpointへ入れられる資源と、正規入手・個別確認した
ローカル限定教師を分け、学習利用と公開可否を別々に判定します。

64 GiB Apple Siliconでの2026-08-09実測では、ランダム重みの20x256 neural MCTSは単局
100 simulationsで約105 NPS、32局面を各100 simulations読むbatchで合計約781 NPSでした。
40x512は単局30 simulationsで約38 NPSです。速度測定であり棋力比較ではありません。この差のため、
最初に大量教師を吸収する主力は20x256とし、40x512は同じheld-out arenaで上回ってから昇格します。

単局1億nodeは現状約266時間です。自動メモリ回収により総node数だけを伸ばすことはできますが、
直近の自動上限は約205万Python tree nodeであり、1億nodeの木を完全保持するものではありません。
実戦的な1億nodeには、compact native tree、置換表、tree reuse、非同期dynamic batchingが必要です。
計測方法とFloodgate到達条件は[`FLOODGATE.md`](FLOODGATE.md)に固定しました。

## bootstrap後の強化学習ループ

浅い自己対局は局面を広く作るactorであり、既定ではその浅いpolicyを直接教師にしません。

```text
batched actor games
  -> Meteo自身が到達した同一履歴を三教師が再解析（OPD）
  -> Meteo対全対戦相手と三教師相互対局も局面源へ加える
  -> 実際の着手は正解にせず、候補・相手最善応手を三教師が再採点
  -> 不一致・低prior浮上・大regret局面をさらに深く読む
  -> actor/teacher探索比に応じてdeep policy/valueを段階導入
  -> 固定probe悪化・非有限loss・過大勾配を検知
  -> 学習と重ならないNAGISA・水匠・奏乗等のreplayでpolicy/value非劣化を確認
  -> candidate（最新）vs champion（直前）の色替わりarena
  -> 独立検証とarenaの両方を通ったcandidateだけchampionへ昇格
```

これは、自己対局の結果を報酬にした方策反復を、より深い探索のpolicy/value教師で安定化した
AlphaZero型の強化学習ループです。この一連は`improve`で1世代以上を連続実行できます。各世代は
`generation-NNNNNN/manifest.json`へ段階、checkpoint SHA-256、学習loss、arena結果を保存し、
`state.json`のchampionポインタは昇格時だけ更新します。checkpoint本体は対局した「最新候補」と
「その直前」の最大2件だけを保持し、完了済みの古い管理対象candidateを削除します。manifestと
replayは小さい監査証跡として残します。色替わりarenaに未完局が1局でもあれば昇格せず、既定では
32以上の独立opening pairについてcluster-bootstrap片側95%下限が0.5を超える必要があります。

checkpoint lineageは親checkpointの全metadataや祖先を埋め込まず、SHA-256・step・model構成・
権利制約だけの固定深度identityを保存します。`metadata.json`は1 MiBを上限とし、再帰的な祖先
コピーや異常に大きいmetadataはcheckpoint作成前に拒否します。

自己対局に最大手数は既定で設けません。詰み、投了、千日手、入玉宣言まで続けます。
`--max-plies` はデバッグ用で、上限到達局は不完全データとして通常学習から除外します。
これにより上限の先にある長い勝ちを偽の引分として学習しません。

忘却を防ぐため、最新局面だけでなく次をanchor replayとして残します。

- 深再解析で浅い最善手が逆転した局面
- 現champion・過去champion・異種NNUE教師との終局済み対局
- 詰み証明局面、定跡、入玉、持将棋、長手数終盤
- ライセンス確認済みの実戦・公開教師データ

`train --anchor-replay ...` で複数世代を混ぜられます。candidate昇格時には固定anchorの
policy/value精度と過去championへの棋力を回帰検査します。

## 巨大定跡への対策

正規入手済みの新ペタショック`user_book1.ybb`は、15,018,656局面版を
持つローカル限定の局面源です。構造監査では索引全件、move領域の連続性、packed SFENの一意性、
登録手の合法性を確認します。対局相手として使う場合は`BookMoves=10000`、
`BookIgnoreRate=0`、`IgnoreBookPly=false`を固定し、現在の最大203手目までを16手で途中打切り
しません。0 nodeの着手だけをYBB索引の登録局面・登録候補手へ直接照合し、正のnodeを使った手は
定跡を外れた後の通常探索として別集計します。定跡外から再び登録局面へ転位した場合も再合流として
記録します。

定跡の手・評価値・深さは、そのままMeteoの教師ラベルにはしません。YBBの全局面を136 shardで
重複なく覆うローカル計画を使い、NAGISA・水匠11Plus・奏乗TSEC7をいずれも定跡なしで
20,000 node / MultiPV 8から再解析します。教師不一致、探索深度での最善手反転、詰み、防御、
定跡からの早期離脱、held-outで弱かった局面は200,000 node、さらに2,000,000 nodeへ上げます。
三教師が提案した候補手と応手候補の和集合を全教師が`searchmoves`で再採点し、合意できない局面を
平均化した中途半端な好手にはせず、
追加探索対象として残します。

実戦棋譜は局面生成と強化学習の勝敗信号には使えますが、実際に指された手を最善手ラベルとは扱いません。
定跡入口、登録手からの分岐、最初のbook miss、その直前直後を別々に抽出し、canonical scorerの
再解析結果だけをpolicy教師にします。固定したYBB索引または局面hashをheld-outへ先に隔離し、学習側へ
混入した場合は昇格を拒否します。全shard・三教師のreceiptと独立held-out receiptが揃うまでは、
canonical v2 optimizerのproduction gateを開きません。

## 相手の癖と弱点

`opponent-learn` は対局後にだけprofileを更新し、将来の手を誤って学習へ漏らしません。
同一局面で選びやすい手、直近16手のprefixからの次手分布、深教師から見た悪手とregretを
保存します。対局では相手予測をpolicy priorの最大30%に混ぜ、残り70%以上は堅牢な通常探索を
維持する`OpponentConditionedEvaluator`を用意しています。現在の標準`play`/`usi`コマンドはprofileを
自動読込しないため、相手適応を実戦で使う接続optionは未実装です。接続後も全合法応手のminimum
visitを残し、相手が突然最善応手を指しても枝を消さない設計です。

## 戦法指定と変則ルールの境界

技巧の10種の定跡を`--teacher-tag yagura`、`aigakari`などに分けて蒸留し、戦型別の棋譜・
評価を混同せず保存できます。`--initial-sfen`で指定局面から自己対局・arenaを始める旧経路は
smoke/debug専用として残していますが、同じ局面を反復してもcandidateは昇格しません。
ただし、現在のMeteo networkには戦法condition入力とUSIの「戦法を選ぶ」optionはまだありません。
実戦時の明示的な戦法選択には、権利確認済みopening bookかstrategy embeddingを追加し、通常棋力を
落とさないことを戦型別arenaで確認する必要があります。

現在のrules backendは標準将棋で、二歩と打ち歩詰めを合法手から除外します。`二歩あり`は
標準将棋/Floodgateと異なるvariantであり、既存Boardのflag一つでは切り替えられません。
実装する場合は、variant専用rules backend、局面schema、replay、model ID、テストを分離し、
標準将棋の教師・rating・checkpointへ混ぜません。

## 外部教師の権利スコープ

外部教師は`public_release_allowed`、`local_authorized_only`、`not_authorized`の三つに分けます。
価格だけで使用可否を決めません。現行9モデルのうち、NAGISA V3.1、AobaNNUE v1.1、水匠5、
振電3、技巧2、Háo、tanuki- Lí-VENGEの7件はoutput-onlyのpublic経路、正規入手して実物と条件を確認した
水匠11Plusとユーザー提供の奏乗TSEC7は、元binary・評価関数・raw labelsを公開しない
`local_authorized_only`教師としてローカル解析・蒸留・学習に使えます。ただし、それらを消費した
Meteo checkpointは権利者の明示許可
receiptがない限り公開できません。無料か有料かにかかわらず、版ごとに権利と出力条件を確認します。

`ExternalUsiTeacher` は現行9モデル（奏乗TSEC7、NAGISA V3.1、水匠11Plus、AobaNNUE v1.1、
tanuki- Lí-VENGE、水匠5、振電3、技巧2、Háo）を別processとして接続します。奏乗は公式
`sojo_tsec7` sourceからApple Silicon向けに
再現buildし、exact `nn.bin`・`progress.bin`の読込と20,000 node探索を確認しています。
実行ファイルや重みを本リポジトリへコピーしない境界です。教師ごとに次の許諾を明示し、
学習利用が未確認ならfail closedします。

`reanalyse-usi`は局面SFEN単体ではなく、`GameRecord.initial_sfen`と
`moves[:PositionSample.ply]`を合法に再生し、対象局面と完全一致した履歴を
`position startpos moves ...`または`position sfen ... moves ...`で教師へ渡します。
sidecarの`history_mode=game_prefix`がこの境界を示します。`benchmark-usi`の実対局も
`play_direct_game`の初期SFENと実際の手順prefixを履歴対応教師へ渡し、reportに同じ
`history_mode`を保存します。`history_mode`が無い旧生成物は`board_only`と扱い、
後から履歴付きと再ラベルしません。履歴依存の教師出力が必要な場合は、履歴付きprovenanceで
再解析または再対局します。

- 通常解析が許可されるか
- 解析出力からの学習・蒸留が許可されるか
- 生成checkpointや教師データを再配布できるか

GPLv3 §2とGNU FAQに従い、別の出力制限が見つからないGPLエンジンの通常USI出力
（指し手、数値評価、nodes、PV/MultiPV）はoutput-only蒸留可と判定します。これは元の評価関数や
重みのコピー・変換・再配布を許す判定ではありません。モデル固有規約を必ず優先します。
現行9件の版固定判定は
[`MODEL_RIGHTS.md`](MODEL_RIGHTS.md)を参照してください。

水匠11Plusと奏乗TSEC7の正確なlocal profileは、`--local-only-user-authorized`とcreate-onlyな
`--local-only-root`を明示したrunだけで起動できます。現行9件以外のIDは登録済みと推測せず、
unknown profileとしてfail closedします。台帳・文書・releaseへ、有料配布
ページ、元評価関数、private path、private artifact hashをコピーしません。ローカル学習の進行と
公開可能なpublic championはlineageで完全に分離します。

## 詰将棋

`tsume-mine` は標準詰将棋（攻方は毎手王手、防方は最善防御）の全応手を調べ、詰む初手が一意な
局面だけを出力します。数千手級はこの短手数solverを深くするだけでは不可能なので、別段階として
逆算生成、検証済み手順部品、df-pn証明木、独立solver再検証を使います。協力詰めは通常詰将棋と
別ルールなので、将来もデータとsolverを分離します。

## コマンド

```bash
uv sync --locked --all-groups

# 環境・モデルshape
uv run simajilord-shogi doctor --profile competition_v1

# neural MCTSのnode/time/NPS、1億node所要時間、peak memory、tree上限
uv run simajilord-shogi bench-nps \
  --profile competition_v1 --parallel-roots 32 --simulations 100

# checkpoint作成
uv run simajilord-shogi init artifacts/initial --profile competition_v2

# 複数局をGPU batchで終局まで自己対局
uv run simajilord-shogi selfplay \
  artifacts/initial artifacts/actor.jsonl --games 32 --mode batched --simulations 800

# 同じ局面を深く再解析
uv run simajilord-shogi reanalyse \
  artifacts/initial artifacts/actor.jsonl artifacts/deep.jsonl \
  --actor-simulations 800 --teacher-simulations 6400 --fraction 0.5

# 深教師＋過去anchorから学習
uv run simajilord-shogi train \
  artifacts/initial artifacts/deep.jsonl artifacts/candidate \
  --anchor-replay artifacts/champion.jsonl --steps 1000 --batch-size 32 \
  --human-play-state-root artifacts/runtime/shogihome

# canonical v2の数学・履歴・独立headはsynthetic fixtureで検証する。
# 実データのtrain CLIはscore-matrix builder receipt実装まで意図的に拒否する。
uv run pytest -q \
  tests/test_distillation_targets_v2.py \
  tests/test_trainer_canonical_v2.py \
  tests/test_model_contract_v2.py

# value labelだけをC=600 / C=756.086496 / held-out通過teacher-fitで比較
# p=sigmoid(cp/C)なので、Meteoの符号付きtargetはtanh(cp/(2C))
uv run simajilord-shogi prepare-value-scale-ablation \
  artifacts/train-teacher.jsonl artifacts/validation-teacher.jsonl \
  artifacts/value-scale-ablation \
  --parent-checkpoint artifacts/initial --training-seed 0 \
  --minimum-fit-games 30 --minimum-validation-games 10

# 公開checkpointへ使える教師 / 正規承認済みlocal-only教師 / 未承認を分けて表示
uv run simajilord-shogi model-rights --public-distillable-only
uv run simajilord-shogi model-rights --local-distillable-only
uv run simajilord-shogi model-rights --not-authorized-only

# 権利確認済み技巧のMultiPVを同一局面へ付与（binary/parameterはrepo外）
uv run simajilord-shogi reanalyse-usi \
  artifacts/actor.jsonl artifacts/gikou-tactics.jsonl \
  --engine /path/to/gikou --engine-cwd /path/to/gikou-data \
  --rights-profile gikou2-v2.0.2 --selection tactical \
  --teacher-tag gikou-tactics --nodes 1000000 --multipv 32 \
  --teacher-ponanza-coefficient 600 \
  --artifact /path/to/params.bin

# YaneuraOu PSVをRAMへ全展開せず学習（教師源名を必須保存）
uv run simajilord-shogi train-psv \
  artifacts/initial teacher.psv artifacts/psv-candidate \
  --source-name reviewed-public-dataset --score-ponanza-coefficient 600 \
  --steps 1000 --batch-size 64

# MIT確認済みHao/tanuki系から、旧labelを捨てて再解析待ちの局面だけRange取得
uv run simajilord-shogi fetch-public-psv-seeds \
  nodchip-shogi-hao-depth9 artifacts/public-seeds/hao-v1 \
  --file-count 8 --records-per-file 4096 --seed meteo-bootstrap-v1

`fetch-public-psv-seeds`の`positions.jsonl`はoptimizer入力ではありません。固定revisionの
PSVから局面だけを抽出し、元のdepth-9 Move16・score・勝敗は`discarded_legacy_annotations`へ
隔離します。CPU側で現在のNAGISA・水匠11Plus・奏乗による候補生成と単一scorer再探索を行い、
新しいscore-matrix receiptが完成した後にだけGPU学習へ渡します。ライセンス未記載の
DL水匠unique版やAobaZero外部棋譜は、このコマンドの許可リストに入りません。

`--teacher-ponanza-coefficient` / `--score-ponanza-coefficient` の既定はPonanza
勝率係数 `C=600` で、内部の符号付きvalue変換は `tanh(cp/(2C))`、すなわち分母
`2C=1200` です。旧 `--teacher-value-scale` / `--score-scale` を明示した場合だけ、
後方互換のため指定値を従来どおり `tanh(cp/scale)` の分母として扱います。新旧flagは
相互排他で、checkpoint lineage / provenanceには `C` と `2C` を両方保存します。

# actor -> 深再解析 -> 学習 -> 独立openingの色替わりarena -> 昇格/棄却
uv run simajilord-shogi improve \
  artifacts/champion artifacts/improvement --generations 1 \
  --games 32 --actor-simulations 800 --teacher-simulations 6400 \
  --opening-suite openings.json --arena-simulations 1600 \
  --promotion-min-pairs 32 \
  --validation-replay heldout-nagisa.jsonl \
  --validation-replay heldout-suisho11plus.jsonl

`openings.json`は次の厳格なread-only形式です。下記はschemaを示すため各splitを1件に
短縮した例です。`actor`と`arena`にはそれぞれ完全なSFEN文字列を並べ、昇格可能な実行では
`arena`に32件以上の独立局面を用意します。

```json
{
  "schema": "meteo-opening-suite-v1",
  "splits": {
    "actor": ["lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"],
    "arena": ["4k4/9/3B5/9/9/9/9/9/4K4 b G 1"]
  }
}
```

局面同一性はSFENの手数fieldを除く盤面・手番・持駒で正規化します。同一split内だけでなく
splitをまたぐ重複も入力時に拒否します。入力fileは`improve`が作成・更新せず、初回manifestに
source SHA-256、正規化後SHA-256、各局面key/hashを固定します。generation途中で内容を変えると
resumeを拒否するため、opening suiteはcreate-onlyの実験入力として扱ってください。

各`--validation-replay`は学習replayと正規化局面が1件でも重なればfail closedします。候補の
policy cross-entropyまたはvalue MSEが直前championより悪化した場合も昇格しません。許容差は
`--max-validation-policy-cross-entropy-increase`と
`--max-validation-value-mse-increase`で明示でき、既定はどちらも0です。独立検証を指定しない
debug実行は対局できますが、promotionは無効です。

arenaは各openingについてcandidate先手の1局、candidate後手の1局をこの順で実行し、この2局を
1 clusterとして片側95% deterministic cluster-bootstrap下限を計算します。独立cluster数が
`--promotion-min-pairs`（既定32）未満、未完局、学習replayとの正規化局面重複、opening反復の
いずれかがあれば昇格しません。cluster結果、CI method/seed/iterations、block理由はmanifestに
保存されます。旧`--arena-games`、`--promotion-min-games`、`--initial-sfen`は後方互換の
smoke/debug用で、production昇格には使えません。generation manifestはschema 3です。旧schema 2は
candidate学習前（`initialized`、`actor_saved`、`reanalysed`）に限り、同じ既存設定を確認してから
独立検証設定を追加して移行できます。それ以降の旧manifestやschema 1はresumeせず拒否します。

# 権利スコープ確認済みのUSIエンジンと直接対局
uv run simajilord-shogi benchmark-usi \
  artifacts/champion artifacts/arena/suisho5 \
  --engine /path/to/YaneuraOu --engine-cwd /path/to/runtime \
  --rights-profile suisho5 --nodes 100000 \
  --opening-suite openings.json --opening-split arena \
  --promotion-min-pairs 32 --bootstrap-iterations 20000 \
  --artifact /path/to/runtime/eval/nn.bin \
  --option Threads=1 --option Hash=1024 --option BookFile=no_book --option FV_SCALE=24 \
  --human-play-state-root artifacts/runtime/shogihome

`benchmark-usi`のproduction経路も、選択splitの各openingでMeteo先手・後手を1局ずつ
実行し、2局を1つの独立clusterとして集計します。read-only suiteはsplitをまたいだ正規化
SFEN重複も拒否し、productionでは最低32組を要求します。reportの信頼区間とEloは個別局では
なくopening-pair clusterに基づき、未完局・反復・組数不足があればEloを出しません。

出力directoryは実走前に存在しないことを確認し、同じ親directoryの一時bundleへ
`games.jsonl`と`report.json`を完成させてから一度だけrenameします。途中失敗時は一時bundleを
削除し、既存出力を上書きしません。reportにはcheckpoint内全fileのSHA-256とlineage、実際の
source-tree/Git identity、engine・別配布artifact・rights row・全option、replay SHA-256、
opening suite/split/hash/key、CI seed/iterations/method、全終局理由と未完数を保存します。

正規入手した`local_authorized_only`教師との対局も、公開可能教師と同じCLIで実行できます。
その場合は`--local-only-user-authorized`と`--local-only-root`を必須とし、出力bundleを
明示したprivate rootの子に閉じ込めます。水匠11Plusの正確なlocal profileでは、
`FV_SCALE=40`、`USI_OwnBook=false`、`BookFile=no_book`、`PvInterval=0`、`EvalDir`、
`Threads`、`USI_Hash`を明示し、起動後にYaneuraOuの`getoption`で全実値を再確認します。
この版のSFNN exporterとYaneuraOuは固定hash規約が異なるため、正しい組み合わせでも起動時に
既知のNNUE hash警告を出します。一般のhash不一致は起動失敗とし、このexact local profileだけは
global 1件・layer 10件の診断文、順序、件数がreview済み期待値と完全一致するときだけ受理します。
`EvalDir/nn.bin`は明示的な`--artifact`がなくてもprivate reportへ自動でhash記録されます。
Finny Tablesはやねうら王側のFT accumulator cacheであり、MeteoのResNet重みやcheckpoint系譜は
変更しません。採用時は同一source・同一評価関数のFinny無効版と固定nodeで評価値、PV、最善手が
一致することを確認してから、外部教師探索の高速化として扱います。
ローカル対局・蒸留・学習の承認と、原評価関数または派生checkpointの公開許可は別のgateです。
private bundle、第三者artifact、入手先URLはrepositoryやreleaseへ含めません。

単一局面の疎通確認だけが必要な場合は`--legacy-single-opening-debug`を明示します。この経路は
必ず先後1組だけで、summaryにdebug blockerを残し、昇格判定にもEloにも使用できません。

# 対局・USI
uv run simajilord-shogi play artifacts/candidate --human black
uv run simajilord-shogi usi artifacts/candidate

# 悪手、相手profile、詰将棋
uv run simajilord-shogi blunders artifacts/deep.jsonl
uv run simajilord-shogi opponent-learn \
  artifacts/deep.jsonl artifacts/opponents/suisho.json --name suisho --color white
uv run simajilord-shogi tsume-mine \
  artifacts/deep.jsonl artifacts/tsume.jsonl --plies 5

# 完全受入
uv run simajilord-shogi verify /tmp/simajilord-shogi-verify \
  --workers 4 --profile competition_v1
```

## 大会級への未完了項目

実装済みの学習基盤だけで大会優勝は保証できません。次の競技機能は今後の明示的な昇格条件です。

- ライセンス確認済みの強い初期教師・数億局面以上の取り込み
- 1手詰めに加えたdf-pn、長手数詰み、必至探索
- compact native tree、置換表、持時間配分、pondering、前局面からのtree reuse、resign calibration
- opening book生成と定跡外しへの深探索
- 現行9モデル・過去championとの大規模固定条件arenaとSPRT
- USIの非同期`stop`、`ponderhit`、CSA bridge、Floodgate再接続、大会運用監視
- strategy-conditioned policyと、標準将棋から隔離した二歩ありvariant backend
- SFEN、MultiPV、regret、詰み証明をtoolとして渡す将棋LLMと、検証可能rewardによる強化学習

## ライセンス境界

本リポジトリはApache-2.0です。MITの`rsshogi`を利用します。GPL等の外部エンジンはUSIの
別process境界に置きます。モデル、教師データ、クラウド解析出力のライセンスはコードとは別に
確認し、公開可能性をcheckpoint metadataへ記録します。
