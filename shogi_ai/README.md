# めてお - Simajilord Shogi AI

**めてお**（英字表記: **Meteo**）は、Simajilord-AI リポジトリ内のオリジナル
将棋AIです。`shogi_ai` は Discord から独立した将棋エンジン・
自己対局・学習コンポーネントです。CLI と USI で単独利用でき、将来の Discord Bot は
この公開境界を呼び出します。USI と checkpoint metadata にも公開名「めてお」を
保存します。

たややん氏／水匠、やねうら王、WCSC公式資料の調査とMeteoへの反映方針は
[`RESEARCH.md`](RESEARCH.md) に分離して記録します。
外部モデルの利用可否は[`MODEL_RIGHTS.md`](MODEL_RIGHTS.md)、実行済みの受入試験と
4種の実エンジン対局・蒸留は[`VALIDATION.md`](VALIDATION.md)、Floodgate参加前の条件は
[`FLOODGATE.md`](FLOODGATE.md)に記録します。

## 現在できること

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

canonical dual-teacher学習は現在、型付きsidecar・集合方策loss・教師別value loss・
exact-resume境界までを実装した第一段階です。training-eligible sidecarを生成するbuilder、
全候補familyの再採点、内部詰み証明、depth/reply安定性、validation重複除去は未実装です。
そのためraw NAGISA／水匠出力の単純平均をcanonical targetとして扱ってはならず、
sidecarが自称するだけの`proven`詰みはloaderが拒否します。これらが揃うまで、同経路の
checkpointを昇格・公開・教師越えの根拠にしません。

## モデル

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

- 同一構造: 無料公開dlshogi ResNet重みをv1へ変換し、深教師データで継続学習
- 構造拡張: v1の対応するconvolution層をv2へ移植し、追加Transformer層は段階的に解凍
- 非互換: AobaNNUE・公開水匠・探索エンジンは重みをコピーせず、USI深解析から蒸留

どの経路も、元重みと教師出力の学習利用・改変版再配布条件が確認できた無料資源だけを
対象にします。

64 GiB Apple Siliconでの2026-08-09実測では、ランダム重みの20x256 neural MCTSは単局
100 simulationsで約105 NPS、32局面を各100 simulations読むbatchで合計約781 NPSでした。
40x512は単局30 simulationsで約38 NPSです。速度測定であり棋力比較ではありません。この差のため、
最初に大量教師を吸収する主力は20x256とし、40x512は同じheld-out arenaで上回ってから昇格します。

単局1億nodeは現状約266時間です。自動メモリ回収により総node数だけを伸ばすことはできますが、
直近の自動上限は約205万Python tree nodeであり、1億nodeの木を完全保持するものではありません。
実戦的な1億nodeには、compact native tree、置換表、tree reuse、非同期dynamic batchingが必要です。
計測方法とFloodgate到達条件は[`FLOODGATE.md`](FLOODGATE.md)に固定しました。

## 強い学習ループ

浅い自己対局は局面を広く作るactorであり、既定ではその浅いpolicyを直接教師にしません。

```text
batched actor games
  -> 同一SFENを6,400+ simulationsで再解析
  -> 不一致・低prior浮上・大regret局面をさらに深く読む
  -> actor/teacher探索比に応じてdeep policy/valueを段階導入
  -> 固定probe悪化・非有限loss・過大勾配を検知
  -> candidate vs champion/history/水匠系のarena
  -> 統計的に勝ったcandidateだけchampionへ昇格
```

この一連は`improve`で1世代以上を連続実行できます。各世代は
`generation-NNNNNN/manifest.json`へ段階、checkpoint SHA-256、学習loss、arena結果を保存し、
`state.json`のchampionポインタは昇格時だけ更新します。色替わりarenaに未完局が1局でも
あれば昇格せず、既定では100局以上かつ勝点率のWilson 95%下限が0.5を超える必要があります。

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

## 権利確認済みの無料外部教師

主経路は無料で再現・公開可能な資源だけを使います。候補には無料公開版のやねうら王・水匠、
AobaZero/AobaNNUE、公開WCSC・電竜戦棋譜、公開水匠教師データ、自己対局と深再解析を含みます。
無料であっても学習データとモデルのライセンスは個別に確認します。

`ExternalUsiTeacher` はNAGISA V3.1、AobaNNUE v1.1、技巧2、水匠5などを別processとして
接続します。4種はこのMacで実際に終局対局とMultiPV蒸留まで確認済みです。
実行ファイルや重みを本リポジトリへコピーしない境界です。教師ごとに次の許諾を明示し、
学習利用が未確認ならfail closedします。

`reanalyse-usi`は局面SFEN単体ではなく、`GameRecord.initial_sfen`と
`moves[:PositionSample.ply]`を合法に再生し、対象局面と完全一致した履歴を
`position startpos moves ...`または`position sfen ... moves ...`で教師へ渡します。
sidecarの`history_mode=game_prefix`がこの境界を示します。`benchmark-usi`の実対局も
`play_direct_game`の初期SFENと実際の手順prefixを履歴対応教師へ渡し、reportに同じ
`history_mode`を保存します。`history_mode`が無い旧生成物は`board_only`と扱い、
後から履歴付きと再ラベルしません。AobaZero系など履歴依存の教師出力が必要な
場合は、履歴付きprovenanceで再解析または再対局します。

- 通常解析が許可されるか
- 解析出力からの学習・蒸留が許可されるか
- 生成checkpointや教師データを再配布できるか

GPLv3 §2とGNU FAQに従い、別の出力制限が見つからないGPLエンジンの通常USI出力
（指し手、数値評価、nodes、PV/MultiPV）はoutput-only蒸留可と判定します。これは元の評価関数や
重みのコピー・変換・再配布を許す判定ではありません。モデル固有規約を必ず優先し、たとえば
`dlshogi-dr2-exhi`は一般蒸留不可です。全23件の版固定判定は
[`MODEL_RIGHTS.md`](MODEL_RIGHTS.md)を参照してください。

有料モデル・棋神アナリティクス等は使用しません。`ExternalTeacherPolicy` は
`requires_payment=True` の教師を、学習出力の許諾があっても起動前に拒否します。有料版は
強さの外部benchmarkとして記録するだけで、自動取得や蒸留の対象にしません。
目標は公開資源と自前学習でそれらを超えることです。

## 詰将棋

`tsume-mine` は標準詰将棋（攻方は毎手王手、防方は最善防御）の全応手を調べ、詰む初手が一意な
局面だけを出力します。数千手級はこの短手数solverを深くするだけでは不可能なので、別段階として
逆算生成、検証済み手順部品、df-pn証明木、独立solver再検証を使います。協力詰めは通常詰将棋と
別ルールなので、将来もデータとsolverを分離します。

## コマンド

```bash
uv sync --project shogi_ai --all-groups

# 環境・モデルshape
uv run --project shogi_ai simajilord-shogi doctor --profile competition_v1

# neural MCTSのnode/time/NPS、1億node所要時間、peak memory、tree上限
uv run --project shogi_ai simajilord-shogi bench-nps \
  --profile competition_v1 --parallel-roots 32 --simulations 100

# checkpoint作成
uv run --project shogi_ai simajilord-shogi init artifacts/initial --profile competition_v2

# 複数局をGPU batchで終局まで自己対局
uv run --project shogi_ai simajilord-shogi selfplay \
  artifacts/initial artifacts/actor.jsonl --games 32 --mode batched --simulations 800

# 同じ局面を深く再解析
uv run --project shogi_ai simajilord-shogi reanalyse \
  artifacts/initial artifacts/actor.jsonl artifacts/deep.jsonl \
  --actor-simulations 800 --teacher-simulations 6400 --fraction 0.5

# 深教師＋過去anchorから学習
uv run --project shogi_ai simajilord-shogi train \
  artifacts/initial artifacts/deep.jsonl artifacts/candidate \
  --anchor-replay artifacts/champion.jsonl --steps 1000 --batch-size 32 \
  --human-play-state-root artifacts/runtime/shogihome

# value labelだけをC=600 / C=756.086496 / held-out通過teacher-fitで比較
# p=sigmoid(cp/C)なので、Meteoの符号付きtargetはtanh(cp/(2C))
uv run --project shogi_ai simajilord-shogi prepare-value-scale-ablation \
  artifacts/train-teacher.jsonl artifacts/validation-teacher.jsonl \
  artifacts/value-scale-ablation \
  --parent-checkpoint artifacts/initial --training-seed 0 \
  --minimum-fit-games 30 --minimum-validation-games 10

# 調査済みモデル権利台帳
uv run --project shogi_ai simajilord-shogi model-rights --distillable-only

# 権利確認済み技巧のMultiPVを同一局面へ付与（binary/parameterはrepo外）
uv run --project shogi_ai simajilord-shogi reanalyse-usi \
  artifacts/actor.jsonl artifacts/gikou-tactics.jsonl \
  --engine /path/to/gikou --engine-cwd /path/to/gikou-data \
  --rights-profile gikou2-v2.0.2 --selection tactical \
  --teacher-tag gikou-tactics --nodes 1000000 --multipv 32 \
  --teacher-ponanza-coefficient 600 \
  --artifact /path/to/params.bin

# YaneuraOu PSVをRAMへ全展開せず学習（教師源名を必須保存）
uv run --project shogi_ai simajilord-shogi train-psv \
  artifacts/initial teacher.psv artifacts/psv-candidate \
  --source-name reviewed-public-dataset --score-ponanza-coefficient 600 \
  --steps 1000 --batch-size 64

`--teacher-ponanza-coefficient` / `--score-ponanza-coefficient` の既定はPonanza
勝率係数 `C=600` で、内部の符号付きvalue変換は `tanh(cp/(2C))`、すなわち分母
`2C=1200` です。旧 `--teacher-value-scale` / `--score-scale` を明示した場合だけ、
後方互換のため指定値を従来どおり `tanh(cp/scale)` の分母として扱います。新旧flagは
相互排他で、checkpoint lineage / provenanceには `C` と `2C` を両方保存します。

# actor -> 深再解析 -> 学習 -> 独立openingの色替わりarena -> 昇格/棄却
uv run --project shogi_ai simajilord-shogi improve \
  artifacts/champion artifacts/improvement --generations 1 \
  --games 32 --actor-simulations 800 --teacher-simulations 6400 \
  --opening-suite openings.json --arena-simulations 1600 \
  --promotion-min-pairs 32

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

arenaは各openingについてcandidate先手の1局、candidate後手の1局をこの順で実行し、この2局を
1 clusterとして片側95% deterministic cluster-bootstrap下限を計算します。独立cluster数が
`--promotion-min-pairs`（既定32）未満、未完局、学習replayとの正規化局面重複、opening反復の
いずれかがあれば昇格しません。cluster結果、CI method/seed/iterations、block理由はmanifestに
保存されます。旧`--arena-games`、`--promotion-min-games`、`--initial-sfen`は後方互換の
smoke/debug用で、production昇格には使えません。旧schema 1の未完generation manifestは
新しいpaired判定へresumeせず拒否します。その場合は既存checkpointをbootstrap championにして
新しいworkdirでschema 2 generationを開始してください。

# ライセンス確認済みの無料USIエンジンと直接対局
uv run --project shogi_ai simajilord-shogi benchmark-usi \
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

単一局面の疎通確認だけが必要な場合は`--legacy-single-opening-debug`を明示します。この経路は
必ず先後1組だけで、summaryにdebug blockerを残し、昇格判定にもEloにも使用できません。

# 対局・USI
uv run --project shogi_ai simajilord-shogi play artifacts/candidate --human black
uv run --project shogi_ai simajilord-shogi usi artifacts/candidate

# 悪手、相手profile、詰将棋
uv run --project shogi_ai simajilord-shogi blunders artifacts/deep.jsonl
uv run --project shogi_ai simajilord-shogi opponent-learn \
  artifacts/deep.jsonl artifacts/opponents/suisho.json --name suisho --color white
uv run --project shogi_ai simajilord-shogi tsume-mine \
  artifacts/deep.jsonl artifacts/tsume.jsonl --plies 5

# 完全受入
uv run --project shogi_ai simajilord-shogi verify /tmp/simajilord-shogi-verify \
  --workers 4 --profile competition_v1
```

## 大会級への未完了項目

実装済みの学習基盤だけで大会優勝は保証できません。次の競技機能は今後の明示的な昇格条件です。

- ライセンス確認済みの強い初期教師・数億局面以上の取り込み
- 1手詰めに加えたdf-pn、長手数詰み、必至探索
- compact native tree、置換表、持時間配分、pondering、前局面からのtree reuse、resign calibration
- opening book生成と定跡外しへの深探索
- 水匠11・氷彗級、最新版dlshogi、過去championとの大規模固定条件arenaとSPRT
- USIの非同期`stop`、`ponderhit`、CSA bridge、Floodgate再接続、大会運用監視
- strategy-conditioned policyと、標準将棋から隔離した二歩ありvariant backend
- SFEN、MultiPV、regret、詰み証明をtoolとして渡す将棋LLMと、検証可能rewardによる強化学習

## ライセンス境界

本subprojectはApache-2.0です。MITの`rsshogi`を利用します。GPL等の外部エンジンはUSIの
別process境界に置きます。モデル、教師データ、クラウド解析出力のライセンスはコードとは別に
確認し、公開可能性をcheckpoint metadataへ記録します。
