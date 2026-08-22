# Meteo 学習戦略

更新: 2026-08-13

この文書は、Meteoの主学習経路と比較実験を区別する運用契約です。実装が存在すること、receiptが
揃うこと、optimizerが実際に動くこと、棋力が改善することを別々に記録します。機械可読な同一契約は
`simajilord-shogi learning-strategy`で出力します。

## 現在の本学習経路

現在のproductionは、NAGISA互換構造のvalue-only NNUE/SFNNです。

```text
Soujou datasets_1のqsearch済みMove16=0 PSV
  -> revision・size・LFS SHA-256固定
  -> 1,000,000局面held-outを学習前に分離
  -> DL水匠scalar scoreのsigmoid(score / 600)分布をvalue-only WRMで学習
  -> network outputはscore / 508へ正規化し、量子化後はYaneuraOu FV_SCALE=16で元scoreへ戻す
  -> exact PSV decodeはCPU、optimizerはApple Silicon GPU / MLX
  -> HalfKA_hm2(Friend) 73305 -> 1024x2 -> 15 -> 64
  -> progress8kpabs LayerStack 9
  -> やねうら王互換nn.binへ変換
```

学生NNUE重みはランダム初期化です。NAGISA archiveから使うのは固定bucket routerの
`progress.bin`だけで、NAGISAのNNUE重みはコピーしません。policy headとpolicy lossは存在せず、
Move16=0から架空のpolicyを生成しません。公開corpusの旧scalarはこの経路では直接教師に使います。
±32000帯はsigmoid targetが安全に0/1へ飽和し、policy headを汚す経路もないため除外しません。
教師targetは従来どおり`sigmoid(score / 600)`です。WRMの`nnue2score=508`は、量子化gain
`127 * 64 = 8128`をYaneuraOuの`FV_SCALE=16`で割った値です。したがって学習時の最適出力
`score / 508`を量子化すると、推論時には`(score / 508) * 8128 / 16 = score`となり、教師scoreの
尺度へ戻ります。量子化前後の分布は同一held-out局面で平均・標準偏差・分位点・相関・回帰傾き・
符号不一致率・sigmoid確率誤差を監査し、許容範囲外ならcheckpointを確定しません。

目標は1000億の`cumulative_presentations`です。`datasets_1`の一意source 49,594,855,063局面を
2周し、残りを3周目から読むので、1000億の一意局面を持つという意味ではありません。source unique、
学習から除外したheld-out、累計提示数を別々に保存します。現在shardと次のprefetch shardだけを
stream cacheに置き、完了後に削除し、checkpoint/exportは最新と直前だけ保持します。

本番optimizerはクラウドGPUを使わず、Apple M4 ProのMLXで実行します。CPUは固定Tatara実装を使って
PSVをbit-exactにdecodeし、HalfKA_hm2特徴とprogress bucketを生成します。MLXは同じNAGISA互換構造の
順伝播・WRM loss・Ranger更新だけを担当します。合成probeの速度は実データstream、初回compile、
checkpoint、量子化exportを含まないため、本番ETAは実測throughputから更新します。

このbootstrap完走後のvalue-only強化学習、Meteo自己対局、3教師との完全対局、共通再解析、
replay、構造比較、promotion、解析/対人配備の正本は
[`POST_100B_ROADMAP.md`](POST_100B_ROADMAP.md)です。

## 旧Policy+Value比較経路

NAGISA型bootstrap後の自己対局・policy研究に残す比較候補は次です。

```text
NAGISA V3.1 + 水匠11Plus + 奏乗TSEC7 が候補を提案
  -> 1教師をscorerに固定
  -> 全候補を同一要求node数で個別searchmoves探索
  -> 上位候補後の応手候補も3教師から収集
  -> 同じscorerで全応手を再探索してnegamax backup
  -> exact Q分布、区間優越、またはUNRESOLVEDへ分類
  -> 学習可能部分だけをMeteoへ蒸留
```

暫定scorerは奏乗TSEC7です。ただしNAGISA V3.1と水匠11Plusも同列の主教師候補です。次の六armを
同じ局面、初期seed、batch順、optimizer、step数、モデル構造、探索条件で作ります。

| arm | 候補提案 | 最終scorer |
| --- | --- | --- |
| A-N | NAGISAのみ | NAGISA |
| A-W | 水匠のみ | 水匠 |
| A-S | 奏乗のみ | 奏乗 |
| B-N | 3教師 | NAGISA |
| B-W | 3教師 | 水匠 |
| B-S | 3教師 | 奏乗 |

全会一致のみのRound Cと、既存の三教師worst-regret合成であるRound Dも対照群に残します。最初から
教師routerを作らず、専門性が戦型・進行度・玉危険度・入玉・詰み距離ごとに実測された場合だけRound Eを
試します。

主教師は短時間総当たり順位だけで決めません。深く裁定した最善手に対するregret、10倍budgetでの
top-1安定率、mate見落とし、評価符号反転、上位5%/1%の重大事故、実勝敗への校正、局面群別の
最大regret、fixed-node/time棋力を使います。

三教師は「独立な3票」ではありません。NAGISAは奏乗WCSC36教師局面集の利用を明記し、
奏乗はDL水匠で再評価したnodchip系局面を主祖先に持ちます。水匠11もDL水匠を含む三モデルの
アンサンブル教師です。そのため、局面数や信頼度を名前だけで加算せず、系譜family単位で
重複を排除します。機械可読台帳は`simajilord-shogi teacher-lineage`、根拠と再利用判定は
[`TEACHER_LINEAGE.md`](TEACHER_LINEAGE.md)、全関連corpusの固定revision・bytes・形式probe・
利用順は[`DATASET_AUDIT.md`](DATASET_AUDIT.md)です。

水匠11の公式文書はRyfamate最新・DL水匠・AobaZeroの3モデルで評価値データを合成したことまで
開示していますが、統合式と重みは非公開です。Round Dでは水匠11の再現とは呼ばず、INUGAMIが
公開した「勝率を共通評価値へ変換して相加平均」、「校正後勝率の平均」、現行のinterval robustを
同条件比較します。A/Bに実測で勝つまでは、いずれもNNUE bootstrap後の強化学習教師へ採用しません。

## bound、policy、mate

1教師の値だけを使っても、`lowerbound`と`upperbound`を点推定へ変えてはいけません。

```text
exact      q -> [q, q]
lowerbound q -> [q, 1]
upperbound q -> [-1, q]
```

`LB(best) > max(UB(other)) + margin`なら区間優越で一意最善手です。区間が重なる場合、policy lossは
0にして追加探索へ戻します。局面valueが`[L,U]`にだけ絞れる場合は、予測が区間内なら0となる
interval lossを使えます。全候補exactの場合だけ、等しい要求node budgetのQから温度校正済みsoftmaxを
作ります。alpha-beta探索の実node配分はmove ordering、LMR、枝刈り、TT、fail-high/lowの影響を受けるため
policy確率にしません。

教師が`mate`を報告しただけでは通常policyへ混ぜず、探索増量と内部DFPNの要求にします。内部solverが
証明した複数の即詰み手はset-valued policy、証明済み勝ちはvalue `+1`として別扱いします。

## 旧Policy+Value経路の段階と昇格

- 20〜100局面: 2x32 `smoke`を意図的に過学習し、符号、履歴、mate集合、save/load、optimizer再開、
  checkpoint fault injectionを検証する。棋力評価も昇格もしない。
- 約1万局面: split漏洩、重複、Q視点、loss mask、bound、長手数終局を監査する。
- 数十万局面: A/B/C/Dと3 scorer候補の方向性を複数seedで比較する。
- 数百万局面以上: 20x256 `competition_v1`をbootstrapする。
- 以後: 教師対局、Meteo自己対局、失敗局面、定跡離脱、特殊局面を反復再解析する。

Round Sの実データ抽出は、完了済みcommittee benchmarkに対して次を実行します。

```bash
simajilord-shogi build-unanimous-smoke BENCHMARK_ROOT OUTPUT \
  --scorer-id soujou-tsec7-paid \
  --train-count 20 --calibration-count 10 --heldout-count 10
```

これはcommitteeの実指し手を教師化しません。3教師の深いroot scoreと応手backupがすべてexact、
一意最善手が3者一致、root budget間でも不変、応手後も同じ手である局面だけを採用します。policyは
奏乗の応手backup済みQのsoftmaxであり、教師間平均ではありません。単一book root、履歴transcript欠落、
全合法手未網羅をreceiptへ明記するため、calibration/heldoutというファイル名でも正式なモデル選択・
Elo昇格には使えません。

昇格には同一局面・同一seed比較、held-out regret、fixed-node、fixed-time、worst-group、複数seedを
要求します。validation lossだけで主経路を決めません。

## 1,000億局面目標

長期目標は`cumulative_presentations = 100,000,000,000`です。これは1,000億個の巨大JSONを常駐保存する
意味ではありません。次のカウンタを分けます。

- `unique_source_positions`: 重複除去前の局面源
- `unique_value_source_positions`: immutable PSVで確認した一意source局面
- `unique_search_labelled_positions`: 将来、教師score matrixとreceiptを持つ一意局面
- `unique_retained_training_positions`: 圧縮replay bufferに残す一意局面
- `cumulative_positions_seen`: optimizerへ実際に提示した局面数（epoch反復を含む）
- `cumulative_high_confidence_positions_seen`: exact、区間優越、mate/終局証明など品質gate通過分

原本とraw score matrixはcreate-only shard、学習入力は圧縮・重複除去・容量制限付きbuffer、古い広域
データと新しいhard caseは層別samplingで混ぜます。数値目標のためにUNRESOLVEDを偽ラベル化したり、
同じ20局面の反復を高品質1,000億として報告したりしません。ローカル空き容量で保持できないunique
corpusは、外部ストレージまたは再生成可能なmanifest/hashがない限り生成開始しません。

## 現NNUE本学習の開始前receipt

production optimizerを開始する前に、少なくとも次を揃えます。

1. `corpus_index.json`: repository revision、全shardのfilename、bytes、records、LFS SHA-256を固定する。
2. `heldout/receipt.json`: 最終shard tail 1,000,000局面を先に固定し、全training passから除外する。
3. `progress-receipt.json`: NAGISA archive hash、抽出member、bytes、SHA-256、NNUE weight非コピーを保存する。
4. `per-shard receipt`: Move16=0、board decode、score/result、source SHA、範囲、loss、throughputを保存する。
5. `checkpoint_complete_marker`: weights、optimizer、RNG、trace、metadataをtempへ書き、flush/fsync、hash、
   再読込後にcomplete markerを最後に作り、同一filesystem内rename後に旧世代を整理する。

旧Policy+Value経路へ進む場合は、これらとは別に`split_receipt`、`score_matrix_receipt`、
`anchor_calibration_receipt`が必要です。NNUEのMove16=0 scalarをpolicy targetとして流用してはいけません。

新ペタショック定跡は開始局面分布と定跡離脱局面の採掘に使い、book moveを正解policyにしません。
最終的には教師を正解生成器から反例生成器へ後退させ、Meteo探索・終局結果・mate証明・自己対局の
新情報で教師依存を減らします。
