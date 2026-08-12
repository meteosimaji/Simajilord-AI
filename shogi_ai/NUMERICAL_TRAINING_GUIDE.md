# MLX数値学習・高速化の汎用ガイド

更新: 2026-08-12

この文書はMeteoのNNUE学習で得た知見を、将来のCNN、Transformer、LLMにも再利用できる形で
残すものです。測定値は64 GiB Apple M4 Pro、MLX 0.32.0でのローカル実測であり、別のモデルや
MLX版へそのまま一般化しません。コードとして再利用する最小部品は
`simajilord_shogi/mlx_training_primitives.py`に分離しています。

## 1. 学習対象はfloatモデルではなく配布時の計算グラフ

量子化前の旧Meteoは、float32のWRM lossを最小化してから最後だけNNUE整数形式へ変換していました。
約5億局面後の8,192局面監査では、float masterと実配布networkの勝率差は次でした。

| 指標 | 旧float学習→整数export |
| --- | ---: |
| 平均絶対勝率差 | 0.0113799 |
| p99絶対勝率差 | 0.0663377 |
| 最大絶対勝率差 | 0.1320950 |
| float held-out loss | 0.0295194 |
| 配布整数 held-out loss | 0.0298374 |

重みの飽和は0件でした。誤差の主因はclipではなく、特徴変換の`1/127`、dense weightの`1/64`、
biasの`1/8128`への丸めと、L1 factorizerをbucket weightへ畳み込んでから一度だけ丸めるexport契約です。
したがって「量子化後に近いことを祈る」のではなく、配布時と同じ前向き値を
straight-through estimator（STE）で最初から目的関数へ入れます。

MeteoのQAT graphは次を明示的に再現します。

- feature transformer: `round(weight * 127)`をi16範囲へ飽和
- Pairwise CReLU: `clip(a,0,127) * clip(b,0,127) >> 7`
- dense weight: `round(weight * 64)`をi8範囲へ飽和
- dense bias: `round(bias * 8128)`
- L1 square branch: `(raw * raw) >> 19`
- CReLU branch: `raw >> 6`
- shared L1 factorizer: bucket差分と加算してから一度だけ量子化
- 最終出力: integer rawを`8128`で割った学習正規化値

合否はfloat masterとの差ではなく、QAT referenceと実際に書き出したnetworkを読むnative evaluatorの
一致で判定します。float masterとの差は診断値として残します。旧checkpointに新監査を適用すると、
QAT referenceとnative exportのp99勝率差は約`1.02e-8`、最大差は稀なfloat accumulation境界で
`0.002049`、held-out loss差は約`1.35e-7`でした。

## 2. ランダム初期化には量子化ウォームアップが必要

初期FT weightの絶対値は約`1/sqrt(73305)`以下で、`1/127`未満です。最初から厳密に丸めると全FT rowが
0になります。MeteoのPairwise CReLUは二つの枝を乗算するため、両方が0だとSTEを入れても相手因子が
0のままで、FTへ届く勾配も0になります。

このため本番契約は二段です。

1. optimizer step 0〜1023（67,108,864 presentations）はFT gather/accumulationをFP16、denseをFP32で学習
2. 完了step 1024から実配布整数graphのSTE-QATへ切替

境界はcheckpoint manifestとログに保存し、再開時は完了optimizer stepだけから決定します。
「何epoch目だったか」やプロセス内フラグには依存しません。CNN/LLMでも、量子化直後のactivationや
gateが全ゼロになる構造では、warm-up、段階的fake quantisation、または非ゼロ初期化を比較すべきです。

## 3. 統合メモリは使用量より再利用率を見る

単にbatchを大きくする、または上限を物理RAM近くへ上げても速くなるとは限りません。今回の最大要因は
MLX allocation cacheでした。65,536局面/batchのQATを、compile後に同じ実PSV batchで12 update測った
結果です。

| MLX cache上限 | QAT局面/秒 | peak MLX memory | 実cache |
| ---: | ---: | ---: | ---: |
| 4 GiB | 53,451 | 20.31 GB | 4.29 GB |
| 24 GiB | 81,693 | 20.31 GB | 22.62 GB |

24 GiB cacheは4 GiB比で約52.8%速くなりました。大きな中間bufferを毎update解放・再確保しない効果です。
これは同一batchを使ったcompute ceilingであり、download、decode、checkpoint、validationを含む本番平均
ではありません。1000億局面をこのceilingだけで割ると約14.17日ですが、本番ETAは必ずログの
end-to-end速度で更新します。

採用した原則は次です。

- `peak active + useful cache + OS/app headroom < recommended working set`を守る
- cacheの実測plateau（今回約22.6 GB）を少し上回る上限にし、それ以上は増やさない
- memory pressure、compression、swap、thermal throttlingを速度と同時に監視する
- batch拡大は学習ダイナミクスも変えるため、同じoptimizer coordinateの比較なしに採用しない
- FT optimizer momentのFP16化は容量を減らしたが速度改善が再現せず、精度リスクもあるため本番不採用

CNN/LLMでも、KV/activation/checkpoint bufferの再利用に必要なcacheと、単なる未使用予約を区別します。
OOM直前まで埋めることは目的ではありません。圧縮・swapが始まれば統合メモリGPUはCPUと一緒に遅くなります。

## 4. I/Oは次のitemをcompute中にprefetchする

PSV native decoderは単体で約456万局面/秒で、GPU updateより十分速いためbatch prefetchで隠せます。
一方、公開corpusは1 shard約20 GBです。shard境界でdownloadを始める逐次実装では、GPUが完全に停止します。

本番runnerは現在のshardを学習中に次の1 shardだけdaemon prefetchします。

- `.part`を使う既存のrange-resume downloaderだけをproducerにする
- 次のschedule itemとfilenameを照合してから消費する
- download失敗は次境界で例外として伝播する
- graceful stopの権威はcheckpoint側に置き、prefetch threadはdaemonにする
- `残download bytes + 12 GiB reserve`の空き容量がない場合はprefetchしない
- 完了shardを消しても、次shardと最新・直前checkpointだけは残る

汎用`DaemonPrefetch[T]`は巨大dataset shard、tokenized LLM corpus、画像tar、validation batchにも使えます。
producer側が中断可能または部分成果物を安全にresumeできることが前提です。

## 5. 保存と再開は数値phaseまでtransactionに含める

checkpointはmodelだけでなくoptimizerの全state、global step、source plan hash、MLX plan hash、次updateの
numeric phaseを持ちます。保存手順は一時directoryへのmodel/optimizer保存、fsync、hash/signature manifest、
directory rename、再load、複数forward modeの一致確認、最後に旧世代削除です。

最低限のfault modelは次です。

- modelだけ保存された
- optimizerだけ保存された
- manifestが途中まで書かれた
- 最新世代が壊れた
- phase境界直前・直後に停止した
- exportはできたがreceiptがない
- source shardはdownload途中だった

「ファイルが存在する」は成功条件ではありません。最新と直前の2世代を残し、完全manifestと再load検証を
通った世代だけをdurable coordinateにします。

## 6. CNN・LLMへ移すときのチェックリスト

1. 実配布graph（int8/int4、group scale、zero point、KV形式、kernel fusion順）を一つの仕様にする
2. merge/fold（BatchNorm、LoRA、factorizer、expert weight）を量子化の前後どちらで行うか固定する
3. float、QAT reference、fast training emulation、native deploymentの4出力を同一sampleで保存する
4. pointwise誤差、p99、task loss、calibration、worst groupを別々に測る
5. 初期量子化で全ゼロになる層・gate・expertがないか調べる
6. warm-up境界をglobal optimizer stepへ固定し、resumeで再現する
7. accelerator ceilingと、download/checkpoint込みend-to-end throughputを分ける
8. cache/batch/precisionは一項目ずつisolated processで比較する
9. held-outをgradient、calibration、threshold調整のどれに使ったかreceiptへ残す
10. loss低下を能力向上と同一視せず、最終taskのarena/evalで昇格させる

Meteo固有の棋力判断にはこの数値監査だけでは足りません。量子化対応モデルが旧floatモデルより強いかは、
同一探索、同一開始局面、先後反転、fixed-node/fixed-time対局で別に確認します。
