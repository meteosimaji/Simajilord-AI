# Meteo Floodgate 到達計画

更新: 2026-08-09

目標は、Meteoを東京大学のコンピュータ将棋対局場
[Floodgate](https://wdoor.c.u-tokyo.ac.jp/shogi/)で継続運用し、対局不能、反則、時間切れ、
既知の探索欠陥による負けをなくした上で、参加相手に対する勝率を最大化することです。

「全勝」は最終目標として追跡しますが、未知の相手、相手側の更新、ネットワーク障害、有限探索が
あるため、事前に保証できる性質ではありません。公開結果で確認できるのは、指定期間の実績が
全勝だったことだけです。したがって、`全勝保証`ではなく、次の三つを別々に記録します。

- 技術的敗因ゼロ: 反則、クラッシュ、時刻超過、接続切れ、誤った投了、誤った入玉宣言が0件
- 棋力: 勝敗、勝点率、Wilson 95%信頼区間、相手別・先後別・戦型別の結果
- 完全勝利期間: 連勝数と、開始・終了時刻を含む実際のFloodgate棋譜

## 現行条件

公式ページで確認した2026-08-09時点の条件です。

- ゲーム名は`floodgate-300-10F`
- 基本持時間300秒、1手ごとに10秒加算するフィッシャールール
- ログイン中のエンジンを毎時0分と30分付近に組み合わせる
- CSAモードは`wdoor.c.u-tokyo.ac.jp:4081`
- 登録不要だが、他と衝突しにくい固有名が推奨される。公開名は`Meteo`
- CSAパスワードは`floodgate-300-10F,trip`形式。平文送信なので、`trip`には他サービスの
  パスワード、API key、個人情報を絶対に使わない
- 約15局からレーティングが計算される
- 512手に達すると引分。Meteo内部の無制限終局対局とは別の大会ルールとして評価する

接続情報はソースコード、checkpoint、ログへ埋め込みません。実際の接続はユーザーが明示的に
開始したときだけ行い、接続前に現在の公式条件を再確認します。

## 現在の判定

**未参加・未合格**です。外部サーバへは接続していません。

現在のMeteo checkpointは学習基盤の受入用ランダム初期重みです。32 neural simulations/手の
短い実動確認では、技巧2、NAGISA V3.1、AobaNNUE v1.1、水匠5の各2局に0勝2敗でした。
この結果は接続・終局・合法手記録が動く証拠であって、Floodgateへ投入できる棋力の証拠では
ありません。

## 接続前ゲート

次をすべて満たすまで公開対局へ接続しません。

1. 学習済みcheckpoint
   - ランダム重みではなく、権利確認済み教師、深い自己再解析、詰み・終盤anchorで学習済み
   - 固定held-out局面のloss、policy top-k、value calibrationが直前championより悪化していない
2. 合法性と耐久性
   - ローカルで終局まで1,000局以上
   - 反則、例外、プロセス停止、未処理の`stop`、不正な`win`、棋譜再生失敗が0件
   - 512手打切り条件と、上限なしの標準終局条件を別々に試験
3. 時間制御
   - CSAの残時間と1手10秒加算を認識するtime manager
   - 通常手、難所、詰み探索、ponder、通信余裕へ予算を分け、300秒を使い切らない
   - 低速化やメモリ圧迫時に探索を安全停止し、必ず合法手を返す
4. 棋力ゲート
   - NAGISA、AobaNNUE、公開水匠、技巧、過去Meteo championと先後同数で対局
   - 同一ハードウェア・同一持時間、定跡条件とハッシュ条件を固定
   - 最低400局、未完局0、勝点率とWilson 95%下限を保存
   - 「全勝候補」と呼ぶには、強い固定対手群で少なくとも400局の負け・引分がともに0
5. 公開前canary
   - 別プロセスのCSA bridgeからUSI Meteoを起動し、ローカルshogi-serverで再接続、先後、
     投了、千日手、入玉宣言、512手引分、サーバ終了を再現
   - engine binary、checkpoint、設定、Git SHA、NPS、peak memoryを一つのrun manifestへ固定

400局全勝でも、未知の相手への将来の全勝を統計的に保証することはできません。0敗/400局なら、
単純な二項モデルで真の勝率の95%片側下限は概ね99.25%ですが、相手や局面が独立同分布でない
Floodgateでは参考値に留めます。

## 強化ループ

Floodgate向けの学習は、単に公開棋譜の勝者の手を模倣する方式にはしません。

```text
権利確認済み公開棋譜・自前対局
  -> 局面重複を除去し、対局単位でtrain/held-outを分離
  -> Meteo浅探索、Meteo深探索、NNUE教師、DL教師へ同一SFENを渡す
  -> MultiPV、評価差、探索深さ、NPS、最善手逆転、詰み証明を保存
  -> 深すぎる教師はpolicy/valueを段階導入
  -> candidateをchampion、履歴、異種エンジンへ先後同数で評価
  -> 統計ゲートを通過したcandidateだけ昇格
  -> Floodgate実戦は最後にheld-out外部評価として解析
```

相手の弱みは、相手名、局面、戦型prefix、選択手、深教師から見たregretとして対局終了後にだけ
更新できます。対局中に相手傾向をpolicyへ混ぜる将来の接続でも30%を上限とし、全合法応手への
探索を残します。現在のFloodgate/標準USI経路には相手profile自動読込をまだ接続していません。
特定相手への過適合で一般棋力を落とさないよう、相手別勝率と全体arenaを同時に昇格条件にします。

## 1億ノードの扱い

`bench-nps`は要求ノード数、経過時間、NPS、MLX peak memory、保持可能な探索木ノード数、
tree recycle回数を表示します。現在の64 GiB Apple Silicon実測では、ResNet 20x256は単局約
105 NPS、32局面batch合計約781 NPSでした。単局1億ノードは現在約266時間、32局面それぞれ
1億ノードは約1,138時間の見積りです。

自動メモリ予算はOSと他アプリ用に約15%または8 GiBの大きい方を残し、現在の空きメモリから
MLXと探索木の上限を決めます。直近の実測では探索木を約205万Python nodeまで保持できます。
それを超える探索はroot統計を残してsubtreeを回収するため、落ちずに総ノード数を増やせても、
1億nodeの完全な木を保持する探索とは同じではありません。

実戦時間内の1億有効ノードを目指すには、次が必要です。

- Python objectの`Node`/`dict`を、連続配列のネイティブC++/Rust treeへ置換
- transposition table、前手からのtree reuse、virtual lossを備えた並列探索
- policy/value推論のdynamic batchingと、探索スレッドとの非同期pipeline
- 低prior枝の証明可能な再展開、詰み・必至用df-pn、時間制御に応じた選択的延長
- NPSだけでなく「最善手発見までのnode数」とheld-out勝率で高速化を判定

枝を早く捨てて表示上の1億nodeだけを作る改良は採用しません。低priorから浮上する長い勝ち筋を
守るため、root全合法手の最低訪問、MultiPV再解析、最善手逆転局面のanchorを維持します。

## 公開棋譜の利用境界

公式ページは2008年以降の年別棋譜archiveとSHA-256を公開しています。ダウンロード可能であることと、
自由な再配布・モデル学習が明示許諾されていることは同じではありません。利用条件を別途確認する
までは、原archiveや再配布用派生datasetをMeteo releaseへ同梱しません。取り込む場合はURL、取得日、
archive SHA-256、棋譜ID、train/held-out splitを保存し、同じ対局が学習と評価へ漏れないようにします。

### ローカル取り込み契約

`simajilord_shogi.floodgate`は、強豪AI同士のCSAを深い再解析前の候補corpusに変換します。
これは生棋譜の指し手を教師正解にするimporterではありません。

- CSAの原手は`chosen_move`だけに保存し、`policy`は空、`actor_best_move`と全`teacher_*`は
  未設定にする。通常の学習経路は教師policyがないためfail closedになる
- 対局結果は実測outcomeとして保存するが、教師評価値とは扱わない。学習前に履歴付き
  MultiPV、全PV、評価値、詰みを再生成する
- CSA V2の初期配置、先後、成り、不成、駒打ち、消費時間、投了、時間切れ、千日手、
  入玉宣言、中断、持将棋を区別し、全手を`rsshogi`で合法再生する。不正・未対応・
  千日手や詰みと盤面履歴の不整合は採用せず理由をmanifestへ残す。`%KACHI`は
  Floodgate serverの宣言受理結果を保存するが、現在の`rsshogi`は宣言rule未設定のため、
  `server_reported_declaration_not_locally_proven`と明記して局面上の証明とは扱わない
- レーティング表はplayer ID、表示名、rating、勝敗、最終対局時刻、URLだけでなく、
  HTML bytes全体のSHA-256を保存する。同じ日付URLでも後から対局が追加されるためである
- strong-vs-strongは両対局者がともにminimum ratingとminimum gamesを満たす場合だけ採用し、
  0敗playerには上方不安定のuncertainty flagを付ける
- engine family、version、hardwareは名前から推測しない。明示的なexact-name ruleがなければ
  すべて`unknown`とし、複数ruleの衝突は拒否する
- splitはraw CSA SHA-256を使うgame単位で決める。開始局面のように複数splitへ現れる
  normalized sampleは`sealed_final_test -> validation -> train`の保護順で一つだけに残す
- archive、rating HTML、各CSA、split、identity rule、除外理由、出力replayのSHA-256を
  create-onlyの一つのmanifestへ固定する

2026年archiveの非展開stream監査では37,897件を読み、37,114件を厳格に合法再生できました。
内訳は投了34,847、時間切れ720、局面履歴で証明できた千日手1,408、入玉宣言139です。
残り783件は終局記録なし、不完全なplayer記録、または局面履歴で千日手を再現できない等の
理由で自動除外されます。この数はarchiveのSHA-256
`74e8e624d17884847a8ac0b6d9acd46763661ae5c8c36a229b044bbdf18994c5`に限定したローカル監査値です。
