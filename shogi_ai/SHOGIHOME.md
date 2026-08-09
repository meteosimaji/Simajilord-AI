# ShogiHome 連携方針

## 結論

Meteo は ShogiHome を改造せず、通常の USI engine として接続する。ShogiHome 内に
学習status panelが必須になるまでforkは作らない。盤面、合法手入力、先後選択、投了、
棋譜表示、自動棋譜保存はupstream ShogiHomeに任せ、Meteo側は次だけを担う。

- 昇格済みcheckpointをatomic manifestで公開する。
- `usinewgame` でのみmanifestを読み、generationとhashを1局中固定する。
- `go` の `info string` に固定generation/hashと別processの学習statusを出す。
- `gameover win|lose|draw` をengine視点から先手・後手の勝者へ変換する。
- USIから観測できた棋譜prefixをcandidate-only hash-chain logへ残す。
- ShogiHomeの自動保存棋譜と照合し、教師再解析・dedup前は学習に入れない。

`simajilord_shogi.human_gui` のHTTP盤面は障害調査用fallbackであり、製品GUIではない。

## 一次資料監査

2026-08-09に `sunfish-shogi/shogihome` の `main` commit
`6b8f5fdface83dda8149a8b3a97b4da4fab1a6ae` を確認した。

- 本体はMIT License。著作権表示とpermission noticeの維持が必要。
- Material IconsはApache License 2.0。配布物には各dependencyのnoticeも必要。
- 外部AIは通常USI executableとして登録され、engineのdirectoryをcwdとして直接spawnされる。
- 対局終了時にKIF/KIFU/KI2/KI2U/CSA/JKFを自動保存できる。
- human playerを含む対局はrepeatとparallelismが1に制限されている。
- runtime plugin APIは確認できない。ShogiHome内status panelの追加はsource forkになる。

一次資料:

- <https://github.com/sunfish-shogi/shogihome>
- <https://github.com/sunfish-shogi/shogihome/blob/6b8f5fdface83dda8149a8b3a97b4da4fab1a6ae/LICENSE>
- <https://github.com/sunfish-shogi/shogihome/blob/6b8f5fdface83dda8149a8b3a97b4da4fab1a6ae/src/background/usi/process.ts>
- <https://github.com/sunfish-shogi/shogihome/blob/6b8f5fdface83dda8149a8b3a97b4da4fab1a6ae/src/renderer/game/game.ts>
- <https://github.com/sunfish-shogi/shogihome/blob/6b8f5fdface83dda8149a8b3a97b4da4fab1a6ae/src/common/settings/game.ts>

## 実行境界

checkpointの公開は学習processが、USIはShogiHomeが起動する別processが行う。

```bash
python -m simajilord_shogi.human_gui publish \
  /absolute/checkpoints \
  /absolute/meteo-gui-state \
  /absolute/checkpoints/generation-42 \
  --generation 42

python -m simajilord_shogi.human_gui training-status \
  /absolute/meteo-gui-state/training-status.json \
  --state running --generation 43 --progress 0.25 \
  --message '水匠11PlusとNAGISAで再解析中' --process-id 12345

python -m simajilord_shogi.human_gui usi \
  /absolute/checkpoints \
  /absolute/meteo-gui-state \
  /absolute/human-candidates.jsonl \
  --simulations 1600 --max-tree-nodes 500000
```

ShogiHomeはengine pathへ追加argumentを渡さず直接spawnする。このため最後のcommandを
固定絶対pathで `exec` する小さなlauncher executableをローカルに用意して登録する。
相対path、symlinkされたcheckpoint、作成途中のcheckpointは使わない。

ShogiHomeでは自動棋譜保存を必ず有効にし、機械処理にはCSAまたはJKFを優先する。
USIの `gameover` より前に人間の終局手がengineへ通知される保証はないため、Meteoの
USI観測logだけを完全棋譜として扱ってはならない。logにはこの制約が固定fieldとして
保存される。

## 並行学習の安全性

state整合性は次の境界で守る。

- checkpoint作成中のtemporary directoryは公開しない。
- active pointerはcomplete checkpointをhash検証してから最後にatomic replaceする。
- 局中のUSI processはactive pointerを再読込しない。
- 人間対局とtrainerはmodel/optimizer/RNG objectを共有しない。
- checkpoint pinと棋譜sessionは`usinewgame`から`gameover` / `quit`まで維持するが、
  compute leaseは各`go`探索の間だけ`human-play-leases/<session>.json`としてheartbeatする。
  人間が考えている間はleaseを持たず、学習・自己対局・蒸留を継続する。
- trainerは循環importを避ける専用adapter
  `simajilord_shogi.compute_interlock.training_step` を使い、実際のoptimizer stepを
  `with training_step(state_root, generation=...):` で囲む。これは内部で同じ
  `ComputeInterlock` implementationへ接続する。
- `train` / `train-psv` / `selfplay` / `reanalyse` / `improve` / `benchmark-usi` は共通の
  `--human-play-state-root` を受け取る。optimizer初期化、固定probe、各optimizer stepに加え、
  自己対局・深再解析・arenaのneural inferenceもbatch単位で同じleaseを使う。
- human leaseとtraining-step leaseの確認・作成は同じprocess lock内で行う。
  待機pollはlock外なので、片方が他方のreleaseを妨げない。
- 人間の`go`が開始待ちに入ると`.human-priority.lock`を保持し、現在の短いneural batch/stepが
  終わった後はtrainerがleaseを即座に取り直せない。これは一手の応答を飢餓させないためで、
  対局全体を優先するものではない。複数trainerも同じstate rootでは
  Metal computeを1件ずつに直列化するため、別々のAdamW/RNG実験を壊さずinterleaveできる。
- どちらのleaseも短いTTLを持ち、期限切れleaseは復活できない。正常なtraining-step
  leaseは1 step後に削除し、USI異常終了時のhuman leaseはTTL後に回復できる。
- lease directoryのpartial JSON、hash不一致、予期しないartifact、symlinkは
  「利用中」と同等にfail closedし、optimizer stepも対局開始も許可しない。
- human candidate logはappend-only、fsync、hash chain、process lockで保存する。
- session versionとlocalhost tokenはHTTP debug fallbackだけの境界である。

Apple Siliconでは学習と対局探索が同じMetal GPU・unified memoryを競合するため、processを
分けるだけでは応答時間やOOMを防げない。現在はoptimizer/probe、自己改善MCTS、各`go`の
実推論だけを直列化し、対局の思考待ち時間は進化へ返している。ただし、各processが保持する
model/optimizer常駐memory、checkpoint
load、CPU側prefetchまで解放する仕組みではない。したがって、state破損と同時Metal実行は防ぐが、
物理memory容量を超える数の学習processを起動してよいとは解釈しない。batch、Metal memory、
USI tree-nodeにも別々の上限を設定する。

## License境界

ShogiHomeはGPLではなくMITなので、MeteoのApache-2.0 repositoryとprotocolで接続するだけなら
code licenseは混ざらない。将来forkする場合は別repositoryを推奨し、ShogiHomeのMIT本文・
copyright、Material IconsのApache-2.0、生成したthird-party noticesをfork配布物に維持する。
ShogiHome sourceをこのrepositoryへ無断で丸ごとcopyしない。
