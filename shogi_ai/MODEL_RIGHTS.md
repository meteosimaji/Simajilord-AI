# Meteo 外部モデル・教師データ権利台帳

更新: 2026-08-12

この台帳は、公開可能なMeteoと、ユーザーが正規に入手したローカル限定Meteoを混同しないために、外部エンジンを
「実行すること」と、その実行結果・棋譜・重みを「学習や公開に使うこと」を分離して
判定したものです。判定の機械可読な正本は
`src/simajilord_shogi/model_rights.py`です。未登録のモデルは許可済みと推測せず、CLIが
fail closedします。

現行台帳は、同条件実測で選定した9モデルだけを登録します。過去の24件台帳や10モデル比較に
含まれたモデル名が履歴artifactに残っていても、現行CLIで有効な`rights_id`ではありません。
新しい版・別配布物・別規約を使う場合は、既存行へ暗黙に読み替えず、利用前に台帳変更として
再審査します。本書は技術上のリスク管理記録であり、法律相談ではありません。

## 判定原則

[GNU GPLv3 §2](https://www.gnu.org/licenses/gpl-3.0.html#section2)は、改変していない
プログラムを実行する無制限の許可を明記し、実行出力がGPL対象になるのは、出力の内容自体が
covered workを構成するときだけとしています。[GNU GPL FAQ](https://www.gnu.org/licenses/gpl-faq.html#WhatCaseIsOutputGPL)
も、プログラム出力には通常、コードのライセンスは及ばず、プログラムから保護対象の表現が
出力へコピーされる場合が例外だと説明しています。

したがってMeteoでは、GPLエンジンが通常のUSIとして返す指し手、数値評価値、探索ノード数、
PV、MultiPV確率を、元プログラムの表現を含まない数値的な観測出力として扱います。別の
出力利用制限が見つからない場合、その出力から独立したMeteoを学習することを許可します。
これは次を許可するものではありません。

- 元の評価関数、ONNX、`nn.bin`、定跡、ソースをMeteoへコピーすること
- 元の重みを形式変換して「別モデル」と呼ぶこと
- 配布物固有の利用規約、契約、再配布禁止を無視すること
- 元プログラムのコードや著作物をMeteoの出力へ埋め込むこと

別規約がある場合はGPL出力の一般則より先に適用します。現行9モデル以外は、過去に調査済みでも
未登録としてfail closedし、必要になった時点で配布物固有の条件をもう一度確認します。

価格は権利判定ではありません。機械可読台帳は、`public_release_allowed`（出力限定で公開Meteoへ
利用可）、`local_authorized_only`（正規入手とrunごとの明示承認を条件にローカル学習可、公開不可）、
`not_authorized`（未入手、未審査、または条件不適合）の三つを返します。正規入手した水匠11Plusと
ユーザー提供の奏乗TSEC7を「有料だから未使用」とは扱いません。一方、購入・起動した事実だけで
公開許諾が生じたとも扱いません。

## 監査結果

凡例: `可`は版固定の確認範囲で使用可、`GPL条件`は直接利用・再配布時にGPLと由来表示を満たす
別成果物として扱う必要あり、`不可`は現在のMeteoへ使用しない、`限定(local)`は正規入手した
ローカル環境だけで解析・蒸留・学習でき、公開checkpointへ流せないことを示します。

| rights_id | モデル／版 | USI出力蒸留 | 終局棋譜学習 | 重み直接利用 | 判定の要点 |
| --- | --- | --- | --- | --- | --- |
| `aobannue-v1.1` | AobaNNUE v1.1 | 可 | 可 | 不可 | repositoryはGPLv3だが、別hostの`nn.bin`固有の許諾をrelease本文・配布ZIPで確認できない。通常USI出力だけを教師にする |
| `nagisa-v3.1` | NAGISA V3.1 | 可 | 可 | 不可 | エンジンGPLv3、評価関数の無断再配布は禁止。出力利用禁止は見つからないため、通常USI出力だけを教師にする |
| `suisho5` | 水匠5 | 可 | 可 | 不可 | やねうら王GPLv3だが、提供`nn.bin`固有の許諾は確認できない。出力制限は見つからないため、通常USI出力だけを教師にする |
| `shinden3-2025-02-21` | 振電3 | 可 | 可 | 不可 | 公式動画のDrive配布にGPLv3ソースと`nn.bin`を同梱。評価ファイル単体の明示許諾はないため、通常USI出力のみを振り飛車専門データに使う |
| `gikou2-v2.0.2` | 技巧2 v2.0.2 | 可 | 可 | GPL条件 | GPLv3。戦術・人間的評価・戦型別定跡の多様性教師に使用 |
| `hao-2023-05-08` | Háo | 可 | 可 | GPL条件 | 公式配布のHalfKP256 NNUE。`FV_SCALE=20`を固定し、tanuki-dr4と同一familyとして重みを重複させない |
| `tanuki-dr4-2023-12-03` | tanuki- Lí-VENGE | 可 | 可 | GPL条件 | HalfKP1024と独自探索。Apple Silicon向けに公式GPL branchから再現buildし、定跡を無効化して解析 |
| `suisho11plus-wcsc36-20260525-local` | 水匠11Plus WCSC36（正規入手済みexact local profile） | 限定(local) | 限定(local) | 不可 | 外部USI出力をローカル蒸留・学習へ使用可。元artifact・raw labels・派生checkpointの公開は権利者receiptまで不可 |
| `soujou-tsec7-paid` | 奏乗 TSEC7（ユーザー提供exact local profile） | 限定(local) | 限定(local) | 不可 | 外部USI出力をローカル蒸留・学習へ使用可。`FV_SCALE=28`・専用`progress.bin`を固定し、元artifact・raw labels・派生checkpointは公開不可 |

現NAGISA型NNUE本学習では、NAGISAの`nn.bin`をコピー・変換・初期値利用しません。正規入手済み
archiveから`progress.bin`だけを、9個のLayerStackを選ぶ固定routerとしてローカルrunへ抽出します。
これは学生の評価重みではありませんが配布物の一部なので、archive hash・member・size・SHA-256を
receiptに残し、runと全exportをlocal-onlyにします。公開・販売・再配布には別の明示許諾、または
独立に生成して互換性を検証したprogress routerが必要です。

主な一次資料は、[NAGISA V3.1 GitHub release](https://github.com/keinoda/YaneuraOu/releases/tag/nagisa-v3.1)、
[NAGISA V3.1公式BOOTH](https://booth.pm/ja/items/8639574)、
[奏乗TSEC7公式BOOTH](https://booth.pm/ja/items/8606196)、
[奏乗TSEC7対応ソース](https://github.com/keinoda/YaneuraOu/tree/sojo_tsec7)、
[AobaNNUE公式](https://github.com/yssaya/AobaNNUE)、
[技巧公式](https://github.com/gikou-official/Gikou)、
[水匠5 release](https://github.com/yaneurao/YaneuraOu/releases/tag/suisho5)、
[振電3公式配布案内](https://www.youtube.com/watch?v=07PhE_I6c1s)、
[Háo公式release](https://github.com/nodchip/tanuki-/releases/tag/tanuki-.halfkp_256x2-32-32.2023-05-08)、
[tanuki-dr4公式release](https://github.com/nodchip/tanuki-/releases/tag/tanuki-dr4)、
[水匠11 WCSC36公式アピール](https://www.apply.computer-shogi.org/wcsc36/appeal/appeal_round2_260502.pdf)
です。

## 手元で実行確認した教師

外部binary・評価関数はすべてリポジトリ外に置き、このリポジトリへはコピーしていません。
公開可能な教師は下表の再現性情報を残します。水匠11Plusと奏乗TSEC7のprivate path、元ファイル名、
配布URL、artifact SHA-256は公開文書・release・Thanksへ載せず、ignore済みのlocal provenanceだけに保存します。

| 教師 | 入手・実行状態 | archive SHA-256 | engine SHA-256 | eval/model SHA-256 |
| --- | --- | --- | --- | --- |
| NAGISA V3.1 | ユーザー取得ZIPのApple M1版を実行 | `d8912d47bc1f4466ff96daeba8d00e3f92ad7efa9c09b243eecad7e352da1041` | `f3fab18477e1be7069719f59bcb4126596fcb6eee606d74502feac313cd29799` | `e6b0b6ac99e95922ceba11633cc8e329968b152d7a79910156405f8a7ea9cdb9` |
| 水匠11Plus | 正規入手したexact local profileを実行 | local provenanceのみ | local provenanceのみ | local provenanceのみ |
| 奏乗 TSEC7 | ユーザー提供のexact local profileを専用設定で実行 | local provenanceのみ | local provenanceのみ | local provenanceのみ |
| AobaNNUE v1.1 | 公式sourceをApple Siliconでビルド | `83d4798d7461e29e95ba1700f4f5bf10e670b360ab1004102e21b913c22ed979` | `9d1609ea75bd76fd90d0391048fd991db3dffb50000aa9d48b1ab38c73561b21` | `f8ee839ae8c08537036f23345dd5ed0416958b22425476fc60177942903219b5` |
| tanuki- Lí-VENGE | 公式GPL branchをApple Siliconで再現build | 検証記録参照 | `69031e5303183a3657a6c6e9727ee6273d24b450da8e4d2ced8762721016333f` | `05aadfe144a3f4b0a5b4d1c35cbb869514258a55a34021916cea50be58db2829` |
| 技巧2 v2.0.2 | 公式sourceをx86_64向けビルドしRosetta実行 | `3bbb667bfd77e0236b052593b5c61b7fe1827ed72b5b5f437760c60974852757` | `1170efc8fefe5754533fe8c5a4fd569823636c5cd4d9a3b7f4fbdfc3ee6f9916` | GPL配布の複数parameter file。個別hashは検証記録参照 |
| 水匠5 | 公式水匠5 + やねうら王V9.00 Apple M1版 | `6734e3a3d28e67b9206c3442f6d10f16148138327dff811cadedfcf581f79809` | `f4bfaee3f411e9688ebf55a585593463311a18bb2ea01b02eeb5fb0babcacb13` | `768068f0d534a0603a5d38bcd143de6bbca820d5f1c95a14d40863e5b7892d76` |
| 振電3 | 公式Drive配布ソースを`TARGET_CPU=APPLEM1`で再現buildしUSI実行 | `e5d1db9fae9268791c1dc6a7afab304d2636ee87c69d60edb318766b006c0d87` | `9106675186a699e344e6edf401b714b764007a3a0290114c241b10fc70d9ae7a` | `f6e808025b9ba54eaf73dafc69ce40f0ec442cb476e5b6d3ca6601e53fad4361` |
| Háo | 公式HalfKP256評価を`FV_SCALE=20`で実行 | 検証記録参照 | `f4bfaee3f411e9688ebf55a585593463311a18bb2ea01b02eeb5fb0babcacb13` | `1141d275bceec911156801f27303dc9ff5beb24f4f59144cc069306c59e80782` |

NAGISA GitHub版Apple Silicon archiveの公式digestは
`caf11a1ceee41b8fb52c8282419f515865461884089319d467ecb241d1b59280`です。手元のBOOTH ZIPは
packagingが異なるためarchive hashは一致しませんが、実行ファイル、`nn.bin`、`progress.bin`、
`eval_options.txt`の4 payloadはGitHub版とbyte単位で一致しました。releaseはv3.1の変更をSPSAによる
探索parameter再調整のみとし、NNUEはv3から不変、YaneuraOu 9.60、`HalfKA_hm2 1024x16x64 /
LayerStack 9`、`FV_SCALE=28`、`progress8kpabs`、commit
`640f46561455436641b2eafb6fb75dfbeaf21f3f`を明示しています。

`reanalyse-usi`は、組み込み台帳の`--rights-profile`が出力蒸留可でなければ起動しません。
`--artifact`で評価関数やparameter fileを指定すると、絶対path、size、SHA-256を教師replayの
provenance sidecarへ保存します。大文字小文字だけ違う重複option、`MultiPV`の上書き、重複
artifactも拒否します。

## 選定9モデルの同条件実測

選定前の10モデル比較（定跡なし、1 thread、1手20,000 node、reserved opening 2局面、先後入替）から、
現行台帳外の対局だけを除いて9モデル・全36組・144局へ再集計しました。各モデル32局なので順位差の
信頼区間はまだ広く、これはElo確定値ではなく、教師の大分類と追加深掘り対象を決めるdescriptive
probeです。履歴artifact自体は再現性のため上書きしません。

| 順位 | モデル | 勝-分-敗 | 得点率 | Wilson 95% |
| ---: | --- | ---: | ---: | ---: |
| 1 | 奏乗 TSEC7 | 27-0-5 | 0.844 | 0.682–0.931 |
| 2 | NAGISA V3.1 | 26-0-6 | 0.812 | 0.647–0.911 |
| 3 | 水匠11Plus | 23-0-9 | 0.719 | 0.546–0.844 |
| 4 | AobaNNUE v1.1 | 18-0-14 | 0.562 | 0.393–0.718 |
| 5 | tanuki- Lí-VENGE | 13-0-19 | 0.406 | 0.255–0.577 |
| 6 | 水匠5 | 11-0-21 | 0.344 | 0.204–0.517 |
| 7 | 振電3 | 10-1-21 | 0.328 | 0.192–0.501 |
| 8 | 技巧2 | 9-1-22 | 0.297 | 0.167–0.470 |
| 9 | Háo | 6-0-26 | 0.188 | 0.089–0.353 |

完全な設定・engine/eval SHA-256・全pairwise結果・replayは
[`artifacts/runs/meteo-clean-reset-20260811-v1/benchmarks/model-strength-roundrobin-10x2-n20k-v1/report.json`](artifacts/runs/meteo-clean-reset-20260811-v1/benchmarks/model-strength-roundrobin-10x2-n20k-v1/report.json)
に固定しています。現行9モデルの順位は同artifactのpairwise結果から決定論的に再集計しています。
上位3者はこの小標本でも他6者から明確に抜けたためcanonical scorer、AobaNNUEと水匠5は公開再現
baseline、技巧・振電・tanuki・Háoは局面分布を広げる
対局相手／候補提案者とし、その手を正解扱いせず上位3者で再解析します。

## checkpointへの権利制約継承と旧checkpoint移行

教師sidecarはローカル運用記録です。そこにはengine・棋譜・artifactのpathやhash、入手元などが
入り得るため、公開checkpointへsidecar本体をコピーしません。`train` / `train-psv`はsidecarを
読み取った時点で、次の項目だけを`lineage.rights_restriction_summary`へ正規化します。

- 台帳で検証した`rights_id`と公開判定
- 有効な`publication_allowed`
- summaryを元sidecarのbytesへ結び付けるsidecar SHA-256
- 権利者receiptが対象を一意に指定する決定論的restriction ID

外部URL、ローカルpath、engine・評価関数・棋譜のhash、artifact名はこのsummaryへ入れません。
合議replayの一般名`meteo-teacher-ensemble`は、同じsidecar内の各教師の正確な`rights_id`へ
解決できた場合だけ単独の未登録教師として扱いません。NAGISAだけのsummaryは現在の台帳判定で
公開可ですが、水匠11Plusを一度でも含むsummaryは、権利者の明示許可receiptがない限り公開を
拒否します。exact resumeは親checkpointのsummaryとrestriction IDを和集合で引き継ぎます。

このsummary導入前の既存checkpointは上書きしません。教師lineageがあるのにsummaryがない旧版は
release guardがfail closedします。移行時は元checkpointを保存したまま、隣接sidecarのbytesと
記録済みhashが一致することを再検証し、そこからsummaryを持つ新しい兄弟checkpoint候補を作ります。
sidecarを再検証できない場合は推測で許可済みにせず、旧lineage欠落用restriction IDを含む新候補に
対して、権利者の許可証跡hash・全restriction ID・その新checkpoint全体のhashを結び付けた
adjacent receiptを用意します。receiptは元の教師artifactやsidecarを公開してよい許可にはなりません。

```bash
uv run simajilord-shogi model-rights --public-distillable-only
uv run simajilord-shogi model-rights --local-distillable-only
uv run simajilord-shogi model-rights --not-authorized-only

uv run simajilord-shogi reanalyse-usi \
  artifacts/actor.jsonl artifacts/gikou-tactics.jsonl \
  --engine /path/to/gikou --engine-cwd /path/to/gikou-data \
  --rights-profile gikou2-v2.0.2 \
  --selection tactical --teacher-tag gikou-tactics \
  --nodes 1000000 --multipv 32 \
  --artifact /path/to/params.bin --artifact /path/to/probability.bin
```

技巧の公式READMEは全戦型、矢倉、相掛かり、横歩取り、角換わり、四間飛車、中飛車、三間飛車、
向かい飛車、その他の10種の定跡を提供しています。`BookFile`を戦型別に切り替え、
`--teacher-tag yagura`などで教師の文脈を分ければ、同じ局面でも「戦法を選んだ教師」として
混同せず学習できます。戦術蒸留は定跡なし・`--selection tactical`で、王手、王手応手、捕獲、
成り、王手をかける着手を優先します。

## 強さと多様性を優先した教師の順序

1. NAGISA V3.1、正規入手済み水匠11Plus、ユーザー提供の奏乗TSEC7を独立canonical scorerにし、
   三教師間の不一致を平均で消さない。三者の候補和集合を全三者で再採点し、最悪教師regretを使う。
2. AobaNNUE v1.1、水匠5を公開再現可能なNNUE baseline、深いαβ教師、arena相手にする。
3. 技巧、振電、tanuki- Lí-VENGE、Háoを候補提案と局面多様性に使い、canonical scorerで再採点する。
   Háoとtanuki- Lí-VENGEは同じtanuki familyとして相関を考慮し、独立2票として過大評価しない。
4. 同一局面をMeteo自身の浅探索と深探索、異種NNUE探索へ同時に渡し、最善手逆転とregretを
   優先学習する。
5. 公開WCSC棋譜、詰み、入玉、千日手、長手数終盤、過去championをanchor replayとして残し、
   最新の浅い自己対局だけで上書きする忘却を防ぐ。

NAGISA公式が掲載する対局結果ではAobaNNUEより強く、水匠11級との比較も行われていますが、
これは作者側の条件での公表値です。Meteoの教師優先度は、同じhardware・色替わり・同一局面集・
固定nodeで再測定して更新します。モデル名や有料価格だけで強さを推定しません。
