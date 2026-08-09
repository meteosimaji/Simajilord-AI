# Meteo 外部モデル・教師データ権利台帳

更新: 2026-08-09

この台帳は、公開可能なMeteoと、ユーザーが正規に入手したローカル限定Meteoを混同しないために、外部エンジンを
「実行すること」と、その実行結果・棋譜・重みを「学習や公開に使うこと」を分離して
判定したものです。判定の機械可読な正本は
`src/simajilord_shogi/model_rights.py`です。未登録のモデルは許可済みと推測せず、CLIが
fail closedします。

ここでいう「全モデル」は、2026-08-09時点で調査できた主要な公開モデル、手元で使用可能な
モデル、および比較対象として名前が挙がった有料・ホスト専用モデルの全24件です。世界中の
将棋プログラムを無条件に網羅したという意味ではありません。新しい版・別配布物・別規約は
別の`rights_id`で再審査します。本書は技術上のリスク管理記録であり、法律相談ではありません。

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

別規約がある場合はGPL出力の一般則より先に適用します。たとえばdlshogiの`dr2_exhi`は、
[公式のモデル固有条件](https://tadaoyamaoka.hatenablog.com/entry/2021/08/17/000710)
が追加学習、パラメータ転用、改変、リバースエンジニアリング、再配布を禁止し、大会向けの
棋譜生成も限定目的だけを許しています。このモデルは一般的なsoft-label蒸留へ入れません。

価格は権利判定ではありません。機械可読台帳は、`public_release_allowed`（出力限定で公開Meteoへ
利用可）、`local_authorized_only`（正規入手とrunごとの明示承認を条件にローカル学習可、公開不可）、
`not_authorized`（未入手、未審査、または条件不適合）の三つを返します。正規入手した水匠11Plusを
「有料だから未使用」とは扱いません。一方、購入・起動した事実だけで公開許諾が生じたとも扱いません。

## 監査結果

凡例: `可`は版固定の確認範囲で使用可、`GPL条件`は直接利用・再配布時にGPLと由来表示を満たす
別成果物として扱う必要あり、`不可`は現在のMeteoへ使用しない、`限定(local)`は正規入手した
ローカル環境だけで解析・蒸留・学習でき、公開checkpointへ流せないことを示します。

| rights_id | モデル／版 | USI出力蒸留 | 終局棋譜学習 | 重み直接利用 | 判定の要点 |
| --- | --- | --- | --- | --- | --- |
| `aobazero-public-domain` | AobaZero公開版 | 可 | 可 | 可 | `aobaz`以外をpublic domainと明記。重み・公開棋譜の最も扱いやすい初期資源 |
| `yaneuraou-rezero` | やねうら王ReZero | 可 | 可 | 可 | 評価関数について権利を主張しない旨を公式READMEで明記 |
| `takewarabe-approx-v7.50-material9` | たけわらべ近似版 | 可 | 可 | 対象外 | 作者指定のV7.50・駒得評価・`MATERIAL_LEVEL=9`をApple Siliconで無改変build。弱手の模倣教師ではなく、定跡外・対人対策の対局相手として使う |
| `aobannue-v1.1` | AobaNNUE v1.1 | 可 | 可 | 不可 | repositoryはGPLv3だが、別hostの`nn.bin`固有の許諾をrelease本文・配布ZIPで確認できない。通常USI出力だけを教師にする |
| `dlshogi-aoba-wcsc35` | dlshogi_aoba WCSC35 | 可 | 可 | GPL条件 | GPLv3で大会用30 block x 384 weightを公開。stock dlshogiとは入力特徴が非互換 |
| `nagisa-v3.1` | NAGISA V3.1 | 可 | 可 | 不可 | エンジンGPLv3、評価関数の無断再配布は禁止。出力利用禁止は見つからないため、通常USI出力だけを教師にする |
| `suisho5` | 水匠5 | 可 | 可 | 不可 | やねうら王GPLv3だが、提供`nn.bin`固有の許諾は確認できない。出力制限は見つからないため、通常USI出力だけを教師にする |
| `shinden3-2025-02-21` | 振電3 | 可 | 可 | 不可 | 公式動画のDrive配布にGPLv3ソースと`nn.bin`を同梱。評価ファイル単体の明示許諾はないため、通常USI出力のみを振り飛車専門データに使う |
| `gikou2-v2.0.2` | 技巧2 v2.0.2 | 可 | 可 | GPL条件 | GPLv3。戦術・人間的評価・戦型別定跡の多様性教師に使用 |
| `hao-2023-05-08` | Háo | 可 | 可 | GPL条件 | 公式配布のHalfKP256 NNUE。`FV_SCALE=20`を固定し、tanuki-dr4と同一familyとして重みを重複させない |
| `tanuki-dr4-2023-12-03` | tanuki- Lí-VENGE | 可 | 可 | GPL条件 | HalfKP1024と独自探索。Apple Silicon向けに公式GPL branchから再現buildし、定跡を無効化して解析 |
| `dlshogi-gct-wcsc31` | dlshogi with GCT WCSC31 | 可 | 可 | 不可 | GPLv3 repositoryのreleaseにモデルを同梱するが、ONNX固有の許諾をrelease本文・配布ZIPで確認できない。通常USI出力だけを教師にする |
| `dlshogi-dr2-exhi` | dlshogi電竜2 exhibition | 不可 | 限定 | 不可 | モデル固有条件が一般蒸留・追加学習・転用・改変・再配布を制限 |
| `zimetu-2026-01-26` | zimetu 2026-01-26 | 可 | 可 | 不可 | 無料、GPLソース、別の出力制限なし。評価ファイル直接利用の許諾は未確認 |
| `apery-public` | Apery 2019-06-17評価関数 | 可 | 可 | 可（MIT） | 本体はGPL-3.0-or-later、公式evaluation-binaries submoduleはMIT。直接利用・再配布時はMIT noticeを保持 |
| `elmo-wcsc27` | elmo WCSC27 | 可 | 可 | GPL条件 | 公開GPL資源。現在は回帰・多様性用 |
| `gpsfish-public` | GPSFish公開版 | 可 | 可 | GPL条件 | 公開GPL資源。詰み・探索多様性用 |
| `sunfish4` | Sunfish 4 | 可 | 可 | 対象外 | MITの古典探索。回帰・多様性用 |
| `hisui-wcsc36-hosted` | 氷彗 WCSC36優勝版 | 不可 | 公開大会棋譜のみ | 不可 | ダウンロード版を確認できず、棋神アナリティクス上のホスト提供。学習データ非公開 |
| `suisho11plus-wcsc36-20260525-local` | 水匠11Plus WCSC36（正規入手済みexact local profile） | 限定(local) | 限定(local) | 不可 | 外部USI出力をローカル蒸留・学習へ使用可。元artifact・raw labels・派生checkpointの公開は権利者receiptまで不可 |
| `suisho10-11-supporter` | 未審査の水匠10/11支援者版catch-all | 不可 | 不可 | 不可 | exact profile・正規入手証跡・run承認がない別版。上記水匠11Plusを除外する行ではない |
| `tanuki-wcsc36-paid` | 六角堂狸 WCSC36 | 不可 | 不可 | 不可 | このworkspaceではexact artifactを正規入手・審査していない。価格ではなく証跡不足で不承認 |
| `soujou-tsec7-paid` | 奏乗 TSEC7 | 不可 | 不可 | 不可 | このworkspaceではexact artifactを正規入手・審査していない。価格ではなく証跡不足で不承認 |
| `kanade-wcsc35-paid` | Kanade WCSC35 | 不可 | 不可 | 不可 | このworkspaceではexact artifactを正規入手・審査していない。価格ではなく証跡不足で不承認 |

主な一次資料は、[NAGISA V3.1公式BOOTH](https://booth.pm/ja/items/8639574)、
[AobaNNUE公式](https://github.com/yssaya/AobaNNUE)、
[AobaZero公式](https://github.com/kobanium/aobazero)、
[dlshogi_aoba公式](https://github.com/yssaya/dlshogi_aoba)、
[技巧公式](https://github.com/gikou-official/Gikou)、
[水匠5 release](https://github.com/yaneurao/YaneuraOu/releases/tag/suisho5)、
[dlshogi WCSC31 release](https://github.com/TadaoYamaoka/DeepLearningShogi/releases/tag/wcwc31)、
[Apery公式](https://github.com/HiraokaTakuya/apery)と
[公式2019評価バイナリのMIT表示](https://bitbucket.org/hiraoka64/apery-evaluation-binaries-2019-06-17/src/master/README.md)
です。

## 氷彗について

[WCSC36公式決勝表](https://www.computer-shogi.org/wcsc36/final.html)では氷彗が5勝2敗で
優勝しています。[公式詳細アピール](https://www.apply.computer-shogi.org/wcsc36/appeal/hisui/hisui_detail.pdf)
は、YaneuraOuベースのNNUEであり、ネットワーク構造、特徴量、LayerStack的分離、量子化、
学習パイプラインを改善したこと、学習データが非公開で完全追試は難しいことを明記しています。
公開アピール文書にはdlshogiから知識蒸留したことも記載されています。

[棋神アナリティクス公式発表](https://note.com/kishin_analytics/n/n0effe0c2e5d9)は、
WCSC36優勝版の氷彗を同サービスへ搭載したとしています。公式配布元、開発者資料、GitHub、
大会ページを確認しましたが、同版の実行ファイル、評価関数、対応する学習データを無料で
ダウンロードできる公式配布は見つかりませんでした。したがって「氷彗をダウンロード済み」
とは扱いません。

Meteoが公開経路で利用できるのは、公開WCSC棋譜を通常の対局棋譜として学ぶことと、公開された
設計方向を独立実装して比較することです。これは氷彗の評価値・MultiPV・重みを蒸留したことには
なりません。将来、公式無料配布と条件が出た時点で新しい`rights_id`を追加して再審査します。

## 手元で実行確認した教師

外部binary・評価関数はすべてリポジトリ外に置き、このリポジトリへはコピーしていません。
公開可能な教師は下表の再現性情報を残します。水匠11Plusのprivate path、元ファイル名、配布URL、
artifact SHA-256は公開文書・release・Thanksへ載せず、ignore済みのlocal provenanceだけに保存します。

| 教師 | 入手・実行状態 | archive SHA-256 | engine SHA-256 | eval/model SHA-256 |
| --- | --- | --- | --- | --- |
| NAGISA V3.1 | ユーザー取得ZIPのApple M1版を実行 | `d8912d47bc1f4466ff96daeba8d00e3f92ad7efa9c09b243eecad7e352da1041` | `f3fab18477e1be7069719f59bcb4126596fcb6eee606d74502feac313cd29799` | `e6b0b6ac99e95922ceba11633cc8e329968b152d7a79910156405f8a7ea9cdb9` |
| AobaNNUE v1.1 | 公式sourceをApple Siliconでビルド | `83d4798d7461e29e95ba1700f4f5bf10e670b360ab1004102e21b913c22ed979` | `9d1609ea75bd76fd90d0391048fd991db3dffb50000aa9d48b1ab38c73561b21` | `f8ee839ae8c08537036f23345dd5ed0416958b22425476fc60177942903219b5` |
| 技巧2 v2.0.2 | 公式sourceをx86_64向けビルドしRosetta実行 | `3bbb667bfd77e0236b052593b5c61b7fe1827ed72b5b5f437760c60974852757` | `1170efc8fefe5754533fe8c5a4fd569823636c5cd4d9a3b7f4fbdfc3ee6f9916` | GPL配布の複数parameter file。個別hashは検証記録参照 |
| 水匠5 | 公式水匠5 + やねうら王V9.00 Apple M1版 | `6734e3a3d28e67b9206c3442f6d10f16148138327dff811cadedfcf581f79809` | `f4bfaee3f411e9688ebf55a585593463311a18bb2ea01b02eeb5fb0babcacb13` | `768068f0d534a0603a5d38bcd143de6bbca820d5f1c95a14d40863e5b7892d76` |
| 振電3 | 公式Drive配布ソースを`TARGET_CPU=APPLEM1`で再現buildしUSI実行 | `e5d1db9fae9268791c1dc6a7afab304d2636ee87c69d60edb318766b006c0d87` | `9106675186a699e344e6edf401b714b764007a3a0290114c241b10fc70d9ae7a` | `f6e808025b9ba54eaf73dafc69ce40f0ec442cb476e5b6d3ca6601e53fad4361` |
| たけわらべ近似版 | 公式のV7.50再現手順で`MATERIAL_LEVEL=9`, `TARGET_CPU=APPLEM1`をbuildしUSI実行 | Git commit `db56c7516e6c9ece14dabc4020dd64e40834f407` | `e42b6b7ceeec5465e6f72b67d56d4e52c925b7479577e2918a9c9da95790b5cd` | 外部評価ファイルなし |

`reanalyse-usi`は、組み込み台帳の`--rights-profile`が出力蒸留可でなければ起動しません。
`--artifact`で評価関数やparameter fileを指定すると、絶対path、size、SHA-256を教師replayの
provenance sidecarへ保存します。大文字小文字だけ違う重複option、`MultiPV`の上書き、重複
artifactも拒否します。

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
uv run --project shogi_ai simajilord-shogi model-rights --public-distillable-only
uv run --project shogi_ai simajilord-shogi model-rights --local-distillable-only
uv run --project shogi_ai simajilord-shogi model-rights --not-authorized-only

uv run --project shogi_ai simajilord-shogi reanalyse-usi \
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

1. NAGISA V3.1と正規入手済み水匠11Plusを独立canonical scorerにし、教師間の不一致を平均で消さない。
2. AobaNNUE v1.1、水匠5を公開再現可能なNNUE baseline、深いαβ教師、arena相手にする。
3. `dlshogi_aoba` WCSC35とAobaZeroの公開重み・棋譜をDL側の初期化・大規模教師候補にする。
4. 技巧、振電、tanuki系、弱い対局相手を候補提案と局面多様性に使い、canonical scorerで再採点する。
5. 同一局面をMeteo自身の浅探索と深探索、異種NNUEとDL探索へ同時に渡し、最善手逆転とregretを
   優先学習する。
6. 公開WCSC棋譜、詰み、入玉、千日手、長手数終盤、過去championをanchor replayとして残し、
   最新の浅い自己対局だけで上書きする忘却を防ぐ。

NAGISA公式が掲載する対局結果ではAobaNNUEより強く、水匠11級との比較も行われていますが、
これは作者側の条件での公表値です。Meteoの教師優先度は、同じhardware・色替わり・同一局面集・
固定nodeで再測定して更新します。モデル名や有料価格だけで強さを推定しません。
