# Meteo 将棋AI調査メモ

更新: 2026-08-09

このメモは、「めてお / Meteo」を無料で再現可能な大会級将棋AIへ育てるために、
たややん氏／水匠の公開情報、やねうら王の公式資料、WCSCアピール文書から
実務上の要点をまとめたものです。他エンジンの主張とMeteoの実測は分けて扱います。

## 確認した公開情報

1. たややん氏の[2026年最新の将棋AI事情](https://book.mynavi.jp/shogi/detail/id%3D149624)は、
   NNUEはCPUで高速に広く読み、DL+MCTSは推論が重い代わりに局面評価が高精度と
   整理しています。2026年1月時点の無償最強候補として `AobaNNUE` を明記しているため、
   無料方針のMeteoでは教師・arena対手の第一候補にします。これは著者の時点付き見解であり、
   Meteoの実測勝敗ではありません。
2. [CEDEC 2024公式講演ページ](https://cedil.cesa.or.jp/cedil_sessions/view/2958)と
   [本人チャンネルの講演映像](https://www.youtube.com/watch?v=TWXP88bTyoQ)は、公開データで
   トップクラスを作る環境、NNUEとDL、自己対局データ、探索改良を扱っています。
   [講演レポート](https://game.watch.impress.co.jp/docs/kikaku/1617895.html)は、DL側の
   attention/Transformer、教師データの序盤・終盤配分、自動定跡、MCTSとαβ探索の
   アイデア融合を改良候補として整理しています。
3. たややん氏の[電竜戦振り返り](https://tayayan-ts.hatenablog.com/)では、評価関数の計測系に
   よって勝率結果が揺れたこと、floodgateと自己対局の約3万棋譜を定跡に利用したこと、
   単純勝数では逆転勝ちや勝率の低い手を誤って高評価する問題、末端の誤評価をツリー全体へ
   伝播させる危険が報告されています。
4. やねうら王公式の[水匠5のR+100記事](https://yaneuraou.yaneu.com/2021/11/26/tayayan-suisho-5-makes-r100-up/)
   は、一台のRTX 3090でもデータ採取・分析・実験設計により大幅改善した例を報告して
   います。`FV_SCALE` はNNUE固有の実験なので、DLのMeteoへ数値をそのまま移植は
   しません。「同じ予算でもラベルの解像度と学習過程を実験する」原則を取り入れます。
5. やねうら王Wikiの[ふかうら王の学習手順](https://github.com/yaneurao/YaneuraOu/wiki/%E3%81%B5%E3%81%8B%E3%81%86%E3%82%89%E7%8E%8B%E3%81%AE%E5%AD%A6%E7%BF%92%E6%89%8B%E9%A0%86)
   は、棋譜内の近接局面の強い自己相関、未知局面の検証accuracy、ResNetの
   warmup + cosine annealing、Transformerでのwarmupの重要性、教師ファイルごとの評価値→
   勝率校正、性質の異なるデータ混合を説明しています。公開実験に基づく経験則として、
   繰り返し学習に必要な教師局面数を「モデルparameter数の10倍以上」と見積もっています。
   一般定理ではないため、Meteoでもheld-out arenaとvalidationで実測します。
6. [WCSC36公式アピール文書](https://www.apply.computer-shogi.org/wcsc36/appeal/appeal_round2_260502.pdf)
   には、水匠由来の公開1.5億局面・1億局面データや、別チームの自前100億教師データの
   利用例が記載されています。大会級ではモデル構造だけでなく、データ量と教師の読みの深さが
   勝負です。
7. やねうら王公式の[水匠5 release](https://github.com/yaneurao/YaneuraOu/releases/tag/suisho5)は、
   標準NNUE型`NNUE_halfKP256`、推奨`FV_SCALE=24`の評価関数を無料公開しています。
   [V9.00 GitHub版](https://github.com/yaneurao/YaneuraOu/releases/tag/V9.00)にはApple Silicon用
   実行ファイルがあります。一方、公式Wikiの2026年8月時点の一覧では水匠10/11系は
   支援者向け頒布です。無料方針では水匠5を再現可能な固定baselineとし、水匠11は結果だけを
   外部benchmarkとして扱います。
8. [NAGISA V3.1の公式BOOTHページ](https://booth.pm/ja/items/8639574)では本体価格は0円、
   1,000円は同一棋力のDiscordサポートと説明されています。ユーザーが取得した公式ZIPの
   Apple M1版を実行し、終局対局、10,000 node MultiPV蒸留、SHA-256固定まで完了しました。
   評価関数の無断再配布禁止を守り、重みのコピー・形式変換は行いません。
9. [GNU GPLv3 §2](https://www.gnu.org/licenses/gpl-3.0.html#section2)と
   [GNU GPL FAQ](https://www.gnu.org/licenses/gpl-faq.html#WhatCaseIsOutputGPL)は、通常の実行出力が
   自動的にGPLになるわけではなく、出力自体がcovered workを構成する場合を例外としています。
   Meteoは別規約がないGPLエンジンの指し手、数値評価、PV/MultiPVをoutput-only蒸留へ使いますが、
   元の重み・評価関数・定跡・コードをコピーする許可とは解釈しません。
10. [dlshogi作者の2025年Gated Attention実験](https://tadaoyamaoka.hatenablog.com/entry/2025/12/27/131714)
    は、ResNetの最後の2 blockをTransformer 1 blockへ置き換える20 block x 256 filterモデルを、
    3.88億局面で学習しています。
    [2026年の40 block x 512 filter改良](https://tadaoyamaoka.hatenablog.com/entry/2026/03/21/133131)
    は、Gated Attention、SwiGLU、10 block間隔のTransformer、相対位置biasを扱います。
    大モデルは精度が上がっても探索速度低下で棋力が上がらない場合があるという作者の実測は、
    Meteoがv1を先に鍛え、v2をarenaで昇格させる根拠です。
11. [MLX 0.32.0公式layer一覧](https://ml-explore.github.io/mlx/build/html/python/nn/layers.html)には
    `Conv2d`、`BatchNorm`、`MultiHeadAttention`、`Transformer`があり、
    [function transforms](https://ml-explore.github.io/mlx/build/html/usage/function_transforms.html)は
    `grad`/`value_and_grad`によるautomatic differentiationを正式に提供します。したがって
    dlshogi型ResNetとResNet+TransformerのApple Silicon学習移植は技術的に可能です。
    Meteoでは両方を実装し、forward、backward、checkpoint再読込を実動確認しました。
12. [WCSC36公式決勝](https://www.computer-shogi.org/wcsc36/final.html)では氷彗が5勝2敗で優勝し、
    [公式詳細アピール](https://www.apply.computer-shogi.org/wcsc36/appeal/hisui/hisui_detail.pdf)は
    YaneuraOu系NNUE、特徴・network・LayerStack的分離・量子化・学習pipelineの改良を説明します。
    一方、対応する学習データは非公開で、無料の公式binary/評価関数配布は確認できませんでした。
    よって氷彗はダウンロード済み・蒸留済みとは扱わず、公開棋譜と設計方向だけを独立利用します。
13. [Floodgate公式ページ](https://wdoor.c.u-tokyo.ac.jp/shogi/)は、300秒+1手10秒、毎時2回、
    512手引分、年別棋譜archiveを公開しています。現在のランダム重みMeteoを接続せず、
    時間制御、CSA bridge、1,000局耐久、強敵400局arenaを先に通す方針を
    [`FLOODGATE.md`](FLOODGATE.md)へ分離しました。
14. [AobaNNUE公式README（監査commit）](https://github.com/yssaya/AobaNNUE/blob/8613a0ac911fe35e3fa70aaf012ab6243b7096eb/README.md#評価関数の作成)は、dlshogiの
    勝率→評価値変換係数756を600にしたところ、作者の環境で約`+40 Elo`と報告しています。
    [公式変換コード（監査commit）](https://github.com/yssaya/cshogi_aoba/blob/b092f262bff8f9e4a0a375d6881285c278923995/psv_shuffle/psvs.cpp#L252-L267)は
    `p=sigmoid(cp/756.086...)`を作り、`cp'=600*logit(p)`と再符号化しています。
    Meteoのvalue headは勝率`p`ではなく`[-1,1]`の符号付きvalueなので、Ponanza係数`C`と
    同じ変換は`2p-1=tanh(cp/(2C))`です。従来の`tanh(cp/600)`は`C=300`相当であり、
    AobaNNUEの`C=600`とは異なります。ただし`+40 Elo`はAobaNNUE固有の事前証拠で、
    Meteoでの改善保証ではありません。

## Meteoへの反映

- まず `competition_v1` (20x256, 24,384,760 parameters) を、無料の公開教師と深再解析で
  強い基準器へ育てます。上記10倍経験則の目安は約2.44億局面です。
- `competition_v2` (40x512, 182,930,748 parameters) の同目安は約18.3億局面です。
  構造は強い候補ですが、少量データでの過学習とM4 Proでの教師生成速度を考え、
  v1を上回るheld-out arena結果が出るまで「強いと確認済み」とは扱いません。
- 現在の深再解析、reversal/regret保存、anchor replayは維持します。次にheld-out validation、
  warmup + cosine、教師源ごとのvalue calibration、負けた定跡枝の自動再採掘を追加します。
- 定跡は単純な勝数集計にしません。深教師で逆転勝ちと誤評価を除外し、対局数と不確実性を
  持つゲーム木として管理します。
- 無料のAobaNNUE、公開版水匠、やねうら王系を外部USI教師・arenaに使う候補とします。
  NAGISA V3.1、AobaNNUE v1.1、技巧2 v2.0.2、水匠5は実際に起動し、終局対局と
  MultiPV蒸留を確認しました。版別の利用・再配布判定は[`MODEL_RIGHTS.md`](MODEL_RIGHTS.md)へ
  固定し、未登録モデルはfail closedします。
- PSVは`numpy.memmap`で40-byteレコードを遅延復号し、教師源、offset、strideを指定して
  学習できるようにしました。`game_result`は手番側視点、dropを含むYaneuraOu Move16として
  検証し、不正サイズ・不正結果・非合法手を拒否します。
- candidate/championの色替わりarenaと自己改善世代manifestを実装しました。勝点率だけでなく
  Wilson 95%下限、最小局数、未完局0を同時に満たしたcandidateだけを昇格させます。
- 評価値labelはPonanza係数600、dlshogiの正確な756.086496、教師別fitを同一局面数・
  policy・split・seedで比較します。fitはtrainだけで推定し、独立validationで両固定係数より
  BCEが改善し、game-cluster bootstrap 95%下限も正の場合だけ学習armを作ります。
  policyのCP温度とvalue係数は別hyperparameterとし、このablationでpolicyを変えません。
- 新規USI再解析とPSV学習の既定はPonanza係数`C=600`、内部`tanh`分母`2C=1200`です。
  旧`--teacher-value-scale` / `--score-scale`を明示したrunは、その値を従来どおり`tanh`分母
  として保持する過飽和対照です。新しい係数flagとは相互排他にし、provenanceには`C`と`2C`
  の両方および入力規約を保存します。
- 深い教師との差が大きい局面では、教師policyの混合率を探索比の平方根で縮小し、valueも同じ
  curriculum scaleで弱めます。固定probe lossが非有限または初期値の1.25倍を超えた学習は拒否し、
  勾配はglobal norm 1.0でclipします。実測では技巧教師/Meteoが最大約900倍の局面を含む34 sampleを
  8 step学習し、effective teacher mixを0.200〜0.750へ調整、probe lossを
  `8.796397 -> 8.301497`へ低下させました。
- 自動メモリ予算は64 GiB中、OS/他アプリへ約10.3 GiBを予約し、現在空きからMLXとPython treeへ
  高水位を割り当てます。直近はMLX約13.3 GiB、tree約8.4 GiB、推定約205万nodeでした。
  圧迫時のsubtree回収はOOM回避であり、深い読みを保持する代替ではありません。

## 引き続き必要な競技項目

- PSV教師の重複除去、held-out split、HCPE互換入力
- held-out validation、データ源別メトリクス、warmup + cosine scheduler
- 各教師の評価値と実際の勝率から学ぶvalue calibration
- 定跡ツリー、敗着枝再採掘、深いMultiPVによる末端再検証
- candidate/champion/history/AobaNNUE/公開水匠を数百～数千局で比べるSPRT arena
- compact native search tree、置換表、前局面tree reuse、非同期dynamic batching
- df-pn詰み・必至探索と、300秒+10秒加算に対応するtime manager
- CSA bridgeをローカルshogi-serverで耐久検証した後のFloodgate canary
