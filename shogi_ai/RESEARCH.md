# Meteo 将棋AI調査メモ

更新: 2026-08-11

このメモは、「めてお / Meteo」を公開可能な系統と正規承認済みlocal-only系統に分けて
大会級将棋AIへ育てるために、
たややん氏／水匠の公開情報、やねうら王の公式資料、WCSCアピール文書から
実務上の要点をまとめたものです。他エンジンの主張とMeteoの実測は分けて扱います。

## 確認した公開情報

1. たややん氏の[2026年最新の将棋AI事情](https://book.mynavi.jp/shogi/detail/id%3D149624)は、
   NNUEはCPUで高速に広く読み、DL+MCTSは推論が重い代わりに局面評価が高精度と
   整理しています。2026年1月時点の無償最強候補として `AobaNNUE` を明記しているため、
   公開可能なMeteo系統では教師・arena対手の第一候補にします。これは著者の時点付き見解であり、
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
   支援者向け頒布です。水匠5は公開再現可能な固定baselineです。一方、ユーザーが正規入手した
   水匠11Plusのexact profileは、元artifactとraw labelsを非公開に保つlocal-only USI教師として
   解析・蒸留・学習へ使用します。派生Meteo checkpointの公開は別の権利gateで拒否します。
8. [NAGISA V3.1の公式GitHub release](https://github.com/keinoda/YaneuraOu/releases/tag/nagisa-v3.1)は、
   YaneuraOu 9.60、`HalfKA_hm2 1024x16x64 / LayerStack 9`、`FV_SCALE=28`、
   `progress8kpabs`、source commitを明示し、v3.1は探索parameterのSPSA再調整だけで評価関数は
   v3から不変と説明しています。[公式BOOTHページ](https://booth.pm/ja/items/8639574)では本体価格は0円、
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
15. [AlphaZero論文](https://arxiv.org/abs/1712.01815)は将棋を対象に含み、現在盤だけでなく
    過去局面を含む時系列表現を入力に使います。これは千日手や到達経路を現在SFENだけへ潰さない
    `history-input-v2`の一次根拠です。[MuZero Reanalyse](https://arxiv.org/abs/2104.06294)は、
    保存済み経験へ新しい探索targetを付け直してsample efficiencyを高める方法を提案しています。
    ただし、どちらもMeteoでの棋力向上を保証するものではないため、同一wall-clock arenaで検証します。
16. [Confidence-Aware Multi-Teacher Knowledge Distillation](https://arxiv.org/abs/2201.00007)は、
    複数教師の固定平均が低品質な教師予測によりstudentを誤誘導し得ることを示し、同論文の実験でも
    単純平均が精度を落としたと報告しています。ただし同手法はground-truth labelで教師信頼度を
    測るため、正解不明の将棋局面へそのまま移植できません。Meteoではこの結果を「平均を避ける」
    根拠に限定し、独立head、全候補の相互再採点、内部証明、arenaを代替の検証境界にします。
17. [Gumbel AlphaZeroのICLR 2022論文](https://openreview.net/forum?id=bERaNdoegnO)は、rootの
    全手を訪問できない少数simulationで従来AlphaZeroのpolicy improvementが保証されない問題を
    扱い、少数simulationでの改善を報告しています。これはMeteoの探索高速化ablation候補ですが、
    canonical教師targetの正しさを保証するものではなく、現PUCTとの同一wall-clock対局を通すまで
    productionへ置換しません。
18. やねうら王公式の[finny tables実装記事](https://yaneuraou.yaneu.com/2026/08/11/finny-tables-implemented-in-yaneuraou/)
    は、玉位置ごとのfeature-transformer accumulatorをcacheして、玉移動時の全特徴再計算を
    差分へ置き換える手法を説明しています。記事掲載値はNNUE推論部30〜50%程度、NPS約15%の
    高速化です。これはNNUE教師のラベル生成速度を上げ得ますが、MeteoのDL networkへ直接移植する
    手法ではありません。またNAGISA、水匠11Plus、奏乗は各配布元の探索・build条件も棋力の一部なので、
    最新やねうら王へ差し替えた版は別engine profileとして同条件arenaで比較してから教師へ採用します。
19. たややん氏の[HiraganaSuisho](https://github.com/tayayan/HiraganaSuisho)と
    [230906 release](https://github.com/tayayan/HiraganaSuisho/releases/tag/230906)を、tag commit
    `2bfb018215ddf91d949f1f025ae94a11e0216219`で監査しました。これはNNUE学習器ではなく、
    USIエンジンの連続対局から定跡木を広げるMITライセンスの生成器です。frontier探索、ランダムな
    枝被覆、終局時だけのtransactional merge、訪問回数、定期snapshotは局面採掘へ再利用できます。
    一方、同tagの反復判定は過去の同一盤面出現を調べる簡略形で、投了結果を枝へ直接伝播するheuristicも
    canonicalな最善手証拠には不足します。生成局面だけを取り込み、現行ルールでreplay検証後、三教師が
    定跡なしで再解析します。
20. [On-Policy Distillationの2026年研究](https://arxiv.org/abs/2604.13016)は、studentが実際に訪れた
    状態の高確率出力へ教師信号を返すこと、失敗時のoff-policy cold start、studentとteacherの
    思考分布の互換性、新能力を持つ教師の必要性を報告しています。
    [OPD survey](https://arxiv.org/abs/2604.00626)もstudent-sampled trajectoryへの反復feedbackを
    exposure bias対策として整理します。将棋への対応は、Meteo自己対局の実到達局面を三教師で
    再解析するonline distillationです。ただし初期random modelだけの自己対局は局面分布が狭いため、
    三教師相互対局、全登録対手、定跡木、詰み・千日手anchorをoff-policy seedとして併用します。
21. [PRO](https://arxiv.org/abs/2306.17492)は複数候補の順位を一対比較へ潰さず学習するranking loss、
    [PLaD](https://arxiv.org/abs/2406.02886)はteacher/student出力からpseudo-preference pairを作る
    蒸留を提案しています。Meteoでは順位だけのPROを主損失にはしません。三教師のraw CP/mateを
    校正したsoft distributionは、最善・次善・それ以下の差の大きさまで保持できるため、教師別CEと
    最悪regret分布CEを主にします。PRO型の順序lossは同点・mate・boundを正しく扱う補助ablation、
    終局WDLは全手を指し切った後の低比重auxiliaryとします。人間選好のRLHFではなく、三教師による
    RLAIF/online distillationと探索policy improvementの組合せです。
22. [dlshogi公式network実装](https://github.com/TadaoYamaoka/DeepLearningShogi/blob/master/dlshogi/network/policy_value_network_resnet.py)は、
    同じ局面表現から全指し手のlogitを出すpolicy headと、1個の期待勝率を出すvalue headを持ちます。
    [公式train実装](https://github.com/TadaoYamaoka/DeepLearningShogi/blob/master/dlshogi/train.py)はpolicyへ
    soft-target cross entropy、valueへ終局結果と探索評価値のBCEを使います。
    [公式HCPE/HCPE3 decoder](https://github.com/TadaoYamaoka/DeepLearningShogi/blob/master/cppshogi/python_module.cpp)は、
    旧HCPEでは教師最善手をone-hot policyにし、HCPE3では各候補手の探索訪問回数を
    `visit^(1/temperature)`で正規化したsoft policy（temperature 0なら最多訪問手のone-hot）へ
    変換します。したがって「分布を学ぶか」は教師データ形式次第ですが、HCPE3は明確に
    最善・次善以下を含む探索後の訪問分布を学習します。
    [ふかうら王の公式UCT実装](https://github.com/yaneurao/YaneuraOu/blob/master/source/engine/dlshogi-engine/UctSearch.cpp)では、
    policy確率をp-UCBのprior、valueを未探索leafの期待勝率に使い、訪問後は探索で蓄積した勝率へ
    置き換えます。したがってDLが直接覚える主対象は「探索コード」ではなく指し手分布と局面価値で、
    良いpriorとleaf評価によって同じsimulation数で有望枝へ探索を集中できることが探索効率化です。
    一方、[やねうら王公式NNUE architecture](https://github.com/yaneurao/YaneuraOu/blob/master/source/eval/nnue/architectures/README.md)は
    最終出力が1値の局面評価関数です。NAGISA、水匠11Plus、奏乗のNNUE/SFNNはpolicy分布を直接出さず、
    指し手生成・枝刈り・反復深化はαβ探索側が担います。

## 配布物に残る学習メモの監査

- NAGISA V3.1のZIPには実行ファイル、`eval/nn.bin`、`progress.bin`、`eval_options.txt`、連絡先だけが
  あり、学習データ、optimizer、epochの説明はありません。headerからSFNN HalfKA_hm2 1024・
  LayerStack 9、optionsから`progress8kpabs`と外部`progress.bin`の指定だけを確認できます。
  `FV_SCALE`や学習時score scaleは同梱optionsにないため、それ以上を推測しません。
- 水匠11Plusの配布archiveは`nn.bin`と短いengine optionsが中心で、学習手順メモはありません。
  architecture headerは構造識別であって、教師データや学習率の証拠ではありません。
- 奏乗TSEC7のZIPは`engine_options.txt`、`eval/nn.bin`、`progress.bin`、Windows実行ファイルで、
  学習手順メモや独自ライセンス文書は見つかりません。確認できるのはHalfKaHmMerged 2048x2、
  LayerStack 9、`FV_SCALE=28`、`progress8kpabs`等の実行条件までです。
- AobaNNUE v1.1の同梱`aobannue.txt`には最も詳しい記録があります。`shogi_hao_depth9`の80億局面を
  WCSC35 AobaZeroの0手読み評価で上書きし、静止探索で局面を書き換え、epoch-size 1,000万を
  32,000 epoch（延べ3,200億局面提示）、minibatch 8192、RTX 4090で15日、27,000 epoch以降
  2,000 stepごとに学習率半減、label smoothing 0.001、momentum 0.9と記録されています。
  `HalfKP_768_x2_16_64`が同時間比較で最良だったという作者の実測もあり、Meteoではデータ再評価、
  大量反復、schedule、architecture-vs-NPS ablationの参考にします。
- 技巧2の同梱READMEと`learning.cc`には、進行度→評価関数→指し手実現確率→定跡の順と、
  `generate-positions`＋RootStrapを反復する強化学習が残っています。さらに自己対局の終局勝敗を使う
  logistic regressionをRootStrapへ足す方が若干強かったと作者が記録しています。Meteoではこれを
  「手ごとの深い三教師分布を主信号、終局WDLを補助信号」に対応させます。
- 振電3、tanuki DR4、Háo、水匠5の手元runtimeからは、版・architecture・options以外の
  十分な学習履歴は確認できませんでした。binary中の偶然の文字列を学習metadataとは扱いません。

## Meteoへの反映

- まず `competition_v1` (20x256, 24,384,760 parameters) を、公開教師、正規承認済みlocal教師、
  深再解析で
  強い基準器へ育てます。上記10倍経験則の目安は約2.44億局面です。
- `competition_v2` (40x512, 182,930,748 parameters) の同目安は約18.3億局面です。
  構造は強い候補ですが、少量データでの過学習とM4 Proでの教師生成速度を考え、
  v1を上回るheld-out arena結果が出るまで「強いと確認済み」とは扱いません。
- 現在の深再解析、reversal/regret保存、anchor replayは維持します。次にheld-out validation、
  warmup + cosine、教師源ごとのvalue calibration、負けた定跡枝の自動再採掘を追加します。
- 定跡は単純な勝数集計にしません。深教師で逆転勝ちと誤評価を除外し、対局数と不確実性を
  持つゲーム木として管理します。
- 現行9モデル（奏乗TSEC7、NAGISA V3.1、水匠11Plus、AobaNNUE v1.1、tanuki- Lí-VENGE、
  水匠5、振電3、技巧2 v2.0.2、Háo）を外部USI教師・arenaに使います。全9件で実際の起動と
  終局対局を確認しました。版別の利用・再配布判定は
  [`MODEL_RIGHTS.md`](MODEL_RIGHTS.md)へ固定し、未登録モデルはfail closedします。
- canonical v1の単一valueへの全教師broadcastは算術平均へ収束するため学習禁止にしました。
  v2はNAGISA／水匠11Plus／奏乗TSEC7のpolicy/valueを三つの独立headで監視し、候補和集合・
  複数budget・三者が提案したprincipal reply和集合を全三教師が
  再採点したときだけ、最悪教師regretの大域的argminをplay targetにします。教師が解消不能に
  対立する局面はplayを学習せず追加解析へ戻します。最善手をsoftmax温度で意図せず薄めません。
  ただし実データ用score-matrix builder、校正artifactの再計算receipt、独立held-out splitは
  未実装です。内部整合する手書きsidecarは実学習の根拠にならないため、canonical v2のtrain CLIは
  これらのreceiptを実装するまでfail closedとし、synthetic fixtureの勾配検証だけを許します。
- 上記canonical v2のworst-regret合成は、現在はRound Dの対照群です。主経路は三教師が
  候補を提案し、1教師だけが全候補と応手を同条件で再採点するmulti-proposer/single-scorerに
  更新しました。奏乗は暫定scorerであり、A-N/A-W/A-SとB-N/B-W/B-SでNAGISA・
  水匠11Plus・奏乗の全員をscorer候補に残します。Round DがA/Bをheld-out regret、
  fixed-node/time、worst-group、arenaで上回るまで、合成値をproduction正解にしません。
- Meteoの実戦推論で現在使う出力はplay policyと符号付きplay valueの2つです。policyはMCTSの
  PUCT prior、valueはleaf評価となり、探索後のroot訪問回数分布が実際の手選択になります。
  canonical v2はこれに三教師別policy/value、WDL、教師不一致uncertaintyを学習用headとして加えますが、
  三教師別headは教師同士の戦略差を失わない監視用、WDL/uncertaintyは補助信号であり、現行
  `MLXEvaluator`は実戦時にそれらを直接読みません。play policyには三教師の最悪regret全分布、
  play valueには再解析後の区間、WDLには終局結果を入れる設計です。2026-08-11時点のclean-reset
  checkpointはstep 0のrandom初期値で、実score-matrix builder receiptが未実装なため学習processは
  再開していません。つまり設計済みのtargetと、実際に学習済みの重みを区別します。
- PSVは`numpy.memmap`で40-byteレコードを遅延復号し、教師源、offset、strideを指定して
  学習できるようにしました。`game_result`は手番側視点、dropを含むYaneuraOu Move16として
  検証し、不正サイズ・不正結果・非合法手を拒否します。
- 奏乗系公開unique corpusの抽出100 recordは全て`Move16=0`でした。現行のpolicy教師
  loaderはvalue-only PSVから合法手を捜造せず明示拒否します。利用許諾とsample別の
  policy-loss maskを備えたvalue専用経路を実装するまで学習に投入しません。系譜、
  正確なrecord数、再利用判定は[`TEACHER_LINEAGE.md`](TEACHER_LINEAGE.md)に固定しました。
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

- canonical score-matrix builder、全候補`searchmoves`再採点、教師/局面phase別CP校正、proof certificate、独立held-out split
- PSV教師の重複除去、held-out split、HCPE互換入力
- held-out validation、データ源別メトリクス、warmup + cosine scheduler
- 各教師の評価値と実際の勝率から学ぶvalue calibration
- 定跡ツリー、敗着枝再採掘、深いMultiPVによる末端再検証
- candidate/champion/history/AobaNNUE/公開水匠を数百～数千局で比べるSPRT arena
- compact native search tree、置換表、前局面tree reuse、非同期dynamic batching
- df-pn詰み・必至探索と、300秒+10秒加算に対応するtime manager
- CSA bridgeをローカルshogi-serverで耐久検証した後のFloodgate canary
