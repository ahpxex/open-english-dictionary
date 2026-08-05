# Open Dictionary

面向语言学习者的开放英语词典,以及生产它的可复现数据管线。

词典的**定义语言是参数化的**:同一条管线可以为任何目标语言生成学习者释义,
中文(简体)是第一个发布的定义语言,也是当前的默认值;其他语言只需切换
`--definition-language-*` 参数即可构建(内容契约、prompt 版本与校验随语言联动)。

当前发布版本 **[v2.0](https://github.com/ahpxex/open-dictionary/releases/tag/v2.0)**
(简体中文定义语言构建):**84,212 个词条**,从 Wiktionary/Wiktextract 快照
出发,经词频筛选、结构化 LLM 生成与契约校验后产出,全链路带运行血缘,可从
源数据完整重建。

## 下载

[Releases 页面](https://github.com/ahpxex/open-dictionary/releases) 提供三种工件
(附 SHA256 校验和):

| 工件 | 内容 |
|---|---|
| `distribution.jsonl.gz` | 每行一个词条,`distribution_entry_v5` 契约 |
| `distribution.sqlite.gz` | 同内容的 SQLite 打包,义项带优先级索引列,客户端可直接查询 |
| `audit.jsonl.gz` | 审计工件:保留 curated 层与生成层的完整分离视图 |

## 词条长什么样

每个词条包含:一条贯穿主要义项的**记忆主线**、三级**义项优先级**
(core/common/rare,客户端可默认折叠生僻义)、核心与常用义项的**双语例句**
(英文原句 + 定义语言翻译)、句式导向的**用法说明**,以及词形变化、US/UK
音标、词源注记和关系词。所有解释性字段均以目标定义语言写成,下面的示例来自
简体中文构建:

```json
{
  "headword": "knot",
  "memory_hook": "想象一根绳子打成一个结：它把东西缠在一起，于是头发会打结、事情会变得棘手……而航海速度单位则来自测量绳上一个个结的传统。",
  "pos_groups": [
    {
      "pos": "noun",
      "summary": "名词：表示绳结、纠结物、凸起、难题或连接，也可表示航海和航空中的速度单位。",
      "meanings": [
        {
          "sense_id": "s1",
          "priority": "core",
          "short_gloss": "绳结",
          "learner_explanation": "绳子、线或其他细长柔软材料绕成的结，通常不能直接拉开，必须让一端穿过圈才能解开。",
          "examples": [
            {
              "text": "Tie a knot in the rope so it will not come loose.",
              "translation": "在绳子上打个结，这样它就不会松开。"
            }
          ]
        }
      ]
    }
  ]
}
```

优先级是诚实标注:整词皆生僻的词条确实存在,按优先级过滤的客户端在过滤结果为空时
必须回退显示全部义项(契约要求,见 `docs/export_contracts.md`)。

## 管线架构

```text
数据层:      raw → curated → llm → exports,meta 层贯穿记录运行血缘

init-db               显式迁移建立全部 schema
ingest-snapshot       下载/注册 Wiktextract 快照,COPY 流式入库(可断点续传)
assemble-entries      词条组装:一个头词一个词条,词性组/词源组/义项骨架,
                      word_selection_v2 词频筛选(wordfreq top-40k,短语 zipf ≥ 4.5)
generate-definitions  LLM 生成:骨架严格对齐校验、大词条分片生成(概览提名 core
                      + 分块生成 + 确定性组装)、provider 池负载均衡、温度阶梯重试
export-distribution / export-distribution-sqlite / export-audit
                      契约校验后的分发与审计工件,支持 prompt 版本优先列表
audit-definitions     启发式质检:每千行健康桶 + 随机样本内容检查(只读)
review-definitions    LLM 裁判质检:分层确定性抽样 + 对照源骨架的五维评审,
                      结果落库、同种子可复现可续查
```

关键机制:

- **断点续跑**:生成按(prompt 版本、定义语言、输入内容哈希、模型集合)判定完成,
  重复运行零重复开销;curated 变更后只有受影响词条会重新生成
- **provider 池**:`LLM_PROVIDERS` JSON 数组配置多个 OpenAI 兼容端点,自动负载
  均衡与故障冷却,每行记录实际服务方与 token 用量
- **版本优先导出**:prompt 迭代后只需定向重生成受影响词条,导出时按版本优先序
  逐词条择优(如 v2.0 由 v14 覆盖 7,678 条、v13 承担其余),无需全量重跑

## 从零复现

```bash
uv sync
# .env 需要 DATABASE_URL 和 LLM_PROVIDERS(见下)
uv run opend init-db
uv run opend ingest-snapshot --workdir data/raw          # 或 --archive-path 本地快照
uv run opend assemble-entries --lang-codes en --top-words 40000 --phrase-min-zipf 4.5
uv run opend generate-definitions --max-workers 40
uv run opend export-distribution --output data/export/distribution.jsonl
uv run opend validate-distribution --input data/export/distribution.jsonl
uv run opend export-distribution-sqlite --output data/export/distribution.sqlite
```

### LLM Provider 配置

生成经由 litellm 走 OpenAI 兼容 provider 池,`LLM_PROVIDERS` 是一个 JSON 数组,
加 provider 就是加一个对象:

```dotenv
LLM_PROVIDERS='[
  {"api": "https://api.provider-one.example/v1", "model": "some/model-name", "key": "sk-...", "rpm": 120},
  {"api": "https://api.provider-two.example/v1", "model": "another/model-name", "key": "sk-..."}
]'
```

- `api`/`model`/`key` 必填,`rpm` 可选(填了按限速加权路由并做派发前容量预检);
  字段缺失或未知字段在启动时报错
- 撞限或故障的 provider 自动冷却,请求切到池内其他成员重试
- 兼容旧式单 provider 变量 `LLM_API`/`LLM_KEY`/`LLM_MODEL`(`LLM_PROVIDERS`
  存在时优先)
- 断点续跑语义:同 prompt 版本、同定义语言、同输入哈希下,池内任一模型已成功
  即视为完成;换池只会重跑成功行来自已移除模型的词条

### 定义语言

定义语言是管线的一等参数,默认 `zh-Hans`(简体中文)。构建其他语言版本:

```bash
uv run opend generate-definitions \
  --definition-language-code fr --definition-language-name French
uv run opend export-distribution \
  --definition-language-code fr --definition-language-name French \
  --output data/export/en-headwords-fr-definitions.jsonl
```

生成 prompt、输出校验、断点续跑与导出匹配都按(prompt 版本 × 定义语言)隔离,
多个语言版本可在同一数据库中并存互不干扰;prompt 版本自动携带语言后缀
(如 `curated_v1_distribution_fields_v14__deflang__fr`)。

## 质量体系

三层:

1. **生成时硬校验**(逐行阻断):骨架对齐(词性组/义项 id 不增不减不改名)、
   优先级枚举、双语例句结构、坏字符拒收,不过即重试,重试穿不过记失败行
2. **启发式审计**(`audit-definitions`):按 enrichment_id 千行分桶的健康指标 +
   随机样本的内容启发式(core 预算、例句覆盖、复读式用法说明、中英混杂)
3. **LLM 裁判评审**(`review-definitions`):按(词条类型 × 规模档)分层抽样,
   风险层保底、逐行记录抽样权重(全局指标无偏回加权);裁判对照 curated 源
   骨架评五维分并输出类型化问题清单;同种子可复现、内容变化自动重审

## 常用命令速查

```bash
uv run opend run --archive-path <快照> --top-words 40000 --phrase-min-zipf 4.5 \
  --worker-tiers 50 16 4 --distribution-output data/export/distribution.jsonl \
  --distribution-sqlite-output data/export/distribution.sqlite --validate-distribution
                                          # 全流程一条命令(仍由阶段契约组合)

uv run opend generate-definitions --limit-entries 50        # 小样本试跑
uv run opend generate-definitions --recompute-existing      # 显式重算
uv run opend export-distribution \
  --prompt-version curated_v1_distribution_fields_v14 \
  --prompt-version curated_v1_distribution_fields_v13       # 版本优先序导出
uv run opend audit-definitions --sample-size 300            # 启发式体检
uv run opend review-definitions --sample-size 1000 --seed qc-2026-08-04
                                          # LLM 裁判评审(同种子可复现)
uv run opend export-audit --output data/export/audit.jsonl --include-unenriched
```

所有命令 stdout 输出一个结构化 JSON 结果,stderr 输出进度事件;`--env-file`
默认读取当前目录 `.env`,LLM 命令的 `--model-env-file` 缺省回落到同一文件。

## 文档

- `docs/curation_rulebook.md` — 策展规则手册(词性策略、选词规则、字段取舍,
  含每条用户批准规则的日期)
- `docs/export_contracts.md` — 分发与审计工件契约(distribution_entry_v5、
  SQLite 打包 schema、客户端展示规则)
- `docs/system_framing.md` — 系统定位与分层框架
- `CLAUDE.md` — 管线章程:分阶段契约、可复现性铁律、数据库与测试要求

## 许可

- **代码**:MIT(见 `LICENSE`)
- **词典数据工件**:CC BY-SA 4.0(见 `LICENSE-DATA.md`)——数据是 Wiktionary
  内容的衍生作品,依 ShareAlike 条款必须以相同许可发布并署名 Wiktionary 贡献者;
  再分发或二次加工数据时同样受此约束

## 可复现性

每次运行都有持久 `run_id` 并记录上游依赖运行;prompt 全文入库版本化;工件元数据
记录源快照哈希、选词规则参数、prompt 版本列表与逐词条的生成审计(实际服务
provider、token 用量、尝试次数、温度)。给定同一快照与同一配置,整条管线可重建
出内容一致的词典。
