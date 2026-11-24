# Verbatim: AI agents for open-ended survey analysis

> Turning thousands of free-text survey answers into coded themes and evidence-backed insights,
> with agents that are evaluated against human coders and built to run on Google Cloud.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![Pydantic AI](https://img.shields.io/badge/Pydantic_AI-E92063?logo=pydantic&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?logo=langchain&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-5E35B1)
![Google ADK](https://img.shields.io/badge/Google_ADK-4285F4?logo=google&logoColor=white)
![Google Cloud](https://img.shields.io/badge/Google_Cloud-4285F4?logo=googlecloud&logoColor=white)
![uv](https://img.shields.io/badge/uv-DE5FE9?logo=uv&logoColor=white)
![Status](https://img.shields.io/badge/status-step_1_of_7-F9A825)

---

## The problem

Almost every customer-experience survey ends with an open question: *"Why did you give that
score?"* Research teams then **code** those verbatims by hand against a codeframe (a fixed list of
themes) before anyone can answer the questions that matter, like *"why are unhappy customers of
service X unhappy?"*.

Hand-coding is slow, expensive and inconsistent between coders. LLMs can do most of it, but a
production system has to answer harder questions than "does the demo work?":

- How closely does the model agree with trained human coders, and on which themes does it fail?
- Respondents can type anything. How do we stop their text from steering the agent?
- How do we keep personal data out of prompts, logs and vector stores?
- What does it cost per 10,000 responses, and how does it scale to a million?
- How do we deploy, monitor and change it without breaking things?

This project builds that system step by step on real survey data.

## What it builds

| # | Component | What it does |
|---|-----------|--------------|
| 1 | **Coding agent** | Assigns codeframe themes and sentiment to each verbatim. Output is typed and validated, and scored against human coders. |
| 2 | **Insights agent** | Answers analyst questions by combining numbers ("what share of negative comments mention waiting times?") with retrieved verbatims as evidence. |
| 3 | **Evaluation harness** | Measures coding agreement, retrieval quality, answer groundedness and agent behaviour, and gates releases on it. |
| 4 | **GCP deployment** | Agent Runtime, BigQuery and Vector Search, with guardrails, tracing and access control. |

---

## The data

Real responses to the **NHS Friends and Family Test (FFT)**, the UK health service's standard
patient-experience survey. It has the same shape as an NPS survey in market research: a closed
rating question followed by an open "why?".

What makes it a good portfolio dataset is that **every answer was coded by trained human coders**
against a 50-theme codeframe. That gives a gold standard to evaluate the LLM coder against, which
most public text datasets do not have.

| | |
|---|---|
| **Source** | [pxtextmining](https://github.com/The-Strategy-Unit/pxtextmining) (The Strategy Unit / NHS England), MIT licence |
| **Organisations** | North East Ambulance Service (NEAS), Northamptonshire Healthcare (NHFT) |
| **Period** | January 2021 to December 2022 |
| **Size** | 7,908 raw answers; 6,205 usable after removing blanks and "Did not answer" |
| **Respondents** | 4,129 with a usable answer; about half answered two questions |
| **Closed question** | FFT rating, 1 = very good … 5 = very poor |
| **Open questions** | "What was good?", "What could we do better?", "FFT Why?", … |
| **Human labels** | Multi-label themes from a [50-code framework](https://the-strategy-unit.github.io/PatientExperience-QDC/framework/framework3.html), plus a 1–5 sentiment score |
| **Segments** | Trust, service (patient transport, 111, mental health, …), date |

A few verbatims, as written by respondents:

> *"Staff were lovely but ambulance door was falling off."*
> *"Maybe a few more checks on the first visit, my dad couldn't even get to the toilet…"*

It is also realistically messy. 80% of answers come from satisfied respondents, the theme
distribution is long-tailed (a handful of themes cover most answers, several have almost no
examples), dates mix UK and US formats, and a few blank answers were still given a theme.

---

## Architecture

```mermaid
%%{init: {'theme': 'default'}}%%
flowchart TB
    subgraph FRAME[" "]
        direction TB
        subgraph SRC["📥 Survey data"]
            direction LR
            RAW[("NHS FFT export<br/>7,908 answers")] ~~~ GOLD[["Human codes<br/>50-theme codeframe"]]
        end

        subgraph S1["① Data foundation · pandas"]
            direction LR
            CLEAN["Clean & normalise"] --> PII["Scrub PII"] --> SPLIT["Split by respondent<br/>train / test"]
        end

        subgraph S2["② Coding · Pydantic AI"]
            direction LR
            CODER["Coding agent<br/>typed output · validators"] --> CODED[("Coded responses<br/>BigQuery")]
        end

        subgraph S3["③ Retrieval · LangChain"]
            direction LR
            EMB["Embed verbatims<br/>+ metadata"] --> VS[("Vector store<br/>Chroma → Vector Search")]
        end

        subgraph S45["④ ⑤ Agents · LangGraph + Google ADK"]
            direction LR
            ROUTER{"Router"} --> QUANT["Quant tool<br/>SQL on coded data"]
            ROUTER --> QUAL["Qual tool<br/>hybrid RAG"]
            QUANT --> WRITER["Insight writer<br/>cites respondent IDs"]
            QUAL --> WRITER
        end

        subgraph S6["⑥ Evaluation"]
            direction LR
            AGREE["Agreement with<br/>human coders"] ~~~ AGEVAL["RAG + agent evals"]
        end

        USER(["🧑‍💼 Analyst"])

        SRC --> S1
        S1 --> S2
        S1 --> S3
        S2 --> S45
        S3 --> S45
        USER -- "question" --> S45
        S45 -- "answer + evidence" --> USER
        S2 -. "scored against<br/>human codes" .-> S6
        S45 -.-> S6
    end

    classDef data fill:#E3F2FD,stroke:#1E88E5,stroke-width:2px,color:#0D1B2A
    classDef pyd fill:#FCE4EC,stroke:#D81B60,stroke-width:2px,color:#0D1B2A
    classDef lc fill:#E8F5E9,stroke:#2E7D32,stroke-width:2px,color:#0D1B2A
    classDef agent fill:#EDE7F6,stroke:#5E35B1,stroke-width:2px,color:#0D1B2A
    classDef eval fill:#FFF8E1,stroke:#F9A825,stroke-width:2px,color:#0D1B2A
    classDef user fill:#ECEFF1,stroke:#455A64,stroke-width:2px,color:#0D1B2A
    class RAW,GOLD,CLEAN,PII,SPLIT data
    class CODER,CODED pyd
    class EMB,VS lc
    class ROUTER,QUANT,QUAL,WRITER agent
    class AGREE,AGEVAL eval
    class USER user
    style SRC fill:#F5FAFF,stroke:#90CAF9,color:#0D1B2A
    style S1 fill:#F5FAFF,stroke:#90CAF9,color:#0D1B2A
    style S2 fill:#FFF5F8,stroke:#F48FB1,color:#0D1B2A
    style S3 fill:#F4FBF4,stroke:#A5D6A7,color:#0D1B2A
    style S45 fill:#F7F4FC,stroke:#B39DDB,color:#0D1B2A
    style S6 fill:#FFFDF2,stroke:#FFE082,color:#0D1B2A
    style FRAME fill:#FFFFFF,stroke:#D0D7DE,stroke-width:1px,color:#FFFFFF
```

### Where each tool fits

Each framework is used where it is strongest, so the trade-offs are visible. In a team codebase I
would standardise on fewer.

| Tool | Role here | Why this tool for this job |
|------|-----------|----------------------------|
| **Pydantic AI** | Coding agent | Type-safe structured output, validators that reject themes outside the codeframe, automatic retries, model-agnostic |
| **LangChain** | Embeddings, vector stores, hybrid retrieval | Mature integrations (Vertex AI, Chroma, BM25); swap the store without rewriting retrieval code |
| **LangGraph** | Analyst workflow: route → retrieve → check → review | Explicit state machine, conditional loops with limits, checkpointing, human-in-the-loop interrupts |
| **Google ADK** | Deployable insights agent | Workflow agents (sequential, parallel, loop), sub-agents, sessions and Memory Bank, eval sets, deploys to Agent Runtime |
| **Google Cloud** | Storage, warehouse, serving | BigQuery, Cloud Storage, Vector Search, Agent Runtime, Cloud Run, IAM |

---

## Agent design (planned)

### Analyst workflow (LangGraph)

```mermaid
%%{init: {'theme': 'default'}}%%
flowchart TD
    subgraph FRAME[" "]
        direction TD
        Q(["❓ Analyst question"]) --> ROUTE{"Route"}
        ROUTE -- "how many? what share?" --> QUANT["Quant<br/>SQL over coded responses"]
        ROUTE -- "why? show examples" --> QUAL["Qual<br/>retrieve verbatims,<br/>filtered by segment"]
        QUANT --> DRAFT["Draft insight<br/>with citations"]
        QUAL --> DRAFT
        DRAFT --> CHECK{"Grounded in<br/>the evidence?"}
        CHECK -- "no · max 2 retries" --> QUAL
        CHECK -- "yes" --> REVIEW["👀 Human review<br/>(optional interrupt)"]
        REVIEW --> A(["✅ Answer + evidence table"])
    end

    classDef io fill:#ECEFF1,stroke:#455A64,stroke-width:2px,color:#0D1B2A
    classDef decide fill:#FFF8E1,stroke:#F9A825,stroke-width:2px,color:#0D1B2A
    classDef step fill:#EDE7F6,stroke:#5E35B1,stroke-width:2px,color:#0D1B2A
    classDef human fill:#E8F5E9,stroke:#2E7D32,stroke-width:2px,color:#0D1B2A
    class Q,A io
    class ROUTE,CHECK decide
    class QUANT,QUAL,DRAFT step
    class REVIEW human
    style FRAME fill:#FFFFFF,stroke:#D0D7DE,stroke-width:1px,color:#FFFFFF
```

### Insights agent (Google ADK)

```mermaid
%%{init: {'theme': 'default'}}%%
flowchart TD
    subgraph FRAME[" "]
        direction TD
        ROOT["🤖 insights_agent<br/>root LlmAgent"]
        ROOT --> CODE["coding_pipeline<br/>SequentialAgent"]
        ROOT --> COMPARE["segment_comparison<br/>ParallelAgent"]
        ROOT --> REPORT["report_writer<br/>LoopAgent"]

        CODE --> G1["guardrail<br/>PII · prompt injection"] --> G2["coder<br/>Pydantic AI tool"] --> G3["validator"]
        COMPARE --> P1["positive-group analyst"]
        COMPARE --> P2["negative-group analyst"]
        REPORT --> R1["drafter"] --> R2["critic"]
        R2 -. "revise" .-> R1
    end

    classDef root fill:#FFF3E0,stroke:#EF6C00,stroke-width:3px,color:#0D1B2A
    classDef workflow fill:#E3F2FD,stroke:#1E88E5,stroke-width:2px,color:#0D1B2A
    classDef leaf fill:#F3E5F5,stroke:#8E24AA,stroke-width:2px,color:#0D1B2A
    classDef guard fill:#FFEBEE,stroke:#C62828,stroke-width:2px,color:#0D1B2A
    class ROOT root
    class CODE,COMPARE,REPORT workflow
    class G2,G3,P1,P2,R1,R2 leaf
    class G1 guard
    style FRAME fill:#FFFFFF,stroke:#D0D7DE,stroke-width:1px,color:#FFFFFF
```

---

## Evaluation

The human codes are split by respondent, so the test set measures performance on people the
system has never seen. Every step ships with its own metric, and step 6 turns them into one
harness that runs in CI.

```mermaid
%%{init: {'theme': 'default'}}%%
flowchart TB
    subgraph FRAME[" "]
        direction TB
        TEST[("Held-out respondents<br/>human-coded")] --> RUN["Run coder / agent"]
        RUN --> M1["Coding quality<br/>micro & macro F1 · Cohen's κ"]
        RUN --> M2["Retrieval quality<br/>recall@k · MRR"]
        RUN --> M3["Answer quality<br/>groundedness · rubric autorater"]
        RUN --> M4["Agent behaviour<br/>ADK evalset trajectories"]
        M1 & M2 & M3 & M4 --> GATE{"Above thresholds?"}
        GATE -- "yes" --> SHIP(["🚀 Release"])
        GATE -- "no" --> FIX(["🔁 Iterate prompt / model / retrieval"])
    end

    classDef data fill:#E3F2FD,stroke:#1E88E5,stroke-width:2px,color:#0D1B2A
    classDef metric fill:#FFF8E1,stroke:#F9A825,stroke-width:2px,color:#0D1B2A
    classDef ok fill:#E8F5E9,stroke:#2E7D32,stroke-width:2px,color:#0D1B2A
    classDef bad fill:#FFEBEE,stroke:#C62828,stroke-width:2px,color:#0D1B2A
    class TEST,RUN data
    class M1,M2,M3,M4,GATE metric
    class SHIP ok
    class FIX bad
    style FRAME fill:#FFFFFF,stroke:#D0D7DE,stroke-width:1px,color:#FFFFFF
```

---

## Built for production

| Concern | How the project handles it | Step |
|---------|----------------------------|------|
| **Untrusted input** | Verbatims are written by the public. They are delimited as data in prompts, screened for injection attempts, and the agent can only call allow-listed tools. Model Armor in production. | 2, 7 |
| **Privacy** | PII is masked before text reaches an LLM, a log or a vector store. Sensitive Data Protection for names in production. | 1, 7 |
| **Evaluation** | Gold human labels, respondent-level holdout, per-theme agreement, ADK eval sets, rubric-based autoraters. | 2–6 |
| **Control** | Closed codeframe, typed outputs with validators, bounded loops, human approval on reports. | 2, 4 |
| **Scalability** | Batch coding with bounded concurrency, retries and caching by text hash; Cloud Run Jobs and BigQuery for volume. | 2, 7 |
| **Cost** | Small fast model for high-volume coding, larger model only for analysis; token usage tracked per run. | 2, 6 |
| **Model choice** | Gemini Flash-Lite vs Flash vs open-weights Gemma, compared on the gold set for quality, latency and cost. | 6 |
| **Memory** | Short-term sessions vs long-term Memory Bank (e.g. an analyst's preferred segments). | 5 |
| **Observability** | OpenTelemetry traces for every model and tool call, exported to Cloud Trace. | 7 |
| **Interoperability** | The insights agent exposed over A2A; data tools exposed over MCP. | 7 |

### Target deployment on Google Cloud

```mermaid
%%{init: {'theme': 'default'}}%%
flowchart LR
    subgraph FRAME[" "]
        direction LR
        USER(["🧑‍💼 Analyst / BI app"]) --> GW["Agent Gateway<br/>auth · policy · Model Armor"]

        subgraph GCP["☁️ Google Cloud project"]
            RT["Agent Runtime<br/>ADK insights agent"]
            MEM[("Sessions +<br/>Memory Bank")]
            LLM["Gemini models<br/>Flash-Lite · Flash"]
            VEC[("Vector Search /<br/>RAG Engine")]
            BQ[("BigQuery<br/>coded responses")]
            GCS[("Cloud Storage<br/>raw survey files")]
            JOB["Cloud Run Job<br/>batch coding"]
            OBS["Cloud Trace<br/>+ Logging"]
            REG["Agent Registry"]
        end

        GW --> RT
        GCS --> JOB --> BQ
        JOB --> LLM
        RT --> MEM
        RT --> LLM
        RT --> VEC
        RT --> BQ
        REG -.-> RT
        RT -. "traces" .-> OBS
    end

    classDef edge fill:#FFEBEE,stroke:#C62828,stroke-width:2px,color:#0D1B2A
    classDef runtime fill:#FFF3E0,stroke:#EF6C00,stroke-width:2px,color:#0D1B2A
    classDef store fill:#E3F2FD,stroke:#1E88E5,stroke-width:2px,color:#0D1B2A
    classDef model fill:#F3E5F5,stroke:#8E24AA,stroke-width:2px,color:#0D1B2A
    classDef ops fill:#ECEFF1,stroke:#455A64,stroke-width:2px,color:#0D1B2A
    class USER,OBS,REG ops
    class GW edge
    class RT,JOB runtime
    class MEM,VEC,BQ,GCS store
    class LLM model
    style GCP fill:#FAFCFF,stroke:#4285F4,stroke-width:2px,color:#0D1B2A
    style FRAME fill:#FFFFFF,stroke:#D0D7DE,stroke-width:1px,color:#FFFFFF
```

| Deployment target | Choose it when |
|-------------------|----------------|
| **Agent Runtime** | It is an ADK agent and you want managed sessions, Memory Bank, scaling and tracing with the least ops work. |
| **Cloud Run** | You need your own API around the agent, a different framework, or scale-to-zero containers you fully control. |
| **GKE** | You already run on Kubernetes, or need GPUs and self-hosted models next to the agent. |

---

## Roadmap

| Step | Topic | Main tools | Status |
|------|-------|-----------|--------|
| 1 | Data foundation: download, clean, PII scrub, respondent-level split | pandas | ✅ Done |
| 2 | Verbatim coding with structured outputs, scored against human coders | Pydantic AI, Gemini | 🔜 Next |
| 3 | Retrieval: embeddings, hybrid search, metadata filters, reranking | LangChain, Chroma, Vertex AI | ⬜ |
| 4 | Analyst workflow with routing, grounding checks and human review | LangGraph | ⬜ |
| 5 | Insights agent with workflow agents, sessions and memory; deploy | Google ADK, Agent Runtime | ⬜ |
| 6 | Evaluation harness and model comparison | Gen AI evaluation service, ADK evalsets | ⬜ |
| 7 | Production hardening: guardrails, tracing, A2A / MCP, CI | Model Armor, Cloud Trace, GitHub Actions | ⬜ |

### Step 1: Data foundation ✅

```mermaid
%%{init: {'theme': 'default'}}%%
flowchart LR
    subgraph FRAME[" "]
        direction LR
        subgraph IN["Ingest"]
            direction TB
            A[("GitHub source<br/>pinned commit")] -- "download +<br/>SHA-256 check" --> B["Raw CSV<br/>7,908 rows · 61 columns"]
        end

        subgraph CL["Clean"]
            direction TB
            C["Decode one-hot codes<br/>→ human_themes list"] --> D["Drop non-answers<br/>blanks, 'Did not answer'"]
            D --> E["Normalise text<br/>+ scrub PII"] --> E2["Parse mixed<br/>UK / US dates"]
        end

        subgraph PR["Prepare"]
            direction TB
            F["FFT score → group<br/>positive · neutral · negative"] --> G["Hash split by respondent<br/>80% train · 20% test"]
            G --> H[("responses.parquet<br/>codeframe.json")]
        end

        IN --> CL --> PR
    end

    classDef src fill:#ECEFF1,stroke:#455A64,stroke-width:2px,color:#0D1B2A
    classDef step fill:#E3F2FD,stroke:#1E88E5,stroke-width:2px,color:#0D1B2A
    classDef out fill:#E8F5E9,stroke:#2E7D32,stroke-width:2px,color:#0D1B2A
    class A src
    class B,C,D,E,E2,F,G step
    class H out
    style IN fill:#FAFAFA,stroke:#B0BEC5,color:#0D1B2A
    style CL fill:#F5FAFF,stroke:#90CAF9,color:#0D1B2A
    style PR fill:#F4FBF4,stroke:#A5D6A7,color:#0D1B2A
    style FRAME fill:#FFFFFF,stroke:#D0D7DE,stroke-width:1px,color:#FFFFFF
```

Decisions worth calling out:

- **Reproducible input.** The source file is pinned to a commit and checked against a SHA-256 hash,
  so a silent upstream change cannot shift the results.
- **Mixed date formats, handled explicitly.** One export used US dates (`9/30/2022`) while the
  rest are UK (`30/09/2021`). Strict parsing surfaced it; the parser resolves it with a documented
  rule instead of letting pandas guess.
- **Split by respondent, not by row.** About half the respondents answered two questions. A
  row-level split could put one of a person's answers in train and the other in test, which
  inflates scores.
- **Hash-based split.** Assignment depends only on the respondent ID, so it stays stable when new
  survey waves are appended.
- **Gold labels are kept apart.** Human themes and sentiment live in `human_*` columns and are
  never sent to a model.
- **PII is masked first.** Everything downstream (prompts, logs, vector store) only sees scrubbed text.

---

## Project structure

```text
pydanticai-rag-langgraph-adk/
├── src/verbatim/
│   ├── config.py        settings from env / .env                   ✅
│   ├── data.py          step 1 · download, clean, split             ✅
│   ├── coding.py        step 2 · Pydantic AI coding agent           🔜
│   ├── retrieval.py     step 3 · LangChain indexing and retrieval   ⬜
│   ├── graph.py         step 4 · LangGraph analyst workflow         ⬜
│   ├── adk_agent/       step 5 · Google ADK agent and deployment    ⬜
│   └── evaluation.py    step 6 · metrics and eval runs              ⬜
├── tests/
│   └── test_data.py                                                 ✅
├── data/                created by step 1, not committed
├── pyproject.toml
└── README.md
```

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+.

```bash
git clone https://github.com/ryan-backer-ds/pydanticai-rag-langgraph-adk.git
cd pydanticai-rag-langgraph-adk
uv sync

uv run python -m verbatim.data   # download and prepare the survey data
uv run pytest                    # run the tests
```

Step 1 writes two files to `data/processed/`:

- `responses.parquet`: one row per usable answer, with segments, FFT score, scrubbed text, human
  labels and a `split` column
- `codeframe.json`: the 49 codeable themes in framework order, with how often coders used each

## Data licence and attribution

The survey data is not stored in this repository. `verbatim.data` downloads it from
[The-Strategy-Unit/pxtextmining](https://github.com/The-Strategy-Unit/pxtextmining) (MIT licence),
pinned to commit `1f3ae76`. It contains Friends and Family Test feedback shared by North East
Ambulance Service and Northamptonshire Healthcare NHS Foundation Trust, labelled by the
Patient Experience Qualitative Data Categorisation project. Credit to The Strategy Unit and
NHS England for making it public.
