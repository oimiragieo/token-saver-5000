# gotcontext.ai — Go-To-Market Plan

**Product**: gotcontext.ai — The context optimization and AI intelligence platform
**Tagline**: "Every token your AI reads costs money. We make sure it only reads what matters."
**Founder**: CEO (business), Claude Code (engineering)
**Status**: Product built, 126 MCP tools, proven benchmarks, ready for launch

**Five product lines:**
1. **Context as a Service (CaaS)** — semantic compression SaaS (freemium + paid)
2. **AI Benchmark Repository** — crowd-sourced global AI inference database (UserBenchmark for AI)
3. **AI News Center** — curated AI industry intelligence hub
4. **Knowledge Hub** — model-agnostic RAG with compressed retrieval (NotebookLM competitor)
5. **Agent Context Hub** — framework doc indexing with semantic compression (ref.tools/Context7 competitor)

---

## 1. Market Opportunity

### Total Addressable Market (TAM) — Combined Platform

| Market Segment | 2026 Size | 2030 Projected | CAGR | Source |
|----------------|-----------|----------------|------|--------|
| **Global AI market** | **$1.02T** | **$1.8T (2027)** | 89% | AI Business Insider |
| **AI inference spend** (55%+ of AI cloud) | **$480B+** | >$1T | ~40% | Oplexa / Deloitte |
| **AI developer tools** | **$3.5B** (DevOps alone) | $10B+ | 21.5% | WifiTalents |
| **AI benchmarking platforms** | **$1.2B** | $4.5B (2033) | 17.8% | FactMR / MarketIntelo |
| **Enterprise AI budget (avg)** | **$7M/yr** | $15M+ | 483% since 2024 | AnalyticsWeek |
| **AI VC funding pace** | **$120B+** | — | — | AI Business Insider |
| **MCP protocol installs** | **97M** | — | — | David & Goliath |

#### gotcontext.ai TAM by Product Line

| Product Line | Addressable Market | Basis |
|--------------|-------------------|-------|
| **CaaS (Context as a Service)** | **$50-100B** | 5-10% of $480B+ inference spend is wasted context |
| **AI Benchmark Repository** | **$1.2-4.5B** | AI benchmarking platform market (FactMR 2026-2033) |
| **AI News Center** | **$500M-1B** | AI-focused dev media (ad + sponsorship + premium content) |
| **Knowledge Hub** | **$2-5B** | Enterprise knowledge management + RAG SaaS (Grand View Research 2026) |
| **Agent Context Hub** | **$1-3B** | Developer documentation tooling + AI agent infrastructure |
| **Combined platform** | **$55-114B** | — |

### The Macro Thesis

The AI inference cost crisis is real and accelerating:

- Per-token costs fell **280x** in two years ($30/MTok → $0.10/MTok), but enterprise AI bills rose **320%** (Oplexa 2026)
- Average enterprise AI budget grew from **$1.2M/yr (2024) → $7M/yr (2026)** — 483% increase
- **73% of enterprises** report AI costs exceeding original budget projections (FinOps Foundation 2026)
- Inference now consumes **55% of AI cloud spend** ($37.5B), projected to reach **70-80% by end of 2026** (ByteIota)
- Agentic workflows use **10-20x more tokens** than simple queries — and agents run 24/7
- **92% of US developers** use AI coding tools daily (WifiTalents 2026)

**The paradox**: tokens are cheaper, but total bills are higher. Volume is growing faster than prices are falling. This is the exact environment where optimization tools thrive — not because tokens are expensive per unit, but because there are so many of them.

### The Pain Point

A developer using Claude Code with 5 MCP servers burns ~100K tokens per request just on
tool schemas + CLI output noise. At Opus pricing ($15/MTok input), that's **$1.50 per request**
in waste. A team of 10 developers making 50 requests/day = **$750/day wasted** = **$195K/year**.

gotcontext.ai cuts this by 55-96%, saving **$107-187K/year** per team.

### Why Now — Three Convergent Tailwinds

1. **Inference cost crisis** — enterprises are spending more despite falling unit prices. Optimization is urgent.
2. **MCP is the standard** — 97M installs. Every major AI agent supports it. gotcontext plugs in natively.
3. **No one owns the benchmark data** — model perf data is scattered across Reddit, Twitter, Discord. There's no central, structured, queryable source. First mover wins the data network effect.

---

## 2. Competitive Positioning

### Direct Comparison

| Feature | gotcontext.ai | RTK (28K★) | Headroom | Atlassian mcp-compressor | Context-Mode |
|---------|--------------|-----------|----------|------------------------|-------------|
| Document compression | **13x** | No | "up to 80%" | No | No |
| CLI output filtering | **11 strategies** | 100+ commands | No | No | No |
| Tool schema compression | **96% reduction** | No | No | 70-97% | No |
| MCP proxy (any server) | **Yes** | No | Yes | Yes | No |
| Session continuity | **Yes** | No | No | No | Yes |
| Code search (AST + trigram) | **Yes** (tensor-grep) | No | No | No | No |
| Proven benchmarks | **Claude/Codex/Gemini** | Claimed 89% | No published | No published | No published |
| Quality predictor | **Yes** (PoC paper) | No | No | No | No |
| Research-backed | **3 papers** | No | No | No | No |
| Pricing | **Freemium + usage** | Free (OSS) | Free (OSS) | Free (OSS) | Elastic License |
| AI benchmark repo | **Yes (planned)** | No | No | No | No |

### Context/RAG Comparison (Knowledge Hub competitors)

| Feature | gotcontext Knowledge Hub | Google NotebookLM | open-notebook (21.8K★) | SurfSense (13.6K★) |
|---------|------------------------|-------------------|----------------------|-------------------|
| Model-agnostic | **Yes (16+ providers)** | Google only | Yes (16+) | Yes |
| Self-hosted | **Yes** | No (cloud only) | Yes (Docker) | Yes (Docker) |
| Semantic compression | **Yes (5-20x)** | No | No | No |
| Multi-format ingest | PDF, code, video, audio, web | PDF, video, web | PDF, video, audio, web | Search, Slack, Linear, Notion, GitHub, YouTube |
| API access | **REST + MCP** | No API | REST API | REST API |
| Multi-tenant | **Yes (scoped)** | Per-user (Google) | Single-user | Team support |
| Compressed retrieval | **Yes (unique)** | No | No | No |
| Podcast generation | Planned | Yes (2 speakers) | Yes (1-4 speakers) | No |
| Pricing | Freemium | Free (Google lock-in) | Free (OSS) | Free + SaaS |

### Doc Context Comparison (Agent Context Hub competitors)

| Feature | gotcontext Agent Context Hub | Context7 (51K installs) | ref.tools | docs-mcp-server (1.2K★) | Nia ($6.2M funded) |
|---------|----------------------------|------------------------|-----------|------------------------|--------------------|
| Compressed retrieval | **Yes (unique)** | No (raw chunks) | ~5K token cap | No | No |
| Offline/local | **Yes** | No (cloud) | No (cloud) | Yes | No |
| Open source | **Yes** | No | Partial | Yes (MIT) | No |
| Version-specific docs | Yes | Yes | Yes | Yes | Yes |
| Self-hosted | **Yes** | No | No | Yes | No |
| Multi-format | 90+ formats | Web docs only | Web docs only | 90+ formats | Codebase + docs |
| Token efficiency | **85-96% compression** | Uncompressed | ~5K cap | Uncompressed | Uncompressed |
| Pricing | Freemium | $10/mo (1K free) | $9/mo | Free (OSS) | $14.99/mo |

### Our Moat

1. **Only tool with proven cross-platform benchmarks** (real API numbers, reproducible)
2. **Deepest compression stack** (semantic graph + token refiner + meta-tokens + CLI filters)
3. **Broadest feature surface** (121 tools vs competitors' 1-5)
4. **Research-backed** (3 arXiv papers implemented, TurboQuant-inspired embeddings)
5. **Full stack**: search (tensor-grep) + compress (token-saver) + filter (CLI optimizer) + proxy
6. **Data network effect** (benchmark repo): crowd-sourced inference data that can't be forked or replicated — grows more valuable with every submission
7. **Compressed retrieval** (unique in market): no doc context tool or RAG platform compresses output before serving — gotcontext is the only platform where retrieval and compression are unified

---

## 3. Pricing Strategy — Five Product Lines

### Product Line 1: Context as a Service (CaaS)

Hybrid model (free tier + usage-based). 67% of B2B SaaS uses hybrid pricing in 2026.

| Tier | Price | CaaS Includes | Target |
|------|-------|---------------|--------|
| **Free (open-source)** | $0 | Local MCP tool, TF-IDF embeddings, basic compression (50-70% savings), community support | Solo devs, OSS users |
| **Pro** | $29/month | Cloud API, SBERT embeddings, full compression (85-96% savings), all 121 tools, proxy mode, CLI optimizer, 5M tokens/month | Individual developers |
| **Team** | $19/user/month (min 5) | Everything in Pro + team dashboard, session history, shared config, 50M tokens/month shared | Dev teams |
| **Enterprise** | Custom | Unlimited, self-hosted option, SSO, audit logs, SLA, dedicated support | Large orgs |
| **Overage** | $2/MTok compressed | Pay-as-you-go beyond tier limits | All paid tiers |

**Freemium strategy**: The open-source tool (this repo) is genuinely useful — not crippled. It uses TF-IDF embeddings (98% less memory, decent quality). The paid SaaS tier unlocks SBERT/ONNX embeddings for measurably better compression ratios, plus cloud API access, team features, and higher throughput. Users upgrade because paid is better, not because free is broken.

### Product Line 2: AI Benchmark Repository

| Tier | Price | Benchmark Access | Target |
|------|-------|-----------------|--------|
| **Free** | $0 | Browse leaderboards, submit benchmarks, basic search | Everyone |
| **Pro** (bundled with CaaS Pro) | $29/month | API access, custom comparisons, export CSV/JSON, recommendation engine | Power users |
| **Enterprise** | Custom | Private fleet benchmarks, procurement recommendations, team analytics | Hardware buyers, ML teams |

**Data network effect**: Every upload makes the database more valuable. Free tier drives volume. API access drives revenue. Enterprise private benchmarks are high-value.

### Product Line 3: AI News Center

| Revenue Stream | Model | Target |
|----------------|-------|--------|
| **Organic traffic** | SEO → funnel to CaaS/Benchmarks | All developers |
| **Sponsored content** | $5-15K/post from AI tool vendors | GPU makers, cloud providers, AI startups |
| **Newsletter premium** | $9/month for deep-dive analysis | AI decision makers |

**Flywheel**: News drives traffic → traffic drives benchmark submissions → benchmarks drive CaaS signups → CaaS usage generates news-worthy data (compression trends, cost savings reports).

### Product Line 4: Knowledge Hub (NotebookLM Competitor)

| Tier | Price | Includes | Target |
|------|-------|----------|--------|
| **Free** | $0 | 3 notebooks, 10 sources each, local processing, TF-IDF retrieval | Individuals, students |
| **Pro** (bundled with CaaS Pro) | $29/month | Unlimited notebooks/sources, SBERT compressed retrieval, podcast generation, API access, 50M tokens/month | Researchers, power users |
| **Team** | $19/user/month (min 5) | Shared notebooks, role-based access, team knowledge base, Slack/Notion/GitHub connectors | Dev teams, research teams |
| **Enterprise** | Custom | Self-hosted, SSO, audit logs, custom connectors, unlimited storage | Enterprises |

**Differentiator vs NotebookLM**: Model-agnostic (not Google-locked), self-hostable, compressed retrieval (5-20x fewer tokens consumed per query), API access, multi-tenant.

**Differentiator vs open-notebook/SurfSense**: Compressed retrieval is unique — same RAG quality, 85% fewer tokens. Plus enterprise features (multi-tenant, SSO, connectors) that OSS tools lack.

### Product Line 5: Agent Context Hub (ref.tools/Context7 Competitor)

| Tier | Price | Includes | Target |
|------|-------|----------|--------|
| **Free** | $0 | 5 indexed libraries, local search, basic compression, 1K queries/month | Solo devs |
| **Pro** (bundled with CaaS Pro) | $29/month | Unlimited libraries, SBERT compressed search, version-specific docs, 90+ formats, 50K queries/month | Individual developers |
| **Team** | $19/user/month (min 5) | Shared library indexes, team doc standards, private repo indexing, 500K queries/month | Dev teams |
| **Enterprise** | Custom | Self-hosted, private library indexes, SSO, custom scraping pipelines, unlimited | Large orgs |

**Differentiator vs Context7**: Offline/local option, no rate limits, compressed retrieval (85% fewer tokens in responses vs Context7's raw dumps), open source core, self-hostable.

**Differentiator vs ref.tools**: Compressed output (ref.tools caps at 5K tokens; gotcontext delivers same info in 750 tokens), open source, self-hosted option, version-specific.

**Differentiator vs docs-mcp-server**: Same local-first approach, but adds semantic compression on top — docs-mcp-server returns full indexed chunks, gotcontext returns compressed skeletons.

### Combined Revenue Projections

| Month | CaaS MRR | Benchmarks MRR | News MRR | Knowledge Hub MRR | Agent Context MRR | Total MRR |
|-------|---------|---------------|---------|-------------------|-------------------|-----------|
| 1 (launch CaaS) | $145 | — | — | — | — | **$145** |
| 6 (launch benchmarks) | $4,800 | $500 | — | — | — | **$5,300** |
| 12 (launch news + context hub) | $24,100 | $5,000 | $2,000 | — | $1,500 | **$32,600** |
| 18 (launch knowledge hub) | $142,000 | $25,000 | $8,000 | $5,000 | $15,000 | **$195,000** |
| 24 | $467,500 | $80,000 | $20,000 | $40,000 | $75,000 | **$682,500** |

**Year 1 ARR** (CaaS only): ~$290K
**Year 2 ARR** (all 5 products): ~$8.2M
**Year 3 ARR** (projected with network effects): ~$30-50M

### Valuation at Series A (Month 18-24)

At $3-8M ARR with 80%+ growth + data moat + 5 product lines:
- Standard SaaS: 15-20x ARR = **$30-100M**
- AI premium (1.5x): **$45-150M**
- Data network effect premium (benchmark repo): additional 2-3x multiplier on that segment
- With proven benchmarks + research backing: **premium end**

---

## 4. Go-To-Market Strategy

### Phase 1: Product-Led Growth (Months 1-3)

**Goal**: 500 free users, 25 paying customers, validate PMF

1. **Launch on Product Hunt / Hacker News**
   - "We saved 55% of tokens across Claude, Codex, and Gemini — here's the proof"
   - Lead with benchmark data (developers love data)
   - Open source core stays free (goodwill + trust)

2. **Developer content marketing**
   - Blog post: "How we cut our Claude Code bill by 55% with one MCP server"
   - Blog post: "RTK saves 89% on CLI output. We built that into a compression proxy."
   - Blog post: "13x document compression: the research papers behind gotcontext.ai"
   - YouTube: 5-minute demo showing real token savings in Claude Code
   - Comparison pages: "gotcontext.ai vs RTK", "gotcontext.ai vs Headroom"

3. **Community seeding**
   - Post in Claude Code Discord, Cursor community, AI coding subreddits
   - Answer token optimization questions on Stack Overflow / HN
   - Contribute to MCP ecosystem discussions

4. **Free tier as funnel**
   - `pip install gotcontext` — works immediately
   - One-line MCP config for Claude/Cursor/Gemini
   - Dashboard shows tokens saved + money saved in real-time

### Phase 2: Growth Engine (Months 3-6)

**Goal**: 2,000 free users, 100 Pro, first Team customers

1. **Integration partnerships**
   - Cursor marketplace listing
   - Claude Code recommended tools
   - Gemini CLI extensions directory
   - VS Code extension (wraps our MCP proxy)

2. **Content flywheel**
   - Weekly "Token Savings Report" newsletter (industry benchmarks)
   - SEO: "best MCP token optimizer", "Claude Code cost reduction", "Gemini CLI optimization"
   - Guest posts on dev blogs (Dev.to, Medium, HN)

3. **Pricing optimization**
   - A/B test pricing page
   - Add annual billing (20% discount)
   - Introduce lifetime deal for early adopters ($299 = Pro forever)

### Phase 3: Enterprise Push (Months 6-12)

**Goal**: First enterprise contracts, $100K+ ARR

1. **Self-hosted option** for enterprises that can't send data externally
2. **Team dashboard** showing org-wide token savings + cost attribution
3. **SOC 2 Type II** compliance (table stakes for enterprise)
4. **Sales-assisted motion** for $500+/month accounts
5. **Case studies** from Phase 1-2 customers with real ROI numbers

### Phase 4: Platform Expansion (Months 12-24)

**Goal**: $1M+ ARR, Series A readiness, all 5 products live

1. **AI News Center live** at gotcontext.ai/news — curated weekly digest, newsletter
2. **Agent Context Hub beta** at gotcontext.ai/docs — compressed doc retrieval for coding agents
3. **Benchmark recommendation engine**: "Best model for your GPU and budget"
4. **Marketplace**: third-party compression plugins
5. **REST/GraphQL API**: gotcontext.ai as a service (not just MCP)
6. **Analytics**: "Token Intelligence" — show where tokens are being wasted
7. **Integrations**: Datadog, Grafana, PagerDuty for token budget alerting
8. **Data-driven reports**: publish industry insights from benchmark + compression data
9. **Sponsored content partnerships** with GPU makers, cloud providers

### Phase 5: Knowledge Platform (Months 18-30)

**Goal**: Full platform flywheel, $5M+ ARR

1. **Knowledge Hub launch** at gotcontext.ai/notebooks — compressed RAG notebooks
2. **Agent Context Hub GA** — version-specific doc indexing for 500+ frameworks
3. **Cross-product integration**: Knowledge Hub uses CaaS compression, Agent Context Hub feeds benchmark data
4. **Podcast generation** in Knowledge Hub (multi-speaker, multi-model)
5. **Enterprise connectors**: Slack, Notion, Confluence, GitHub, Jira for Knowledge Hub
6. **Self-hosted enterprise bundles**: all 5 products in a single deployment

---

## 5. Website Structure (gotcontext.ai)

### Homepage

```
[HERO]
"Stop wasting 55% of your AI tokens on noise."
[Subheadline] gotcontext.ai compresses context for Claude Code, Codex, and Gemini CLI.
             Drop-in MCP proxy. Works in 30 seconds.
[CTA] Get Started Free → | See Benchmarks →
[Social proof] "Saved 14,000 tokens on my first request" — @developer

[BENCHMARK SECTION]
Real API measurements, not marketing claims.
| Provider | Before | After | Savings |
| Claude   | 61K    | 45K   | 26.4%   |
| Codex    | 37K    | 23K   | 38.2%   |
| Gemini   | 69K    | 31K   | 55.7%   |

[PLATFORM OVERVIEW - 5 pillars]
1. Context Compression — 85-96% token savings via semantic compression
2. AI Benchmark Repository — find the best model for your hardware and budget
3. AI News & Intelligence — stay ahead of AI infrastructure trends
4. Knowledge Hub — upload docs, chat with compressed context (NotebookLM, but open)
5. Agent Context Hub — always-current framework docs for your AI agent (Context7, but compressed)

[HOW IT WORKS - 3 steps]
1. Install: pip install gotcontext
2. Configure: one line in your MCP config
3. Save: see token savings in real-time

[FEATURES BENTO GRID]
- 13x Document Compression
- CLI Output Optimizer (11 strategies)
- Tool Schema Compression (96% reduction)
- Session Continuity (survives compaction)
- Research-Backed (3 arXiv papers)
- Cross-Platform (Claude, Codex, Gemini)
- Knowledge Hub (upload → chat → compressed RAG)
- Agent Context Hub (docs → compressed MCP retrieval)
- Global Benchmark Database (coming soon)

[PRICING]
Free / Pro $29 / Team $19/user / Enterprise

[COMPARISON]
gotcontext.ai vs RTK vs Headroom vs mcp-compressor

[FOOTER]
GitHub | Docs | Blog | Benchmarks | News | Discord | Twitter
```

### Key Pages

- `/` — landing page with live demo, pricing, agent savings comparison
- `/pricing` — comparison table with ROI calculator
- `/benchmarks` — **AI Benchmark Repository** (crowd-sourced inference database)
- `/benchmarks/leaderboard` — rankings by GPU, model, quant format, cost-per-quality
- `/benchmarks/submit` — upload inference runs (web UI or API)
- `/benchmarks/compare` — side-by-side model/hardware/quant comparisons
- `/notebooks` — **Knowledge Hub** (upload docs, chat with compressed context)
- `/notebooks/new` — create notebook, add sources (PDF, web, video, audio, code)
- `/notebooks/:id/chat` — RAG chat with compressed retrieval
- `/docs` — **Agent Context Hub** (framework doc search for AI agents)
- `/docs/index` — browse/manage indexed libraries (version-specific)
- `/docs/search` — semantic search across indexed docs (compressed results)
- `/news` — **AI News Center** (curated industry intelligence)
- `/news/weekly` — weekly digest of AI infrastructure and tooling
- `/getting-started` — setup guide, MCP config, API reference
- `/blog` — technical content, case studies, token savings reports
- `/compare/rtk` — head-to-head comparison with ROI calculator
- `/compare/headroom` — feature comparison
- `/compare/notebooklm` — Knowledge Hub vs NotebookLM comparison
- `/compare/context7` — Agent Context Hub vs Context7 comparison
- `/enterprise` — self-hosted, SOC 2, SLA details

---

## 6. Tech Stack for SaaS

| Component | Tool | Why |
|-----------|------|-----|
| **Landing page** | Next.js + Vercel | Fast, SEO-friendly, edge deployment |
| **Auth** | Clerk or Auth0 | Pre-built, SOC 2 ready |
| **Payments** | Stripe | Usage-based billing, metering API |
| **Backend API** | FastAPI (Python) | Same language as Token Saver core |
| **Database** | Supabase (Postgres) | Real-time, auth, storage in one |
| **Benchmark DB** | Supabase (Postgres) + TimescaleDB | Time-series inference metrics at scale |
| **Metering** | Stripe Meters or Orb | Token usage tracking |
| **Analytics** | PostHog | Product analytics, self-hostable |
| **Monitoring** | Better Stack | Uptime, logs, status page |
| **Docs** | Mintlify or Nextra | Beautiful dev docs |
| **Email** | Resend | Transactional + marketing |
| **CDN** | Cloudflare | Edge, WAF, bot protection |
| **Search** | Meilisearch or Typesense | Instant benchmark search/filtering |
| **Vector store** | Qdrant or Weaviate | Knowledge Hub + Agent Context Hub embeddings at scale |
| **Object storage** | S3 / R2 (Cloudflare) | Knowledge Hub document storage (PDF, audio, video) |
| **Job queue** | BullMQ (Redis) or Celery | Async doc ingestion, indexing, podcast generation |

---

## 7. Launch Timeline — Phased

### Phase 1: CaaS (Months 1-3) — Ship first, revenue first

| Week | Action | Goal |
|------|--------|------|
| 1 | Landing page live (gotcontext.ai) | Email signups |
| 1 | PyPI package published (`pip install gotcontext`) | Install funnel |
| 2 | Product Hunt launch | 500 upvotes, 100 signups |
| 2 | Hacker News "Show HN" post | Front page, community discussion |
| 3 | Blog: benchmark results post | SEO, backlinks |
| 3 | Discord community launched | Support channel |
| 4 | Pro tier payments live (Stripe) | First revenue |
| 6 | VS Code extension published | IDE distribution |
| 8 | Team tier live | Multi-user accounts |
| 12 | Enterprise pilot (first 3 orgs) | Pipeline for enterprise sales |

### Phase 2: AI Benchmark Repository (Months 4-8)

| Week | Action | Goal |
|------|--------|------|
| 16 | Seed database with automated runs (50+ models × 10+ GPUs × 5 quant formats) | Content before launch |
| 18 | Public beta at gotcontext.ai/benchmarks | Community submissions |
| 20 | CLI upload tool (`gotcontext benchmark submit`) | Frictionless contribution |
| 22 | MCP tool for AI agents to auto-submit benchmarks | Agent-driven data growth |
| 24 | Leaderboards + recommendation engine live | "Best model for your GPU" |
| 28 | API access for Pro/Enterprise tiers | Revenue from benchmark data |
| 32 | Hardware vendor partnerships (NVIDIA, AMD) | Sponsored benchmarks, credibility |

### Phase 3: AI News Center (Months 6-12)

| Week | Action | Goal |
|------|--------|------|
| 24 | Launch gotcontext.ai/news with curated weekly digest | SEO traffic |
| 28 | Newsletter (free + premium) | Email list growth |
| 32 | Sponsored content partnerships | Ad revenue from AI vendors |
| 36 | Data-driven reports (from benchmark repo data) | Thought leadership |
| 40 | Premium analysis tier ($9/month) | Subscription revenue |

### Phase 4: Agent Context Hub (Months 10-18)

| Week | Action | Goal |
|------|--------|------|
| 40 | Fork docs-mcp-server (MIT), integrate gotcontext compression pipeline | Baseline framework |
| 44 | Beta at gotcontext.ai/docs — 50 frameworks indexed (React, Next.js, LangChain, etc.) | Developer testing |
| 48 | MCP tool for AI agents: `gotcontext_search_docs` + `gotcontext_read_doc` | Agent integration |
| 52 | Version-specific indexing — pin docs to project dependency versions | Key differentiator |
| 56 | Self-hosted option (Docker) for enterprise | Enterprise pipeline |
| 60 | 500+ libraries indexed, comparison pages vs Context7/ref.tools | Growth marketing |

### Phase 5: Knowledge Hub (Months 14-24)

| Week | Action | Goal |
|------|--------|------|
| 56 | Fork open-notebook (MIT), replace LangChain RAG with gotcontext compression | Baseline framework |
| 60 | Beta at gotcontext.ai/notebooks — PDF/web/video/audio ingestion | User testing |
| 64 | Compressed RAG chat — same quality, 85% fewer tokens per query | Core differentiator |
| 68 | Team notebooks with role-based access | Team adoption |
| 72 | Enterprise connectors (Slack, Notion, Confluence, GitHub) | Enterprise pipeline |
| 80 | Podcast generation (multi-speaker, multi-model) | Feature parity with NotebookLM |
| 96 | Self-hosted enterprise bundle (all 5 products) | Enterprise upsell |

---

## 8. Key Metrics to Track

| Metric | Target (Month 6) | Target (Month 12) | Target (Month 24) |
|--------|-----------------|-------------------|-------------------|
| Free signups (CaaS) | 2,000 | 10,000 | 50,000 |
| Free → Pro conversion | 5% | 7% | 10% |
| Combined MRR | $5,300 | $32,600 | $682,500 |
| Churn (monthly) | <5% | <3% | <2% |
| NRR | >110% | >120% | >130% |
| Tokens compressed/month | 500M | 5B | 50B |
| Benchmark submissions | — | 50,000 | 500,000 |
| Benchmark MAU | — | 5,000 | 50,000 |
| News center monthly traffic | — | 10,000 | 100,000 |
| Agent Context Hub libraries indexed | — | 50 | 500+ |
| Agent Context Hub queries/month | — | 10,000 | 500,000 |
| Knowledge Hub notebooks created | — | — | 25,000 |
| Knowledge Hub MAU | — | — | 10,000 |
| DAU (active compression) | 200 | 1,000 | 5,000 |

---

## 9. Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|-----------|
| Token prices drop so fast optimization becomes irrelevant | Medium | High | Pivot to quality (better context = better answers, not just cheaper). Benchmark repo + news center are price-independent. |
| RTK (28K stars) adds document compression | Low | High | 2-year head start on semantic compression + research backing. Benchmark repo is a separate moat. |
| Anthropic/OpenAI build native optimization | Medium | High | Move up-stack to analytics + intelligence. Benchmark repo is vendor-neutral — benefits from more providers. |
| Benchmark repo can't reach critical mass | Medium | High | Seed with automated runs before launch (50+ models × 10+ GPUs). Add MCP tool so AI agents auto-submit. |
| Slow enterprise adoption | High | Medium | Focus on PLG first, enterprise comes with proof |
| Open source competitors copy CaaS features | Medium | Low | Benchmark data moat + SaaS convenience is the moat. Data network effects can't be forked. |
| Content treadmill burns out solo founder | Medium | Medium | Automate news curation using own compression tools. Focus on data-driven reports from benchmark data. |
| Fake/manipulated benchmark submissions | Medium | Medium | Hardware fingerprinting, statistical outlier detection, verified-device badges |
| Context7/Nia outpace Agent Context Hub | Medium | Medium | Compressed retrieval is a fundamental differentiator they can't replicate without rebuilding their architecture. Offline/self-hosted is a moat for enterprise. |
| NotebookLM adds model-agnostic support | Low | High | Google is unlikely to abandon their own models. Self-hosted + compression are structural advantages. open-notebook community provides innovation velocity. |
| OSS baselines diverge from our forks | Medium | Low | Contribute upstream, maintain thin integration layers. Core value is compression, not the fork. |

---

## 10. Future Vision: Global AI Benchmark Repository

**gotcontext.ai/benchmarks** — the world's first open, crowd-sourced benchmark database for AI model inference. Think [UserBenchmark.com](https://www.userbenchmark.com/) but for AI.

### What It Is

A global repository where developers, researchers, and AI agents can upload and query real-world inference data:

| Data Category | Examples |
|---------------|----------|
| **Model metadata** | Model name, parameter count, quant format (GGUF Q4_K_M, AWQ, GPTQ, FP16), architecture |
| **Hardware profiles** | GPU (VRAM, model, driver), CPU, RAM, OS, CUDA/ROCm version |
| **Token throughput** | Tokens/sec generation, prompt processing speed (tok/s), time-to-first-token |
| **Quantization comparisons** | Quality vs speed tradeoffs across quant levels for the same model |
| **Cost metrics** | $/MTok for API providers, $/hr for self-hosted, cost-per-quality-point |
| **Prompt characteristics** | Input token count, output token count, context window utilization |
| **Quality scores** | Perplexity deltas, benchmark scores (MMLU, HumanEval, etc.) per quant level |

### How It Works

1. **Users upload** — CLI tool or web UI submits inference runs with hardware + model + performance data
2. **AI agents upload** — MCP tool allows AI coding agents to auto-report their own inference metrics
3. **Everyone queries** — search by model, hardware, quant format, budget, or use case
4. **Rankings** — leaderboards for fastest tok/s per GPU, best quality-per-dollar, best quant for each model
5. **Recommendations** — "Given your RTX 4090 and $0.50/hr budget, run Llama 3.3 70B at Q4_K_M for best quality"

### Why This Is Huge

- **No central source of truth exists.** People share benchmarks on Reddit, Twitter, and Discord — fragmented, anecdotal, unsearchable.
- **Quantization choice is a nightmare.** Developers guess which quant format balances speed vs quality. A global database with real data solves this.
- **Hardware purchase decisions are blind.** "Should I buy a 4090 or two 3090s for inference?" — there is no UserBenchmark equivalent to answer this.
- **API vs self-hosted cost comparison is manual.** A database that tracks both lets teams optimize spend.
- **AI agents can self-optimize.** With an MCP tool for benchmark submission/query, agents can pick models and settings based on real crowd-sourced data — not vibes.

### Synergy with gotcontext.ai Core

This is a natural extension of the token optimization mission:
- gotcontext.ai **reduces** what you send to models (context compression)
- gotcontext.ai/benchmarks helps you **pick the right model** to send it to
- Combined: "We compressed your 100K context to 10K tokens. Based on global benchmarks, route it to Llama 3.3 70B Q5_K_M on your hardware for best quality-per-dollar."

### Revenue Model

See §3 Product Line 2 for pricing tiers. Key insight: the free tier drives data volume (network effect), while Pro/Enterprise API access drives revenue. Bundling with CaaS Pro means benchmark users become compression customers and vice versa.

---

## 11. OSS Baseline Research — Build vs Fork Analysis

### Knowledge Hub Baselines (NotebookLM Competitors)

| Project | Stars | Stack | License | Recommendation |
|---------|-------|-------|---------|---------------|
| **[open-notebook](https://github.com/lfnovo/open-notebook)** | **21.8K** ⭐ | Python (FastAPI) + Next.js + SurrealDB + LangChain | MIT | **🏆 Fork as baseline.** Python backend aligns with gotcontext core. 16+ model providers, Docker deploy, REST API, multi-format ingestion (PDF, video, audio, web), podcast generation (1-4 speakers). Active community, trending on GitHub. |
| **[SurfSense](https://github.com/modsetter/surfsense)** | **13.6K** ⭐ | Next.js + FastAPI | MIT | **Strong reference.** Teams-focused, already has SaaS (surfsense.com). External source connectors (Slack, Linear, Notion, GitHub, YouTube) are a roadmap template. Don't fork — learn from their connector architecture. |
| [KnowNote](https://github.com/MrSibe/KnowNote) | 966 ⭐ | Electron + TypeScript | MIT | Skip. Desktop-only, not web-native. |
| [InsightsLM](https://github.com/theaiautomators/insights-lm-public) | 512 ⭐ | React + Supabase + N8N | MIT | Reference only. Lightweight template with audio summaries. |
| [Local-NotebookLM](https://github.com/nicholasgriffintn/Local-NotebookLM) | 867 ⭐ | Python | MIT | Reference only. Fully local, minimal features. |

**Recommended approach**: Fork **open-notebook**, replace its LangChain RAG pipeline with gotcontext's `SemanticCompressor` + `EmbeddingManager`. Keep its multi-format ingestion, multi-model support, and podcast generation. Add multi-tenant scoping from `identity_scope.py`. Result: same RAG quality, 5-20x fewer tokens per retrieval.

### Agent Context Hub Baselines (ref.tools/Context7 Competitors)

| Project | Stars | Stack | License | Recommendation |
|---------|-------|-------|---------|---------------|
| **[docs-mcp-server](https://github.com/arabold/docs-mcp-server)** (Grounded Docs) | **1,176** ⭐ | TypeScript + SQLite | MIT | **🏆 Fork as baseline.** Explicitly bills itself as "open-source Context7/Nia/Ref.Tools alternative". Web UI, CLI, 90+ file formats, version-specific indexing, optional embeddings (OpenAI, Ollama, Gemini). Web scraping + local files + GitHub + npm + PyPI sources. |
| **[Context (Neuledge)](https://github.com/neuledge/context)** | **141** ⭐ | TypeScript + SQLite FTS5 | Apache 2.0 | **Strong reference.** Fully offline, BM25 scoring, token-aware (~2K cap). Lightweight architecture worth studying for the local-first experience. |
| [ref-tools-mcp](https://github.com/ref-tools/ref-tools-mcp) | 1K ⭐ | TypeScript | Proprietary | Reference only. ref.tools' own server. Session-aware filtering, ~5K token cap. |
| [Docfork](https://github.com/docfork/docfork) | — | TypeScript | MIT | Reference. 9,000+ libraries, "Cabinets" for context isolation — interesting feature. |
| [GitMCP](https://gitmcp.io/) | — | TypeScript | OSS | Skip. Too simple — just reads README/llms.txt from GitHub repos. |
| [Deepcon](https://deepcon.ai/) | — | Cloud service | Proprietary | Reference. Claims 90% accuracy vs Context7's 65%. Semantic search approach worth studying. |
| [Nia](https://trynia.ai/) | — | Cloud service ($6.2M YC) | Proprietary | Competitor. Indexes codebase + docs, cross-session context, 15+ tools. Richest feature set but cloud-only. |

**Recommended approach**: Fork **docs-mcp-server**, add a gotcontext compression layer in the retrieval pipeline. When a user/agent queries `gotcontext_search_docs("React useEffect cleanup")`, the server fetches indexed chunks then compresses them through `SemanticCompressor` before returning. Same relevance, 85% fewer tokens. Keep the web UI, version-specific indexing, and 90+ format support.

### Reddit Sentiment & Community Demand

**NotebookLM alternatives** (r/selfhosted, r/ollama, r/notebooklm, r/LocalLLaMA):
- Massive demand for self-hosted NotebookLM. Multiple threads with hundreds of upvotes.
- SurfSense is the most-recommended alternative — consistently top answer.
- Key themes: **privacy** ("I don't want Google reading my research"), **no data limits** (NotebookLM caps sources), **model choice** ("why am I locked to Gemini?").
- open-notebook gaining rapid traction (21.8K stars, trending on GitHub).

**Doc context tools** (r/cursor, r/ClaudeAI, r/mcp, r/LocalLLaMA):
- Context7 was hyped but backlash is growing: rate limits dropped from 6K to 1K free requests (Jan 2026), $10/mo paid tier.
- Top complaint: **token bloat** — "responses are too large, eating my context window." This is literally gotcontext's value prop.
- Active thread: r/cursor "What's the best Docs MCP? (Context7, Ref, etc)" — no clear winner, developers switching between tools frequently.
- r/ClaudeAI: "Am I missing something with the Context7 MCP hype?" — skepticism about whether any of these tools deliver real value. Opportunity to differentiate with benchmarks.
- r/LocalLLaMA: "What is the best Opensource Context7 Alternative" — demand for local-first, self-hosted solutions.

### The gotcontext Advantage: Compressed Retrieval

**No existing tool compresses its retrieval output.** This is the unique differentiator:

| What exists today | What gotcontext adds |
|---|---|
| docs-mcp-server fetches 10K tokens of React docs | gotcontext returns a 1.5K compressed skeleton with citations |
| open-notebook returns full RAG chunks | gotcontext returns compressed, ranked chunks with importance scores |
| Context7 serves large uncompressed responses | gotcontext serves the **skeleton** — same info, 85% fewer tokens |
| ref.tools hard-caps at 5K tokens (lossy truncation) | gotcontext compresses semantically — nothing lost, just denser |

This is a fundamental architectural advantage. Competitors would need to rebuild their retrieval pipelines to match it. gotcontext already has the compression engine — it just needs a retrieval frontend.

---

## 12. Immediate Next Steps (This Week)

**CaaS launch (priority):**
1. **Register gotcontext.ai domain**
2. **Create GitHub org** (gotcontext or gotcontext-ai)
3. **Deploy landing page** (Next.js on Vercel, dark mode, dev-focused)
4. **Set up Stripe** with Pro tier ($29/month, usage metering)
5. **Publish to PyPI** as `gotcontext` (wrapper around token-saver-5000)
6. **Write launch blog post** with benchmark data
7. **Prepare Product Hunt listing** (scheduled for Tuesday 12:01 AM PT)
8. **Create Discord server** for community

**Benchmark repo groundwork (parallel):**
9. **Design benchmark submission schema** (model, hardware, quant, throughput, cost)
10. **Prototype database schema** (Postgres + TimescaleDB for time-series metrics)
11. **Plan automated seed runs** — identify 50+ models × 10+ GPUs for initial data

**News center groundwork (low-effort now):**
12. **Register social accounts** (Twitter/X, LinkedIn, YouTube for gotcontext brand)
13. **Draft first 3 newsletter issues** from existing benchmark data and market research

**Agent Context Hub groundwork (parallel research):**
14. **Clone and evaluate docs-mcp-server** (arabold) — assess architecture, identify compression insertion points
15. **Prototype compressed retrieval**: index a framework (React), serve compressed vs raw, measure token savings
16. **Design MCP tool schema**: `gotcontext_search_docs` and `gotcontext_read_doc` with compression parameters

**Knowledge Hub groundwork (future phase):**
17. **Clone and evaluate open-notebook** — assess SurrealDB dependency, identify LangChain RAG replacement points
18. **Map integration points**: where `SemanticCompressor` replaces LangChain's retrieval chain
19. **Design multi-tenant notebook schema** using existing `identity_scope.py` patterns

---

## Research Sources

### Market Sizing & AI Economics
- [AI Inference Cost Crisis 2026 (Oplexa)](https://oplexa.com/ai-inference-cost-crisis-2026/) — 280x price drop, 320% spend increase, $7M avg enterprise budget
- [AI Inference 55% of Cloud Spending (ByteIota)](https://byteiota.com/ai-inference-costs-55-of-cloud-spending-in-2026/) — $37.5B cloud infra, $50B+ inference chip market
- [2026 AI Market Report: $1T (AI Business Insider)](https://aibusinessinsider.org/2026-ai-market-report-1-trillion-and-growing-9/) — $1.02T market, $120B+ VC funding pace
- [AI Inference Hardware Benchmarking Market (FactMR)](https://www.factmr.com/report/ai-inference-hardware-benchmarking-test-market) — $1.2B → $4.5B (2026-2033)
- [AI Benchmarking Platform Market (MarketIntelo)](https://marketintelo.com/report/ai-benchmarking-platform-market) — market sizing through 2033
- [AI Developer Tools Industry (WifiTalents)](https://wifitalents.com/ai-developer-tools-industry-statistics/) — 92% dev adoption, 21.5% CAGR, $3.5B DevOps
- [MCP Reaches 97M Installs (David & Goliath)](https://davidandgoliath.ai/daily-ai-briefing/mcp-97-million-installs-ai-standard) — protocol adoption data
- [AI Token Spend Dynamics (Deloitte)](https://www.deloitte.com/us/en/insights/topics/emerging-technologies/ai-tokens-how-to-navigate-spend-dynamics.html)
- [Gartner: 90% LLM Cost Drop by 2030](https://www.gartner.com/en/newsroom/press-releases/2026-03-25-gartner-predicts-that-by-2030-performing-inference-on-an-llm-with-1-trillion-parameters-will-cost-genai-providers-over-90-percent-less-than-in-2025)

### SaaS Strategy
- [SaaS GTM Playbook 2026](https://thesmarketers.com/toolkits-guides/saas-gtm-playbook-2026/)
- [SaaS Pricing Strategy Guide 2026](https://www.nxcode.io/resources/news/saas-pricing-strategy-guide-2026)
- [AI Startup Valuation Multiples](https://qubit.capital/blog/ai-startup-valuation-multiples)
- [SaaS Benchmarks Report 2026](https://www.averi.ai/how-to/the-saas-benchmarks-report-2026-how-your-startup-stacks-up-(from-pre-seed-to-series-a))
- [Solo Founder SaaS Guide 2026](https://www.twocents.software/blog/solo-founders-guide-to-launching-saas/)
- [SaaS Landing Page Trends 2026](https://www.saasframe.io/blog/10-saas-landing-page-trends-for-2026-with-real-examples)
- [AI SaaS Solo Founder Success Stories](https://crazyburst.com/ai-saas-solo-founder-success-stories-2026/)

### Comparable Models
- [UserBenchmark](https://www.userbenchmark.com/) — hardware benchmark aggregation model (crowd-sourced, ad-supported, ~50M visits/month)
- [Open LLM Leaderboard (Hugging Face)](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) — model quality rankings (no hardware/cost data)

### OSS Baselines & Competitors (Knowledge Hub)
- [open-notebook](https://github.com/lfnovo/open-notebook) — 21.8K★, MIT, Python + Next.js + SurrealDB, multi-model, multi-format, podcast generation
- [SurfSense](https://github.com/modsetter/surfsense) — 13.6K★, MIT, NotebookLM for teams, external source connectors (Slack, Linear, Notion, GitHub)
- [Google NotebookLM](https://notebooklm.google.com/) — Google-only models, free tier, no API, no self-hosting
- [KnowNote](https://github.com/MrSibe/KnowNote) — 966★, Electron desktop app, local-first

### OSS Baselines & Competitors (Agent Context Hub)
- [docs-mcp-server / Grounded Docs](https://github.com/arabold/docs-mcp-server) — 1,176★, MIT, TypeScript, 90+ formats, "open-source Context7/Nia/Ref.Tools alternative"
- [Context7 (Upstash)](https://context7.com/) — 51.5K MCP installs, cloud-hosted, $10/mo, rate-limited free tier (1K requests)
- [Context (Neuledge)](https://github.com/neuledge/context) — 141★, Apache 2.0, fully local, SQLite FTS5, BM25 scoring
- [ref.tools](https://ref.tools/) — session-aware filtering, ~5K token cap, $9/mo, MCP-native
- [Nia](https://trynia.ai/) — YC-backed ($6.2M), codebase + docs indexing, 15+ tools, $14.99/mo
- [Deepcon](https://deepcon.ai/) — semantic search, claims 90% accuracy, $8-$20/mo
- [Docfork](https://github.com/docfork/docfork) — MIT, 9,000+ libraries, "Cabinets" for context isolation
- [Top 7 MCP Alternatives for Context7 (DEV.to)](https://dev.to/moshe_io/top-7-mcp-alternatives-for-context7-in-2026-2555) — landscape overview with comparison table
