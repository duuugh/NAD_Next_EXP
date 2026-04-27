import { useMemo, useState } from "react";
import { Badge } from "../components/Badge";
import { DataState } from "../components/DataState";
import { MetricsBarChart } from "../components/MetricsBarChart";
import { MetricsLineChart } from "../components/MetricsLineChart";
import { SummaryStatGrid } from "../components/SummaryStatGrid";
import { TimelineView } from "../components/TimelineView";
import { useJsonData } from "../hooks/useJsonData";
import type { ExperimentCard, ExportedCards, ResearchTimeline } from "../types/research";

const failureReasons: Record<string, string> = {
  bon_cluster_router: "结构更复杂，但没有带来足够稳定的整体收益，复杂度不值得。",
  bon_activation_a1: "activation 信号有解释价值，但作为主选择器不够稳。",
  bon_mixed_v3_tailveto: "本地 AIME 有小幅收益，但没有穿过真实 leaderboard 检验。",
  bon_em_regularized_m4: "paper-inspired 方案值得试，但最终没有超越更稳的 targeted patch 主线。",
  bon_local_head: "candidate-level 信号有价值，但最后只支持极小 patch，不足以替代主干。",
};

function buildCacheRows(cards: ExperimentCard[], metric: string) {
  const cacheKeys = Array.from(new Set(cards.flatMap((card) => Object.keys(card.perCache ?? {}))));
  return cacheKeys.map((cacheKey) => {
    const row: Record<string, number | string> = { cache: cacheKey };
    for (const card of cards) {
      const value = card.perCache?.[cacheKey]?.[metric];
      if (typeof value === "number") {
        row[card.title] = Number(value.toFixed(4));
      }
    }
    return row;
  });
}

export function BestOfNPage() {
  const cardsState = useJsonData<ExportedCards>("/data/best_of_n_cards.json");
  const timelineState = useJsonData<ResearchTimeline>("/data/research_timeline.json");

  const [metric, setMetric] = useState("mean_score");
  const [focusId, setFocusId] = useState("bon_mixed_v2_logprob");
  const [cacheFilter, setCacheFilter] = useState("all");

  const cards = cardsState.data?.cards ?? [];
  const finalBest = cards.find((card) => card.id === cardsState.data?.finalBestId);
  const focus = cards.find((card) => card.id === focusId) ?? finalBest ?? cards[0];

  const compareCards = useMemo(() => {
    const importantIds = [cardsState.data?.finalBestId, "bon_mixed_v1_complete", focus?.id].filter(Boolean);
    const matched = importantIds
      .map((id) => cards.find((card) => card.id === id))
      .filter((card): card is ExperimentCard => Boolean(card));
    return Array.from(new Map(matched.map((card) => [card.id, card])).values());
  }, [cards, cardsState.data?.finalBestId, focus?.id]);

  const filteredCacheRows = useMemo(() => {
    return buildCacheRows(compareCards, metric).filter((row) => {
      const cacheKey = String(row.cache);
      if (cacheFilter === "all") return true;
      if (cacheFilter === "DS-R1") return cacheKey.startsWith("DS-R1/");
      if (cacheFilter === "Qwen3-4B") return cacheKey.startsWith("Qwen3-4B/");
      return cacheKey.endsWith(`/${cacheFilter}`);
    });
  }, [compareCards, metric, cacheFilter]);

  const summaryBarData = useMemo(() => {
    const metricRows = ["mean_score", "std_score", "score_range"].map((metricName) => {
      const row: Record<string, number | string> = { metric: metricName };
      for (const card of compareCards) {
        const value = card.overall?.[metricName];
        if (typeof value === "number") row[card.title] = Number(value.toFixed(4));
      }
      return row;
    });
    return metricRows;
  }, [compareCards]);

  const failureCards = cards.filter((card) => card.status === "deprecated" || card.status === "promising");

  return (
    <div className="space-y-6">
      <section className="hero-card gradient-border-blue">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="max-w-4xl">
            <div className="text-xs font-semibold uppercase tracking-[0.18em] text-blue-700">Best-of-N Final Route</div>
            <h1 className="mt-2 text-3xl font-bold tracking-tight text-slate-950 lg:text-4xl">{finalBest?.title ?? "nad_mixed_v2_aime_top2_gap1e3_logprob"}</h1>
            <p className="mt-4 text-sm leading-7 text-slate-700">
              最终胜出的不是大一统新系统，而是一个很克制的 AIME-only small patch：只在 top2 很接近时触发，用 tok_logprob 作为 tie-break，其他 cache 尽量保持 baseline 不动。
            </p>
          </div>
          {finalBest ? <Badge status={finalBest.status} /> : null}
        </div>
        <div className="mt-5">
          <SummaryStatGrid
            metrics={[
              { label: "mean_score", value: finalBest?.overall?.mean_score, accent: "blue", hint: "最终导出主线的总体 score 中心" },
              { label: "Patch Scope", value: "AIME 4 caches", accent: "blue", hint: "只动最有把握的局部范围" },
              { label: "Trigger", value: "top2 gap ≤ 1e-3", accent: "slate", hint: "只在 near-tie cases 出手" },
              { label: "Tie-break", value: "tok_logprob", accent: "slate", hint: "保留更稳的 token-level 证据" },
            ]}
          />
        </div>
      </section>

      <section className="grid gap-5 lg:grid-cols-[1.15fr_0.85fr]">
        <article className="card">
          <h2 className="section-title">为什么 small patch 成为最终方案</h2>
          <div className="mt-4 grid gap-3 md:grid-cols-3">
            <div className="panel-muted">
              <div className="text-sm font-semibold text-slate-900">改动范围小</div>
              <p className="mt-2 text-sm leading-6 text-slate-700">只修 AIME-only near-tie cases，不把全局 submission 一起拖动。</p>
            </div>
            <div className="panel-muted">
              <div className="text-sm font-semibold text-slate-900">证据链短</div>
              <p className="mt-2 text-sm leading-6 text-slate-700">规则透明：gap 很小才触发，核心依据是 tok_logprob，不依赖难解释的全局黑箱。</p>
            </div>
            <div className="panel-muted">
              <div className="text-sm font-semibold text-slate-900">风险可控</div>
              <p className="mt-2 text-sm leading-6 text-slate-700">其他 benchmark 不跟着一起改，最大化保留 baseline 的稳定性。</p>
            </div>
          </div>
        </article>

        <article className="card">
          <h2 className="section-title">关键指标</h2>
          <div className="mt-4">
            <SummaryStatGrid
              columns={2}
              compact
              metrics={[
                { label: "score_range", value: focus?.overall?.score_range, accent: "blue" },
                { label: "std_score", value: focus?.overall?.std_score, accent: "blue" },
                { label: "problem_count", value: focus?.overall?.problem_count, accent: "slate" },
                { label: "candidate_count", value: focus?.overall?.candidate_count, accent: "slate" },
              ]}
            />
          </div>
          <div className="mt-4 text-sm leading-6 text-slate-700">
            当前页面优先展示导出数据里真正稳定可用的 score-based 指标，并把“为什么这条路线更稳”讲清楚。
          </div>
        </article>
      </section>

      <section className="card">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div>
            <h2 className="section-title">方案切换与对比</h2>
            <p className="section-subtitle">支持按方案切换查看，并将最终方案与 baseline / 当前聚焦方案放到同一张图里对比。</p>
          </div>
          <div className="flex flex-wrap gap-3 text-sm">
            <select className="rounded-full border border-slate-300 px-3 py-2" value={focus?.id ?? ""} onChange={(e) => setFocusId(e.target.value)}>
              {cards.map((card) => <option key={card.id} value={card.id}>{card.title}</option>)}
            </select>
            <select className="rounded-full border border-slate-300 px-3 py-2" value={metric} onChange={(e) => setMetric(e.target.value)}>
              {["mean_score", "std_score", "score_range"].map((item) => <option key={item}>{item}</option>)}
            </select>
            <select className="rounded-full border border-slate-300 px-3 py-2" value={cacheFilter} onChange={(e) => setCacheFilter(e.target.value)}>
              <option value="all">全部 cache</option>
              <option value="DS-R1">DS-R1</option>
              <option value="Qwen3-4B">Qwen3-4B</option>
              <option value="aime24">aime24</option>
              <option value="aime25">aime25</option>
              <option value="brumo25">brumo25</option>
              <option value="gpqa">gpqa</option>
              <option value="hmmt25">hmmt25</option>
              <option value="lcb_v5">lcb_v5</option>
            </select>
          </div>
        </div>

        <div className="mt-5 grid gap-5 xl:grid-cols-[0.9fr_1.1fr]">
          <div className="space-y-4">
            {focus ? (
              <article className="panel-muted">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <div className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">当前聚焦方案</div>
                    <div className="mt-1 text-lg font-semibold text-slate-950">{focus.title}</div>
                  </div>
                  <Badge status={focus.status} />
                </div>
                <p className="mt-3 text-sm leading-6 text-slate-700">{focus.shortDescription}</p>
                {(focus.notes ?? []).length ? (
                  <ul className="mt-3 list-disc space-y-1 pl-5 text-sm leading-6 text-slate-600">
                    {focus.notes?.slice(0, 4).map((note) => <li key={note}>{note}</li>)}
                  </ul>
                ) : null}
              </article>
            ) : null}

            <article className="panel-muted">
              <div className="text-sm font-semibold text-slate-900">总指标对比</div>
              <p className="mt-2 text-sm leading-6 text-slate-600">把最终方案、baseline 和当前聚焦方案放在一起看，hover 可查看具体数值。</p>
              <div className="mt-4">
                <MetricsBarChart data={summaryBarData} xKey="metric" yKeys={compareCards.map((card) => card.title)} height={280} />
              </div>
            </article>
          </div>

          <article className="panel-muted">
            <div className="text-sm font-semibold text-slate-900">按 cache 的 {metric} 对比</div>
            <p className="mt-2 text-sm leading-6 text-slate-600">重点展示 final best 与其他代表方案在不同 cache 上的表现差异。</p>
            <div className="mt-4">
              <MetricsLineChart data={filteredCacheRows} xKey="cache" yKeys={compareCards.map((card) => card.title)} height={360} />
            </div>
          </article>
        </div>
      </section>

      <section className="card">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div>
            <h2 className="section-title">失败路线 / 被放弃路线</h2>
            <p className="section-subtitle">只保留简洁结论：它试过了、有什么局部价值、为什么没有成为最终主方案。</p>
          </div>
        </div>
        <div className="mt-5 grid gap-4 lg:grid-cols-2 xl:grid-cols-3">
          {failureCards.map((card) => (
            <article key={card.id} className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <div className="text-base font-semibold text-slate-950">{card.title}</div>
                  <div className="mt-2 text-sm leading-6 text-slate-700">{card.shortDescription}</div>
                </div>
                <Badge status={card.status} />
              </div>
              <div className="mt-3 rounded-xl border border-slate-200 bg-white p-3 text-sm leading-6 text-slate-600">
                {failureReasons[card.id] ?? "有局部信号，但没有成为更稳的最终提交路线。"}
              </div>
            </article>
          ))}
        </div>
      </section>

      <DataState loading={cardsState.loading || timelineState.loading} error={cardsState.error || timelineState.error}>
        <section className="card">
          <h2 className="section-title">Best-of-N 方法演化时间线</h2>
          <p className="section-subtitle">从 baseline 到 small patch，再到几条没有成为最终路线的支线。</p>
          <div className="mt-5">
            <TimelineView nodes={(timelineState.data?.nodes ?? []).filter((node) => node.task === "best_of_n" || node.task === "cross")} />
          </div>
        </section>
      </DataState>
    </div>
  );
}
