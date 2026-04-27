import { useMemo, useState } from "react";
import { Badge } from "../components/Badge";
import { DataState } from "../components/DataState";
import { DynamicsInteractiveViewer } from "../components/DynamicsInteractiveViewer";
import { MetricsBarChart } from "../components/MetricsBarChart";
import { MetricsLineChart } from "../components/MetricsLineChart";
import { SummaryStatGrid } from "../components/SummaryStatGrid";
import { TimelineView } from "../components/TimelineView";
import { useJsonData } from "../hooks/useJsonData";
import type { ExperimentCard, ExportedCards, ResearchTimeline } from "../types/research";

const overallMetrics = ["AUC-AUROC", "AUC-SelAcc", "AUROC@100%", "Stop@100%"];
const budgetMetricOptions = ["AUROC", "SelAcc"];

const routerRows = [
  { benchmark: "DS-R1/aime24", route: "rho_tail + -A_accel", decision: "enable", note: "prior validated strong case" },
  { benchmark: "DS-R1/aime25", route: "-A_accel only", decision: "enable", note: "strong and stable" },
  { benchmark: "DS-R1/hmmt25", route: "rho_tail + -A_accel", decision: "enable", note: "strong and stable" },
  { benchmark: "DS-R1/gpqa", route: "disable in conservative", decision: "borderline", note: "aggressive 可开，主提交不启用" },
  { benchmark: "DS-R1/lcb_v5", route: "disable in conservative", decision: "borderline", note: "收益边缘，不进主提交" },
  { benchmark: "DS-R1/brumo25", route: "disable", decision: "off", note: "未观察到值得单独启用的稳定增益" },
];

const droppedStrategies = [
  {
    id: "early_stop_dynamics_v1",
    title: "early_stop_dynamics_v1",
    reason: "只覆盖 1 个 dynamics cache，940/970 个 problem 实际仍走 fallback，证据范围太窄。",
    accent: "border-slate-200 bg-slate-50",
  },
  {
    id: "early_stop_dynamics_v2_local",
    title: "early_stop_dynamics_v2_local",
    reason: "适合作为 benchmark 级局部审计，但主要证据仍集中在少数 DS-R1 cache，不适合作为最终总方案。",
    accent: "border-amber-200 bg-amber-50/70",
  },
  {
    id: "early_stop_mean_confidence_plus_dyn_conservative",
    title: "early_stop_mean_confidence_plus_dyn_conservative",
    reason: "这是很强的 offline backbone，但主页最终突出的是更清晰、更安全的 explicit router 版本。",
    accent: "border-blue-200 bg-blue-50/70",
  },
];

function buildBudgetRows(cards: ExperimentCard[], cacheKey: string, metric: string) {
  const budgets = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
  return budgets.map((budget) => {
    const row: Record<string, number | string> = { budget: `${Math.round(budget * 100)}%` };
    for (const card of cards) {
      const points = card.perCache?.[cacheKey]?.by_budget;
      if (Array.isArray(points)) {
        const hit = points.find((item) => typeof item.budget === "number" && Math.abs(item.budget - budget) < 1e-6);
        const value = hit?.[metric];
        if (typeof value === "number") {
          row[card.title] = Number(value.toFixed(4));
        }
      }
    }
    return row;
  });
}

export function EarlyStopPage() {
  const { loading, error, data } = useJsonData<ExportedCards>("/data/early_stop_cards.json");
  const timelineState = useJsonData<ResearchTimeline>("/data/research_timeline.json");
  const [budgetMetric, setBudgetMetric] = useState<(typeof budgetMetricOptions)[number]>("AUROC");
  const [selectedCache, setSelectedCache] = useState("DS-R1/aime24");

  const cards = data?.cards ?? [];
  const finalBest = cards.find((card) => card.id === data?.finalBestId);
  const backbone = cards.find((card) => card.id === "early_stop_confidence_only");
  const localDynamics = cards.find((card) => card.id === "early_stop_dynamics_local");
  const pluginRoute = cards.find((card) => card.id === "early_stop_confidence_plus_dynamics");

  const compareCards = [finalBest, backbone, localDynamics, pluginRoute].filter((card): card is ExperimentCard => Boolean(card));

  const overallBarData = useMemo(() => {
    return overallMetrics.map((metricName) => {
      const row: Record<string, number | string> = { metric: metricName };
      for (const card of compareCards) {
        const value = card.overall?.[metricName];
        if (typeof value === "number") row[card.title] = Number(value.toFixed(4));
      }
      return row;
    });
  }, [compareCards]);

  const budgetRows = useMemo(() => buildBudgetRows(compareCards, selectedCache, budgetMetric), [compareCards, selectedCache, budgetMetric]);

  const selectableCaches = useMemo(() => {
    return Object.keys(finalBest?.perCache ?? {}).filter((cacheKey) => finalBest?.perCache?.[cacheKey]?.labeled);
  }, [finalBest]);

  return (
    <div className="space-y-6">
      <section className="hero-card gradient-border-emerald highlight-ring">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="max-w-4xl">
            <div className="text-xs font-semibold uppercase tracking-[0.18em] text-emerald-700">Early Stop Final Route</div>
            <h1 className="mt-2 text-3xl font-bold tracking-tight text-slate-950 lg:text-4xl">{finalBest?.title ?? "early_stop_dynamics_router_conservative_submit"}</h1>
            <p className="mt-4 text-sm leading-7 text-slate-700">
              最终主展示方案是 conservative router：不是把 dynamics plugin 全局打开，而是只在证据最稳定的 benchmark 上启用，让 rho_tail 和 -A_accel 变成“选择性增强器”。
            </p>
          </div>
          {finalBest ? <Badge status={finalBest.status} /> : null}
        </div>
        <div className="mt-5">
          <SummaryStatGrid
            metrics={[
              { label: "AUC-AUROC", value: finalBest?.overall?.["AUC-AUROC"], accent: "emerald" },
              { label: "AUC-SelAcc", value: finalBest?.overall?.["AUC-SelAcc"], accent: "emerald" },
              { label: "AUROC@100%", value: finalBest?.overall?.["AUROC@100%"], accent: "slate" },
              { label: "Stop@100%", value: finalBest?.overall?.["Stop@100%"], accent: "slate" },
            ]}
          />
        </div>
      </section>

      <section className="grid gap-5 lg:grid-cols-[1.05fr_0.95fr]">
        <article className="card">
          <h2 className="section-title">Router 策略一眼看懂</h2>
          <p className="section-subtitle">用表格说明 rho_tail / -A_accel 在不同 benchmark 上如何被选择性启用，而不是全局启用。</p>
          <div className="mt-5 overflow-x-auto">
            <table className="min-w-full text-sm">
              <thead>
                <tr className="border-b border-slate-200 text-left text-slate-500">
                  <th className="py-2 pr-4">Benchmark</th>
                  <th className="py-2 pr-4">Route</th>
                  <th className="py-2 pr-4">Decision</th>
                  <th className="py-2 pr-4">Why</th>
                </tr>
              </thead>
              <tbody>
                {routerRows.map((row) => (
                  <tr key={row.benchmark} className="border-b border-slate-100 align-top">
                    <td className="py-3 pr-4 font-medium text-slate-900">{row.benchmark}</td>
                    <td className="py-3 pr-4 text-slate-700">{row.route}</td>
                    <td className="py-3 pr-4">
                      <span className={`rounded-full px-2.5 py-1 text-xs font-medium ${row.decision === "enable" ? "bg-emerald-100 text-emerald-700" : row.decision === "borderline" ? "bg-amber-100 text-amber-700" : "bg-slate-100 text-slate-600"}`}>
                        {row.decision}
                      </span>
                    </td>
                    <td className="py-3 text-slate-600">{row.note}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </article>

        <article className="card">
          <h2 className="section-title">为什么选择性启用更好</h2>
          <div className="mt-4 grid gap-3">
            <div className="panel-muted">
              <div className="text-sm font-semibold text-slate-900">不是 every benchmark 都需要 dynamics</div>
              <p className="mt-2 text-sm leading-6 text-slate-700">gpqa 和 lcb_v5 在 aggressive 下可以尝试，但 conservative 主提交故意不启用，避免边缘收益吞掉整体稳定性。</p>
            </div>
            <div className="panel-muted">
              <div className="text-sm font-semibold text-slate-900">rho_tail 和 -A_accel 各有适用区间</div>
              <p className="mt-2 text-sm leading-6 text-slate-700">aime25 更像 -A_accel only；aime24 和 hmmt25 更适合组合路由，这正是 router 必须存在的原因。</p>
            </div>
            <div className="panel-muted">
              <div className="text-sm font-semibold text-slate-900">最终主线强调“安全上线”</div>
              <p className="mt-2 text-sm leading-6 text-slate-700">这页把 conservative 方案放到最显眼位置，因为它最符合稳定、可解释、可辩护的最终展示标准。</p>
            </div>
          </div>
        </article>
      </section>

      <section className="card">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div>
            <h2 className="section-title">方案对比：总体指标</h2>
            <p className="section-subtitle">对比 final router、confidence backbone、local dynamics 和 plugin 版本，hover 可看具体数值。</p>
          </div>
        </div>
        <div className="mt-5">
          <MetricsBarChart data={overallBarData} xKey="metric" yKeys={compareCards.map((card) => card.title)} height={340} />
        </div>
      </section>

      <section className="card">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div>
            <h2 className="section-title">不同 budget 下的性能变化</h2>
            <p className="section-subtitle">通过选择框切换 benchmark，并看不同方案在 budget 轴上的 AUROC / SelAcc 曲线。</p>
          </div>
          <div className="flex flex-wrap gap-3 text-sm">
            <select className="rounded-full border border-slate-300 px-3 py-2" value={selectedCache} onChange={(e) => setSelectedCache(e.target.value)}>
              {selectableCaches.map((cacheKey) => <option key={cacheKey}>{cacheKey}</option>)}
            </select>
            <select className="rounded-full border border-slate-300 px-3 py-2" value={budgetMetric} onChange={(e) => setBudgetMetric(e.target.value as "AUROC" | "SelAcc")}>
              {budgetMetricOptions.map((item) => <option key={item}>{item}</option>)}
            </select>
          </div>
        </div>
        <div className="mt-5">
          <MetricsLineChart data={budgetRows} xKey="budget" yKeys={compareCards.map((card) => card.title)} height={360} />
        </div>
      </section>

      <section className="card">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div>
            <h2 className="section-title">rho_tail 与 -A_accel 的交互可视化</h2>
            <p className="section-subtitle">保留 hover 交互，让观众直接看到不同 benchmark 的 dynamics 分布，而不是只看静态结论。</p>
          </div>
        </div>
        <div className="mt-5">
          <DynamicsInteractiveViewer dataPath="/data/dynamics_interactive.json" chartKey="phase_plane" />
        </div>
      </section>

      <section className="card">
        <h2 className="section-title">被放弃或退居次要位置的方案</h2>
        <p className="section-subtitle">不是这些方案毫无价值，而是它们没有成为“最适合公开展示的最终主方案”。</p>
        <div className="mt-5 grid gap-4 lg:grid-cols-3">
          {droppedStrategies.map((item) => (
            <article key={item.id} className={`rounded-2xl border p-4 ${item.accent}`}>
              <div className="text-base font-semibold text-slate-950">{item.title}</div>
              <p className="mt-3 text-sm leading-6 text-slate-700">{item.reason}</p>
            </article>
          ))}
        </div>
      </section>

      <DataState loading={loading || timelineState.loading} error={error || timelineState.error}>
        <section className="card">
          <h2 className="section-title">Early Stop 方法演化时间线</h2>
          <p className="section-subtitle">从 confidence baseline 到 local dynamics 审计，再到最后的 conservative router。</p>
          <div className="mt-5">
            <TimelineView nodes={(timelineState.data?.nodes ?? []).filter((node) => node.task === "early_stop" || node.task === "cross")} />
          </div>
        </section>
      </DataState>
    </div>
  );
}
