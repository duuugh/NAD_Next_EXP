import { Link } from "react-router-dom";
import { Badge } from "../components/Badge";
import { DataState } from "../components/DataState";
import { SummaryStatGrid } from "../components/SummaryStatGrid";
import { TimelineView } from "../components/TimelineView";
import { useJsonData } from "../hooks/useJsonData";
import type { ExportedCards, ResearchTimeline } from "../types/research";

function fallbackFindings() {
  return [
    "Best-of-N 最终不是靠大一统重写，而是靠 AIME-only 的 very small patch 稳定胜出。",
    "Early Stop 最有效的不是全局启用 dynamics，而是 benchmark-selective conservative router。",
    "activation / dynamics 信号都有价值，但更适合作为局部修正，而不是无差别全局替换。",
    "真正能上主页的方案，都同时满足：可解释、改动小、外部风险可控。",
  ];
}

export function HomePage() {
  const early = useJsonData<ExportedCards>("/data/early_stop_cards.json");
  const bon = useJsonData<ExportedCards>("/data/best_of_n_cards.json");
  const timeline = useJsonData<ResearchTimeline>("/data/research_timeline.json");

  const loading = early.loading || bon.loading || timeline.loading;
  const error = early.error || bon.error || timeline.error;

  const earlyBest = early.data?.cards.find((card) => card.id === early.data?.finalBestId);
  const bonBest = bon.data?.cards.find((card) => card.id === bon.data?.finalBestId);
  const findings = [...(early.data?.conclusions ?? []), ...(bon.data?.conclusions ?? [])];
  const coreFindings = findings.length ? findings.slice(0, 4) : fallbackFindings();

  return (
    <div className="space-y-8">
      <section className="hero-card overflow-hidden bg-gradient-to-br from-slate-950 via-slate-900 to-slate-800 text-white">
        <div className="grid gap-8 lg:grid-cols-[1.3fr_0.9fr]">
          <div>
            <div className="inline-flex rounded-full border border-white/15 bg-white/10 px-3 py-1 text-xs font-semibold uppercase tracking-[0.18em] text-slate-100">
              Research Overview
            </div>
            <h1 className="mt-5 text-4xl font-bold tracking-tight lg:text-5xl">把两条主线讲清楚：Best-of-N 和 Early Stop</h1>
            <p className="mt-4 max-w-3xl text-base leading-7 text-slate-200">
              这个站点只保留最重要的结论：哪个方案最终赢了、它为什么赢、哪些更复杂或更激进的路线最后没有成为主方案。
            </p>
            <div className="mt-6 flex flex-wrap gap-3">
              <Link className="rounded-full bg-emerald-400 px-5 py-2.5 text-sm font-semibold text-slate-950" to="/early-stop">看 Early Stop 最佳方案</Link>
              <Link className="rounded-full bg-blue-500 px-5 py-2.5 text-sm font-semibold text-white" to="/best-of-n">看 Best-of-N 最佳方案</Link>
              <Link className="rounded-full border border-white/15 bg-white/5 px-5 py-2.5 text-sm font-semibold text-white" to="/timeline">看方法演化时间线</Link>
            </div>
          </div>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-1">
            <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
              <div className="text-xs uppercase tracking-[0.16em] text-emerald-200">Final Early Stop</div>
              <div className="mt-2 text-lg font-semibold">{earlyBest?.title ?? "early_stop_dynamics_router_conservative_submit"}</div>
              <div className="mt-2 text-sm leading-6 text-slate-200">只在稳定 benchmark 上启用 dynamics plugin，其余 benchmark 保守禁用。</div>
            </div>
            <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
              <div className="text-xs uppercase tracking-[0.16em] text-blue-200">Final Best-of-N</div>
              <div className="mt-2 text-lg font-semibold">{bonBest?.title ?? "nad_mixed_v2_aime_top2_gap1e3_logprob"}</div>
              <div className="mt-2 text-sm leading-6 text-slate-200">只修 AIME 的 near-tie cases，保留全局 baseline 的稳定结构。</div>
            </div>
          </div>
        </div>
      </section>

      <DataState loading={loading} error={error}>
        <section className="grid gap-5 lg:grid-cols-2">
          <article className="hero-card gradient-border-emerald highlight-ring">
            <div className="flex items-start justify-between gap-3">
              <div>
                <div className="text-xs font-semibold uppercase tracking-[0.18em] text-emerald-700">Early Stop 最佳方案</div>
                <h2 className="mt-2 text-3xl font-bold tracking-tight text-slate-950">{earlyBest?.title ?? "early_stop_dynamics_router_conservative_submit"}</h2>
              </div>
              {earlyBest ? <Badge status={earlyBest.status} /> : null}
            </div>
            <p className="mt-4 text-sm leading-7 text-slate-700">{earlyBest?.shortDescription}</p>
            <div className="mt-5">
              <SummaryStatGrid
                columns={2}
                metrics={[
                  { label: "AUC-AUROC", value: earlyBest?.overall?.["AUC-AUROC"], accent: "emerald" },
                  { label: "AUC-SelAcc", value: earlyBest?.overall?.["AUC-SelAcc"], accent: "emerald" },
                  { label: "AUROC@100%", value: earlyBest?.overall?.["AUROC@100%"], accent: "slate" },
                  { label: "Stop@100%", value: earlyBest?.overall?.["Stop@100%"], accent: "slate" },
                ]}
              />
            </div>
          </article>

          <article className="hero-card gradient-border-blue">
            <div className="flex items-start justify-between gap-3">
              <div>
                <div className="text-xs font-semibold uppercase tracking-[0.18em] text-blue-700">Best-of-N 最佳方案</div>
                <h2 className="mt-2 text-3xl font-bold tracking-tight text-slate-950">{bonBest?.title ?? "nad_mixed_v2_aime_top2_gap1e3_logprob"}</h2>
              </div>
              {bonBest ? <Badge status={bonBest.status} /> : null}
            </div>
            <p className="mt-4 text-sm leading-7 text-slate-700">{bonBest?.shortDescription}</p>
            <div className="mt-5">
              <SummaryStatGrid
                columns={2}
                metrics={[
                  { label: "mean_score", value: bonBest?.overall?.mean_score, accent: "blue" },
                  { label: "score_range", value: bonBest?.overall?.score_range, accent: "blue" },
                  { label: "Patch Scope", value: "AIME 4 caches", accent: "slate" },
                  { label: "Trigger", value: "top2 gap ≤ 1e-3", accent: "slate" },
                ]}
              />
            </div>
          </article>
        </section>

        <section className="card">
          <div className="flex flex-wrap items-end justify-between gap-3">
            <div>
              <h2 className="section-title">核心发现</h2>
              <p className="section-subtitle">只保留最值得观众记住的结论；技术细节移到折叠区域，页面先讲“最后什么成立”。</p>
            </div>
          </div>
          <div className="mt-5 grid gap-4 md:grid-cols-2 xl:grid-cols-4">
            {coreFindings.map((item, idx) => (
              <article key={item} className={`rounded-2xl border p-4 ${idx % 2 === 0 ? "border-emerald-200 bg-emerald-50/70" : "border-blue-200 bg-blue-50/70"}`}>
                <div className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">Finding {idx + 1}</div>
                <p className="mt-3 text-sm leading-6 text-slate-800">{item}</p>
              </article>
            ))}
          </div>
        </section>

        <section className="card details-reset">
          <div className="flex flex-wrap items-end justify-between gap-3">
            <div>
              <h2 className="section-title">方法演化总览</h2>
              <p className="section-subtitle">把 Early Stop 和 Best-of-N 放在同一条叙事线上：baseline → 局部修正 → 最终保留方案。</p>
            </div>
            <Link className="rounded-full bg-slate-950 px-4 py-2 text-sm font-semibold text-white" to="/timeline">打开完整时间线</Link>
          </div>
          <div className="mt-5">
            <TimelineView nodes={(timeline.data?.nodes ?? []).slice(0, 6)} />
          </div>
        </section>

        <section className="grid gap-4 lg:grid-cols-2">
          <details className="card details-reset group">
            <summary className="flex cursor-pointer items-center justify-between gap-3 text-base font-semibold text-slate-950">
              <span>展开 Best-of-N 技术细节</span>
              <span className="text-sm text-slate-500 group-open:rotate-180 transition">⌄</span>
            </summary>
            <div className="mt-4 space-y-3 text-sm leading-7 text-slate-700">
              <p>最终保留的是 small patch 思路：只在 AIME-only near-tie cases 上动手，其余 cache 不强行重写。</p>
              <p>被放弃的路线包括 activation 主导、tail warning veto、更复杂的 cluster router，以及没有稳定转化成最终 top1 的 local head 路线。</p>
            </div>
          </details>

          <details className="card details-reset group">
            <summary className="flex cursor-pointer items-center justify-between gap-3 text-base font-semibold text-slate-950">
              <span>展开 Early Stop 技术细节</span>
              <span className="text-sm text-slate-500 group-open:rotate-180 transition">⌄</span>
            </summary>
            <div className="mt-4 space-y-3 text-sm leading-7 text-slate-700">
              <p>最终不是“全局 dynamics”，而是 conservative router：只在 DS-R1/aime24、aime25、hmmt25 上启用最稳定的 dynamics 规则。</p>
              <p>rho_tail 和 -A_accel 在不同 benchmark 上效果并不一致，因此必须按 benchmark 决策，而不是一把梭全开。</p>
            </div>
          </details>
        </section>
      </DataState>
    </div>
  );
}
