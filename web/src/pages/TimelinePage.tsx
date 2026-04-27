import { DataState } from "../components/DataState";
import { TimelineView } from "../components/TimelineView";
import { useJsonData } from "../hooks/useJsonData";
import type { ResearchTimeline } from "../types/research";

export function TimelinePage() {
  const state = useJsonData<ResearchTimeline>("/data/research_timeline.json");

  return (
    <div className="space-y-6">
      <section className="hero-card">
        <div className="text-xs font-semibold uppercase tracking-[0.18em] text-violet-700">Unified Timeline</div>
        <h1 className="mt-2 text-3xl font-bold tracking-tight text-slate-950">把 Early Stop 和 Best-of-N 合并成一条完整演化线</h1>
        <p className="mt-4 max-w-4xl text-sm leading-7 text-slate-700">
          这条时间线把两类实验统一在同一个叙事里：先建立 baseline，再做局部 patch，最后保留那些真正能在稳定性、解释性和最终展示效果上同时成立的方法。
        </p>
        <div className="mt-5 flex flex-wrap gap-2 text-xs text-slate-700">
          <span className="metric-chip">蓝色 = Best-of-N</span>
          <span className="metric-chip">绿色 = Early Stop</span>
          <span className="metric-chip">紫色 = Cross insight</span>
        </div>
      </section>

      <DataState loading={state.loading} error={state.error}>
        <section className="card">
          <h2 className="section-title">方法演化总览</h2>
          <p className="section-subtitle">从 baseline 到最终方案的过渡，同时保留关键分叉和被放弃路线。</p>
          <div className="mt-5">
            <TimelineView nodes={state.data?.nodes ?? []} />
          </div>
        </section>
      </DataState>
    </div>
  );
}
