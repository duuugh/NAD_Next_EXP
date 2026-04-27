type MetricValue = number | string | boolean | null | undefined;

type Props = {
  metrics: Array<{
    label: string;
    value: MetricValue | undefined;
    hint?: string;
    accent?: "emerald" | "blue" | "violet" | "amber" | "slate";
  }>;
  columns?: 2 | 3 | 4;
  compact?: boolean;
};

function formatValue(value: MetricValue): string {
  if (value === null || value === undefined) {
    return "—";
  }
  if (typeof value === "number") {
    if (Math.abs(value) >= 1000) return value.toLocaleString();
    if (Math.abs(value) >= 10) return value.toFixed(2);
    return value.toFixed(4);
  }
  return String(value);
}

const accentClass = {
  emerald: "from-emerald-500/10 to-emerald-500/5 border-emerald-200",
  blue: "from-blue-500/10 to-blue-500/5 border-blue-200",
  violet: "from-violet-500/10 to-violet-500/5 border-violet-200",
  amber: "from-amber-500/10 to-amber-500/5 border-amber-200",
  slate: "from-slate-500/10 to-slate-500/5 border-slate-200",
};

export function SummaryStatGrid({ metrics, columns = 4, compact = false }: Props) {
  const gridClass = columns === 2 ? "md:grid-cols-2" : columns === 3 ? "md:grid-cols-3" : "md:grid-cols-4";
  const paddingClass = compact ? "p-3" : "p-4";
  const valueClass = compact ? "text-xl" : "text-2xl";

  return (
    <div className={`grid gap-3 ${gridClass}`}>
      {metrics.map((metric) => (
        <div
          key={metric.label}
          className={`rounded-2xl border bg-gradient-to-br ${paddingClass} ${accentClass[metric.accent ?? "slate"]}`}
        >
          <div className="text-xs font-medium uppercase tracking-[0.16em] text-slate-500">{metric.label}</div>
          <div className={`mt-2 font-semibold text-slate-950 ${valueClass}`}>{formatValue(metric.value)}</div>
          {metric.hint ? <div className="mt-1 text-xs leading-5 text-slate-600">{metric.hint}</div> : null}
        </div>
      ))}
    </div>
  );
}
