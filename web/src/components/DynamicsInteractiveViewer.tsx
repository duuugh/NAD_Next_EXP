import { useEffect, useMemo, useState } from "react";
import { CartesianGrid, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from "recharts";

type Row = Record<string, number | string | boolean | null>;

type ChartConfig = {
  title: string;
  description: string;
  xKey: string;
  yKey: string;
  rows: Row[];
};

type Payload = {
  generatedAt: string;
  charts: Record<string, ChartConfig>;
};

type Props = {
  dataPath: string;
  chartKey: string;
};

function resolveDataPath(path: string): string {
  if (/^https?:\/\//.test(path)) {
    return path;
  }
  const normalized = path.startsWith("/") ? path.slice(1) : path;
  return new URL(normalized, window.location.href).toString();
}

function chartColor(row: Row): string {
  if (row.cache_key === "DS-R1/aime25") return "#2563eb";
  if (row.cache_key === "DS-R1/hmmt25") return "#16a34a";
  if (row.cache_key === "DS-R1/gpqa") return "#d97706";
  if (row.cache_key === "DS-R1/lcb_v5") return "#7c3aed";
  return "#64748b";
}

const legendLabelMap: Record<string, string> = {
  "DS-R1/aime25": "DS-R1/aime25",
  "DS-R1/hmmt25": "DS-R1/hmmt25",
  "DS-R1/gpqa": "DS-R1/gpqa",
  "DS-R1/lcb_v5": "DS-R1/lcb_v5",
  other: "其他 benchmark / cache",
};

export function DynamicsInteractiveViewer({ dataPath, chartKey }: Props) {
  const [payload, setPayload] = useState<Payload | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    setPayload(null);
    setError(null);
    fetch(resolveDataPath(dataPath))
      .then(async (resp) => {
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const json = (await resp.json()) as Payload;
        if (alive) setPayload(json);
      })
      .catch((err: unknown) => {
        if (!alive) return;
        setError(err instanceof Error ? err.message : "unknown error");
      });
    return () => {
      alive = false;
    };
  }, [dataPath]);

  const chart = useMemo(() => payload?.charts?.[chartKey], [payload, chartKey]);
  const groupedRows = useMemo(() => {
    if (!chart) return [];
    const groups = new Map<string, Row[]>();
    for (const row of chart.rows) {
      const key = String(row.cache_key ?? "other");
      const existing = groups.get(key) ?? [];
      existing.push(row);
      groups.set(key, existing);
    }
    return Array.from(groups.entries());
  }, [chart]);

  if (error) {
    return <div className="rounded-xl border border-rose-200 bg-rose-50 p-4 text-sm text-rose-700">交互数据加载失败：{error}</div>;
  }
  if (!chart) {
    return <div className="rounded-xl border border-slate-200 bg-slate-50 p-4 text-sm text-slate-500">正在加载 dynamics 交互数据...</div>;
  }

  return (
    <div className="space-y-3">
      <div className="space-y-2">
        <div className="font-medium text-slate-900">{chart.title}</div>
        <div className="mt-1 text-sm text-slate-600">{chart.description}</div>
        <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-xs leading-5 text-slate-700">
          上方小图是原始导出的静态 PNG；下方大图是基于同类 dynamics 原始表格重绘的交互版，用于悬停查看具体 benchmark、problem 与数值。
        </div>
      </div>

      <div className="flex flex-wrap gap-2 text-xs text-slate-700">
        {groupedRows.map(([groupKey, rows]) => (
          <div key={groupKey} className="flex items-center gap-2 rounded-full border border-slate-200 bg-white px-3 py-1.5">
            <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ backgroundColor: chartColor(rows[0]) }} />
            <span>{legendLabelMap[groupKey] ?? groupKey}</span>
          </div>
        ))}
      </div>

      <div className="h-[70vh] min-h-[420px] rounded-xl border border-slate-200 bg-white p-3">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 16, right: 18, left: 8, bottom: 18 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" dataKey={chart.xKey} name={chart.xKey} tick={{ fontSize: 12 }} />
            <YAxis type="number" dataKey={chart.yKey} name={chart.yKey} tick={{ fontSize: 12 }} />
            <Tooltip
              cursor={{ strokeDasharray: "3 3" }}
              content={({ active, payload: tooltipPayload }) => {
                if (!active || !tooltipPayload?.length) return null;
                const row = tooltipPayload[0].payload as Row;
                return (
                  <div className="rounded border border-slate-200 bg-white p-2 text-xs shadow">
                    <div className="font-semibold text-slate-900">{String(row.cache_key ?? "dynamics run")}</div>
                    <div>{chart.xKey}: {String(row[chart.xKey])}</div>
                    <div>{chart.yKey}: {String(row[chart.yKey])}</div>
                    {row.problem_id !== undefined ? <div>problem_id: {String(row.problem_id)}</div> : null}
                    {row.run_id !== undefined ? <div>run_id: {String(row.run_id)}</div> : null}
                    {row.is_correct !== undefined ? <div>is_correct: {String(row.is_correct)}</div> : null}
                  </div>
                );
              }}
            />
            {groupedRows.map(([groupKey, rows]) => (
              <Scatter key={groupKey} data={rows} fill={chartColor(rows[0])} isAnimationActive={false} />
            ))}
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
