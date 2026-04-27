import { useEffect, useMemo, useState } from "react";
import { CartesianGrid, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from "recharts";

type ActivationPoint = {
  token: number;
  neurons: number;
  entropySum: number;
  sampleId: number;
  highlightTags: string[];
};

type ActivationCase = {
  problemId: string;
  note?: string;
  runs: Array<{
    sampleId: number;
    pointCount: number;
    maxToken: number;
    maxNeurons: number;
    highlightTags: string[];
    points: ActivationPoint[];
  }>;
  highlighted: Array<{
    label: string;
    sampleId: number;
    isCorrect?: boolean;
  }>;
};

type ActivationPayload = {
  generatedAt: string;
  cases: Record<string, ActivationCase>;
};

type Props = {
  dataPath: string;
  caseKey: string;
};

function resolveDataPath(path: string): string {
  if (/^https?:\/\//.test(path)) {
    return path;
  }
  const normalized = path.startsWith("/") ? path.slice(1) : path;
  return new URL(normalized, window.location.href).toString();
}

function highlightColor(tags: string[]): string {
  if (tags.includes("activation_tiebreak")) return "#dc2626";
  if (tags.includes("medoid")) return "#2563eb";
  return "rgba(100,116,139,0.25)";
}

const legendItems = [
  { label: "Activation tie-break 选中", color: "#dc2626" },
  { label: "Medoid 选中", color: "#2563eb" },
  { label: "其他 sample", color: "#94a3b8" },
];

export function ActivationTrajectoryViewer({ dataPath, caseKey }: Props) {
  const [payload, setPayload] = useState<ActivationPayload | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    setPayload(null);
    setError(null);
    fetch(resolveDataPath(dataPath))
      .then(async (resp) => {
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const json = (await resp.json()) as ActivationPayload;
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

  const caseData = payload?.cases?.[caseKey];
  const runs = useMemo(() => {
    if (!caseData) return [];
    return [...caseData.runs].sort((left, right) => {
      const leftPriority = left.highlightTags.length ? 1 : 0;
      const rightPriority = right.highlightTags.length ? 1 : 0;
      if (leftPriority !== rightPriority) return leftPriority - rightPriority;
      return left.sampleId - right.sampleId;
    });
  }, [caseData]);

  if (error) {
    return <div className="rounded-xl border border-rose-200 bg-rose-50 p-4 text-sm text-rose-700">交互数据加载失败：{error}</div>;
  }
  if (!caseData) {
    return <div className="rounded-xl border border-slate-200 bg-slate-50 p-4 text-sm text-slate-500">正在加载 activation 交互数据...</div>;
  }

  return (
    <div className="space-y-3">
      <div className="space-y-2 text-sm text-slate-600">
        <div className="font-medium text-slate-800">题目 {caseData.problemId}</div>
        {caseData.note ? <div className="mt-1">{caseData.note}</div> : null}
        <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-xs leading-5 text-slate-700">
          上方小图是仓库里的原始 PNG；下方大图是根据同题原始轨迹数据重绘的交互版，方便悬停查看 sample、token 与累计 neuron 数值。
        </div>
        <div className="mt-1">共 {caseData.runs.length} 条轨迹。</div>
      </div>

      <div className="flex flex-wrap gap-2 text-xs text-slate-700">
        {legendItems.map((item) => (
          <div key={item.label} className="flex items-center gap-2 rounded-full border border-slate-200 bg-white px-3 py-1.5">
            <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ backgroundColor: item.color }} />
            <span>{item.label}</span>
          </div>
        ))}
      </div>

      {caseData.highlighted.length ? (
        <div className="flex flex-wrap gap-2 text-xs">
          {caseData.highlighted.map((item) => (
            <span key={`${item.label}-${item.sampleId}`} className="rounded-full bg-slate-100 px-2.5 py-1 text-slate-700">
              {item.label}: sample {item.sampleId}{typeof item.isCorrect === "boolean" ? ` · ${item.isCorrect ? "correct" : "incorrect"}` : ""}
            </span>
          ))}
        </div>
      ) : null}

      <div className="h-[70vh] min-h-[420px] rounded-xl border border-slate-200 bg-white p-3">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 16, right: 18, left: 8, bottom: 18 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" dataKey="token" name="Token Position" tick={{ fontSize: 12 }} />
            <YAxis type="number" dataKey="neurons" name="Cumulative Unique Neurons" tick={{ fontSize: 12 }} />
            <Tooltip
              cursor={{ strokeDasharray: "3 3" }}
              formatter={(value: number, name: string) => [Number(value).toFixed(2), name]}
              labelFormatter={() => ""}
              content={({ active, payload: tooltipPayload }) => {
                if (!active || !tooltipPayload?.length) return null;
                const point = tooltipPayload[0].payload as ActivationPoint;
                return (
                  <div className="rounded border border-slate-200 bg-white p-2 text-xs shadow">
                    <div className="font-semibold text-slate-900">sample {point.sampleId}</div>
                    <div>Token Position: {point.token}</div>
                    <div>Cumulative Neurons: {point.neurons}</div>
                    <div>Entropy Sum: {point.entropySum.toFixed(4)}</div>
                    {point.highlightTags.length ? <div>Tags: {point.highlightTags.join(", ")}</div> : null}
                  </div>
                );
              }}
            />
            {runs.map((run) => (
              <Scatter
                key={`run-${run.sampleId}`}
                data={run.points}
                fill={highlightColor(run.highlightTags)}
                line={{ stroke: highlightColor(run.highlightTags), strokeWidth: run.highlightTags.length ? 2.4 : 1.1 }}
                shape={false}
                isAnimationActive={false}
              />
            ))}
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
