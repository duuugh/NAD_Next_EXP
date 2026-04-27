import { Bar, BarChart, CartesianGrid, Cell, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

type Props = {
  data: Array<Record<string, number | string>>;
  xKey: string;
  yKeys: string[];
  height?: number;
};

const colors = ["#2563eb", "#10b981", "#7c3aed", "#f59e0b", "#ef4444", "#0891b2"];

export function MetricsBarChart({ data, xKey, yKeys, height = 320 }: Props) {
  if (!data.length || !yKeys.length) {
    return <div className="panel-muted text-sm text-slate-500">暂无可绘制柱状图数据</div>;
  }

  return (
    <div className="panel-muted" style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 8, right: 12, left: -10, bottom: 8 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey={xKey} tick={{ fontSize: 12 }} />
          <YAxis tick={{ fontSize: 12 }} />
          <Tooltip />
          <Legend />
          {yKeys.map((key, idx) => (
            <Bar key={key} dataKey={key} radius={[8, 8, 0, 0]} fill={colors[idx % colors.length]}>
              {data.map((_, rowIdx) => (
                <Cell key={`${key}-${rowIdx}`} fill={colors[idx % colors.length]} />
              ))}
            </Bar>
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
