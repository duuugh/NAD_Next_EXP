import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

type Props = {
  data: Array<Record<string, number | string>>;
  xKey: string;
  yKeys: string[];
  height?: number;
};

const colors = ["#2563eb", "#10b981", "#7c3aed", "#f59e0b", "#ef4444", "#0891b2"];

export function MetricsLineChart({ data, xKey, yKeys, height = 320 }: Props) {
  if (!data.length || !yKeys.length) {
    return <div className="panel-muted text-sm text-slate-500">暂无可绘制曲线数据</div>;
  }

  return (
    <div className="panel-muted" style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 8, right: 12, left: -14, bottom: 8 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey={xKey} tick={{ fontSize: 12 }} />
          <YAxis tick={{ fontSize: 12 }} />
          <Tooltip />
          <Legend />
          {yKeys.map((key, idx) => (
            <Line
              key={key}
              type="monotone"
              dataKey={key}
              stroke={colors[idx % colors.length]}
              strokeWidth={2.5}
              dot={{ r: 2 }}
              activeDot={{ r: 5 }}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
