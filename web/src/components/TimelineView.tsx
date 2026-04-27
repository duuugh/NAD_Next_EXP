import { motion } from "framer-motion";
import type { TimelineNode } from "../types/research";
import { Badge } from "./Badge";

type Props = {
  nodes: TimelineNode[];
};

const taskMeta = {
  best_of_n: {
    label: "Best-of-N",
    accent: "border-blue-200 bg-blue-50/70",
    dot: "bg-blue-600",
    text: "text-blue-700",
  },
  early_stop: {
    label: "Early Stop",
    accent: "border-emerald-200 bg-emerald-50/70",
    dot: "bg-emerald-600",
    text: "text-emerald-700",
  },
  cross: {
    label: "Cross Insight",
    accent: "border-violet-200 bg-violet-50/70",
    dot: "bg-violet-600",
    text: "text-violet-700",
  },
} as const;

export function TimelineView({ nodes }: Props) {
  if (!nodes.length) {
    return <div className="panel-muted text-sm text-slate-500">时间线数据缺失 / 待补充</div>;
  }

  return (
    <div className="relative space-y-4 before:absolute before:left-4 before:top-2 before:h-[calc(100%-1rem)] before:w-px before:bg-slate-200 md:before:left-6">
      {nodes.map((node, idx) => {
        const meta = taskMeta[node.task];
        return (
          <motion.div
            key={node.id}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: Math.min(idx * 0.05, 0.35) }}
            className="relative pl-10 md:pl-14"
          >
            <div className={`absolute left-[9px] top-5 h-3.5 w-3.5 rounded-full ring-4 ring-white md:left-[17px] ${meta.dot}`} />
            <div className={`rounded-2xl border p-4 shadow-sm ${meta.accent}`}>
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <div className={`text-xs font-semibold uppercase tracking-[0.18em] ${meta.text}`}>{meta.label}</div>
                  <div className="mt-1 text-lg font-semibold text-slate-950">{node.title}</div>
                  <div className="mt-2 text-sm leading-6 text-slate-700">{node.summary}</div>
                </div>
                <Badge status={node.status} />
              </div>
            </div>
          </motion.div>
        );
      })}
    </div>
  );
}
