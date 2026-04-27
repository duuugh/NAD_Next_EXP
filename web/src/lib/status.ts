import type { CardStatus } from "../types/research";

export const statusLabel: Record<CardStatus, string> = {
  final_best: "最终最佳（Live）",
  important: "重点方案",
  promising: "潜力路线",
  deprecated: "探索/未采用",
};

export const statusClass: Record<CardStatus, string> = {
  final_best: "bg-emerald-100 text-emerald-700",
  important: "bg-blue-100 text-blue-700",
  promising: "bg-amber-100 text-amber-700",
  deprecated: "bg-slate-100 text-slate-600",
};
