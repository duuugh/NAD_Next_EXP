import type { ReactNode } from "react";

type Props = {
  loading: boolean;
  error: string | null;
  emptyText?: string;
  children: ReactNode;
};

export function DataState({ loading, error, emptyText = "数据缺失 / 待补充", children }: Props) {
  if (loading) {
    return <div className="card text-sm text-slate-500">正在加载数据...</div>;
  }
  if (error) {
    return <div className="card text-sm text-rose-600">加载失败：{error}（{emptyText}）</div>;
  }
  return <>{children}</>;
}
