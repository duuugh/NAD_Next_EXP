import { DataState } from "../components/DataState";
import { useJsonData } from "../hooks/useJsonData";

type DataIndex = {
  generatedAt: string;
  groups: Array<{
    name: string;
    description: string;
    files: Array<{ label: string; originalPath: string; publicPath: string }>;
  }>;
};

export function DataPage() {
  const state = useJsonData<DataIndex>("/data/data_index.json");

  return (
    <div className="space-y-4">
      <section className="card">
        <h1 className="text-2xl font-bold">数据与下载</h1>
        <p className="mt-2 text-sm text-slate-700">按 Early Stop / Best-of-N 分类列出关键 JSON / notes / report，支持本地下载。</p>
      </section>

      <DataState loading={state.loading} error={state.error}>
        {(state.data?.groups ?? []).map((group) => (
          <section key={group.name} className="card">
            <h2 className="text-lg font-semibold">{group.name}</h2>
            <p className="mt-1 text-sm text-slate-600">{group.description}</p>
            <ul className="mt-3 space-y-2 text-sm">
              {group.files.map((file) => (
                <li key={`${group.name}-${file.publicPath}`} className="flex items-center justify-between rounded border border-slate-200 px-3 py-2">
                  <div>
                    <div className="font-medium">{file.label}</div>
                    <div className="text-xs text-slate-500">{file.originalPath}</div>
                  </div>
                  <a className="rounded bg-slate-900 px-3 py-1.5 text-xs text-white" href={file.publicPath} download>下载</a>
                </li>
              ))}
            </ul>
          </section>
        ))}
      </DataState>
    </div>
  );
}
