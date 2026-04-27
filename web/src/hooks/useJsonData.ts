import { useEffect, useState } from "react";

type LoadState<T> = {
  loading: boolean;
  data: T | null;
  error: string | null;
};

function resolveDataPath(path: string): string {
  if (/^https?:\/\//.test(path)) {
    return path;
  }
  const normalized = path.startsWith("/") ? path.slice(1) : path;
  return new URL(normalized, window.location.href).toString();
}

export function useJsonData<T>(path: string): LoadState<T> {
  const [state, setState] = useState<LoadState<T>>({ loading: true, data: null, error: null });

  useEffect(() => {
    let alive = true;
    setState({ loading: true, data: null, error: null });
    fetch(resolveDataPath(path))
      .then(async (resp) => {
        if (!resp.ok) {
          throw new Error(`HTTP ${resp.status}`);
        }
        const json = (await resp.json()) as T;
        if (alive) {
          setState({ loading: false, data: json, error: null });
        }
      })
      .catch((err: unknown) => {
        if (!alive) {
          return;
        }
        const message = err instanceof Error ? err.message : "unknown error";
        setState({ loading: false, data: null, error: message });
      });

    return () => {
      alive = false;
    };
  }, [path]);

  return state;
}
