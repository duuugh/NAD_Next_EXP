import { useEffect } from "react";
import { motion } from "framer-motion";
import type { VisualAsset } from "../types/research";
import { ActivationTrajectoryViewer } from "./ActivationTrajectoryViewer";
import { DynamicsInteractiveViewer } from "./DynamicsInteractiveViewer";

type Props = {
  visual: VisualAsset;
  onClose: () => void;
  mode?: "interactive" | "static";
};

export function InteractiveVisualModal({ visual, onClose, mode = "interactive" }: Props) {
  useEffect(() => {
    const previousOverflow = document.body.style.overflow;
    const previousPaddingRight = document.body.style.paddingRight;
    const scrollbarWidth = window.innerWidth - document.documentElement.clientWidth;

    document.body.style.overflow = "hidden";
    if (scrollbarWidth > 0) {
      document.body.style.paddingRight = `${scrollbarWidth}px`;
    }

    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onClose();
      }
    };

    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = previousOverflow;
      document.body.style.paddingRight = previousPaddingRight;
    };
  }, [onClose]);

  return (
    <motion.div
      key={`${visual.publicPath}-${mode}`}
      className="fixed inset-0 z-[100] flex items-center justify-center bg-slate-950/60 p-3 backdrop-blur-sm md:p-4"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.12, ease: "easeOut" }}
    >
      <motion.div
        className="max-h-[96vh] w-full max-w-6xl overflow-auto rounded-2xl border border-slate-200 bg-white p-4 shadow-2xl md:p-5"
        onClick={(event) => event.stopPropagation()}
        initial={{ opacity: 0, scale: 0.992, y: 6 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        transition={{ duration: 0.14, ease: "easeOut" }}
      >
        <div className="mb-4 flex items-start justify-between gap-4">
          <div>
            <h3 className="text-lg font-semibold text-slate-900">{visual.title}</h3>
            {visual.caption ? <p className="mt-1 text-sm text-slate-600">{visual.caption}</p> : null}
            <p className="mt-2 text-xs text-slate-500">按 `Esc` 或点击遮罩可关闭</p>
          </div>
          <button
            type="button"
            className="shrink-0 rounded-full border border-slate-300 px-3 py-1.5 text-sm text-slate-700 transition hover:bg-slate-50 active:scale-95"
            onClick={onClose}
            aria-label="关闭弹窗"
          >
            关闭
          </button>
        </div>

        {mode === "static" ? (
          <div className="overflow-hidden rounded-xl border border-slate-200 bg-slate-50">
            <img
              className="h-auto max-h-[82vh] w-full select-none object-contain"
              src={visual.publicPath}
              alt={visual.title}
              loading="eager"
              decoding="async"
              draggable={false}
            />
          </div>
        ) : null}

        {mode === "interactive" && visual.interactiveKind === "activation" && visual.interactiveDataPath && visual.interactiveKey ? (
          <ActivationTrajectoryViewer dataPath={visual.interactiveDataPath} caseKey={visual.interactiveKey} />
        ) : null}
        {mode === "interactive" && visual.interactiveKind === "dynamics" && visual.interactiveDataPath && visual.interactiveKey ? (
          <DynamicsInteractiveViewer dataPath={visual.interactiveDataPath} chartKey={visual.interactiveKey} />
        ) : null}
      </motion.div>
    </motion.div>
  );
}
