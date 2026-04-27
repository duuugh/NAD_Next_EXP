import { memo, useCallback, useMemo, useState } from "react";
import type { VisualAsset } from "../types/research";
import { ActivationTrajectoryViewer } from "./ActivationTrajectoryViewer";
import { DynamicsInteractiveViewer } from "./DynamicsInteractiveViewer";
import { InteractiveVisualModal } from "./InteractiveVisualModal";

type Props = {
  visuals: VisualAsset[];
};

type ModalState =
  | { visual: VisualAsset; mode: "interactive" | "static" }
  | null;

type VisualCardProps = {
  visual: VisualAsset;
  staticExpanded: boolean;
  onToggleStatic: (visualId: string) => void;
  onOpenStatic: (visual: VisualAsset) => void;
  onOpenInteractive: (visual: VisualAsset) => void;
};

const VisualCard = memo(function VisualCard({
  visual,
  staticExpanded,
  onToggleStatic,
  onOpenStatic,
  onOpenInteractive,
}: VisualCardProps) {
  const isInteractive = Boolean(visual.interactiveKind && visual.interactiveDataPath && visual.interactiveKey);
  const visualId = `${visual.publicPath}-${visual.title}`;

  return (
    <div className="overflow-hidden rounded-xl border border-slate-200 bg-slate-50 transition-shadow hover:shadow-sm">
      <figure>
        <div className="border-b border-slate-200 bg-slate-100 px-3 py-2">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="text-xs font-medium uppercase tracking-wide text-slate-500">{isInteractive ? "Static PNG" : "Image"}</div>
            <div className="flex flex-wrap items-center gap-2">
              {isInteractive ? (
                <button
                  type="button"
                  className="rounded-md border border-slate-300 bg-white px-2.5 py-1 text-xs text-slate-700 transition hover:bg-slate-100 active:scale-95"
                  onClick={() => onToggleStatic(visualId)}
                >
                  {staticExpanded ? "隐藏原图" : "查看原图"}
                </button>
              ) : null}
              <button
                type="button"
                className="rounded-md border border-slate-300 bg-white px-2.5 py-1 text-xs text-slate-700 transition hover:bg-slate-100 active:scale-95"
                onClick={() => onOpenStatic(visual)}
              >
                点开大图
              </button>
            </div>
          </div>
          {isInteractive ? <div className="mt-1 text-xs text-slate-600">默认先看交互版；原 PNG 可按需展开对照。</div> : null}
        </div>

        {!isInteractive || staticExpanded ? (
          <div className="relative bg-slate-100">
            <button
              type="button"
              className="block w-full cursor-zoom-in transition-opacity hover:opacity-95"
              onClick={() => onOpenStatic(visual)}
            >
              <img
                className="h-auto w-full select-none object-cover"
                src={visual.publicPath}
                alt={visual.title}
                loading="lazy"
                decoding="async"
                draggable={false}
              />
            </button>
            {isInteractive ? (
              <div className="pointer-events-none absolute inset-x-0 bottom-0 bg-slate-900/75 px-3 py-2 text-xs text-white">
                这是原始静态图；点击图片可查看大图，下方为交互重绘版
              </div>
            ) : null}
          </div>
        ) : null}
        <figcaption className="space-y-1 px-3 py-2">
          <div className="flex items-center justify-between gap-3">
            <div className="text-sm font-medium text-slate-800">{visual.title}</div>
            {isInteractive ? (
              <button
                type="button"
                className="rounded-md border border-slate-300 bg-white px-2.5 py-1 text-xs text-slate-700 transition hover:bg-slate-100 active:scale-95"
                onClick={() => onOpenInteractive(visual)}
              >
                放大交互图
              </button>
            ) : null}
          </div>
          {visual.caption ? <div className="text-xs text-slate-600">{visual.caption}</div> : null}
        </figcaption>
      </figure>

      {isInteractive ? (
        <div className="border-t border-slate-200 bg-white p-3">
          <div className="mb-2 text-xs font-medium uppercase tracking-wide text-slate-500">Interactive View</div>
          {visual.interactiveKind === "activation" && visual.interactiveDataPath && visual.interactiveKey ? (
            <ActivationTrajectoryViewer dataPath={visual.interactiveDataPath} caseKey={visual.interactiveKey} />
          ) : null}
          {visual.interactiveKind === "dynamics" && visual.interactiveDataPath && visual.interactiveKey ? (
            <DynamicsInteractiveViewer dataPath={visual.interactiveDataPath} chartKey={visual.interactiveKey} />
          ) : null}
        </div>
      ) : null}
    </div>
  );
});

export function VisualGallery({ visuals }: Props) {
  const [modalState, setModalState] = useState<ModalState>(null);
  const [expandedStatics, setExpandedStatics] = useState<Record<string, boolean>>({});

  const openStaticModal = useCallback((visual: VisualAsset) => {
    setModalState({ visual, mode: "static" });
  }, []);

  const openInteractiveModal = useCallback((visual: VisualAsset) => {
    setModalState({ visual, mode: "interactive" });
  }, []);

  const closeModal = useCallback(() => {
    setModalState(null);
  }, []);

  const toggleStatic = useCallback((visualId: string) => {
    setExpandedStatics((current) => ({
      ...current,
      [visualId]: !current[visualId],
    }));
  }, []);

  const expandedMap = useMemo(() => expandedStatics, [expandedStatics]);

  if (!visuals.length) {
    return null;
  }

  return (
    <div className="mt-4 grid gap-4">
      {visuals.map((visual) => {
        const visualId = `${visual.publicPath}-${visual.title}`;
        return (
          <VisualCard
            key={visualId}
            visual={visual}
            staticExpanded={expandedMap[visualId] ?? false}
            onToggleStatic={toggleStatic}
            onOpenStatic={openStaticModal}
            onOpenInteractive={openInteractiveModal}
          />
        );
      })}

      {modalState ? (
        <InteractiveVisualModal
          visual={modalState.visual}
          mode={modalState.mode}
          onClose={closeModal}
        />
      ) : null}
    </div>
  );
}
