export type TaskType = "early_stop" | "best_of_n";

export type CardStatus = "final_best" | "important" | "promising" | "deprecated";

export type VisualKind = "activation" | "dynamics";

export type VisualAsset = {
  title: string;
  publicPath: string;
  caption?: string;
  interactiveKind?: VisualKind;
  interactiveDataPath?: string;
  interactiveKey?: string;
};

export type BudgetPoint = {
  budget: number;
  [key: string]: string | number | boolean | null | undefined;
};

export type MetricValue = string | number | boolean | null;

export type PerCacheMetrics = Record<string, MetricValue | BudgetPoint[]> & {
  by_budget?: BudgetPoint[];
  labeled?: boolean;
};

export type ExperimentCard = {
  id: string;
  task: TaskType;
  title: string;
  shortDescription: string;
  status: CardStatus;
  sourceFiles: string[];
  overall?: Record<string, MetricValue>;
  perCache?: Record<string, PerCacheMetrics>;
  notes?: string[];
  tags?: string[];
  timelineGroup?: string;
  visuals?: VisualAsset[];
};

export type ExportedCards = {
  generatedAt: string;
  task: TaskType;
  finalBestId?: string;
  highlightedIds: string[];
  cards: ExperimentCard[];
  metricOptions: string[];
  metricHints?: Record<string, string>;
  conclusions: string[];
};

export type TimelineNode = {
  id: string;
  task: "early_stop" | "best_of_n" | "cross";
  title: string;
  status: CardStatus;
  summary: string;
};

export type ResearchTimeline = {
  generatedAt: string;
  nodes: TimelineNode[];
};
