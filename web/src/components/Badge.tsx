import type { CardStatus } from "../types/research";
import { statusClass, statusLabel } from "../lib/status";

export function Badge({ status }: { status: CardStatus }) {
  return <span className={`badge ${statusClass[status]}`}>{statusLabel[status]}</span>;
}
