import { Badge, CodeLink } from "@/components/ui/TacticalUI";
import { BottomSheet } from "@/components/layout/BottomSheet";
import type { ExperimentRun } from "@/lib/experiments";

interface RunInspectorProps {
  run: ExperimentRun | null;
  isOpen: boolean;
  onClose: () => void;
}

function formatTimestamp(value: string) {
  return new Intl.DateTimeFormat("en-US", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(value));
}

export function RunInspector({ run, isOpen, onClose }: RunInspectorProps) {
  return (
    <BottomSheet isOpen={isOpen && run !== null} onClose={onClose} title={run?.label}>
      {run ? (
        <div className="space-y-8">
          <div className="flex flex-wrap items-center gap-3">
            <Badge variant="gold">{run.modelName}</Badge>
            <Badge variant="green">{run.datasetName}</Badge>
            <span className="font-sans text-sm text-text-dim">
              Updated {formatTimestamp(run.updatedAt)}
            </span>
          </div>

          <div className="grid gap-6 md:grid-cols-3">
            <div className="rounded-xl border border-border-light bg-bg-panel p-4">
              <p className="font-sans text-[10px] uppercase tracking-widest text-text-dim">
                Metric Keys
              </p>
              <p className="mt-3 font-serif text-3xl text-text-main">
                {Object.keys(run.metrics).length}
              </p>
            </div>
            <div className="rounded-xl border border-border-light bg-bg-panel p-4">
              <p className="font-sans text-[10px] uppercase tracking-widest text-text-dim">
                Prediction Rows
              </p>
              <p className="mt-3 font-serif text-3xl text-text-main">
                {run.predictions.totalCount}
              </p>
            </div>
            <div className="rounded-xl border border-border-light bg-bg-panel p-4">
              <p className="font-sans text-[10px] uppercase tracking-widest text-text-dim">
                Importance Rows
              </p>
              <p className="mt-3 font-serif text-3xl text-text-main">
                {run.featureImportance.totalCount}
              </p>
            </div>
          </div>

          <div className="flex flex-wrap gap-3">
            <CodeLink href="#metrics">OPEN_METRICS</CodeLink>
            <CodeLink href="#predictions">OPEN_PREDICTIONS</CodeLink>
            <CodeLink href="#importance">OPEN_IMPORTANCE</CodeLink>
            <CodeLink href="#comparison">OPEN_COMPARISON</CodeLink>
          </div>

          <div className="space-y-3">
            <p className="font-sans text-[10px] uppercase tracking-widest text-text-dim">
              Run Metadata
            </p>
            <pre className="overflow-x-auto rounded-xl border border-border-light bg-bg-base p-4 font-mono text-xs leading-relaxed text-text-dim">
              {JSON.stringify(run.metadata, null, 2)}
            </pre>
          </div>
        </div>
      ) : null}
    </BottomSheet>
  );
}
