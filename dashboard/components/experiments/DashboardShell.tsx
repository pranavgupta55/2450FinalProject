"use client";

import { useState } from "react";

import { RunInspector } from "@/components/experiments/RunInspector";
import {
  ComparisonTable,
  ConfusionMatrixChart,
  CurveComparisonPanel,
  FeatureImportanceChart,
  MetricsGraph,
  formatMetricValue,
} from "@/components/experiments/Visuals";
import { Badge, CodeLink, DataCard, Eyebrow } from "@/components/ui/TacticalUI";
import type { DashboardData } from "@/lib/experiments";

function formatTimestamp(value: string) {
  return new Intl.DateTimeFormat("en-US", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(value));
}

function TablePanel({
  title,
  count,
  columns,
  rows,
  emptyLabel,
}: {
  title: string;
  count: string;
  columns: string[];
  rows: Array<Record<string, string>>;
  emptyLabel: string;
}) {
  return (
    <div className="rounded-xl border border-border-light bg-bg-panel p-6 shadow-2xl">
      <Eyebrow title={title} count={count} />
      {rows.length === 0 || columns.length === 0 ? (
        <div className="py-8 text-center font-mono text-sm text-text-dim">{emptyLabel}</div>
      ) : (
        <div className="overflow-x-auto">
          <table className="min-w-full border-separate border-spacing-0 text-left">
            <thead>
              <tr>
                {columns.map((column) => (
                  <th
                    key={column}
                    className="border-b border-border-light px-3 py-3 font-sans text-[10px] uppercase tracking-widest text-text-dim"
                  >
                    {column}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, rowIndex) => (
                <tr key={`${title}-${rowIndex}`} className="align-top">
                  {columns.map((column) => (
                    <td
                      key={`${title}-${rowIndex}-${column}`}
                      className="border-b border-border-light/40 px-3 py-3 font-mono text-xs text-text-main"
                    >
                      {row[column] ?? ""}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

export function DashboardShell({ data }: { data: DashboardData }) {
  const [activeRunId, setActiveRunId] = useState<string | null>(data.runs[0]?.id ?? null);
  const [inspectedRunId, setInspectedRunId] = useState<string | null>(null);

  const activeRun =
    data.runs.find((run) => run.id === activeRunId) ?? data.runs[0] ?? null;
  const inspectedRun = data.runs.find((run) => run.id === inspectedRunId) ?? null;

  return (
    <div className="space-y-48 pb-48">
      <section id="overview" className="max-w-3xl scroll-mt-32 space-y-8 pt-16">
        <div className="flex items-center gap-4">
          <span className="font-sans text-sm uppercase tracking-widest text-accent-red">01</span>
          <div className="h-px w-12 bg-border-light" />
        </div>
        <h1 className="text-5xl font-bold tracking-tight md:text-7xl">
          <span className="font-sans">Alpha Signal</span>{" "}
          <span className="font-serif text-accent-red">Dashboard</span>
        </h1>
        <p className="max-w-2xl font-serif text-lg leading-relaxed text-text-dim">
          A tactical artifact browser for the existing Alpha Signal experiment outputs.
          This interface reads directly from <span className="font-mono">artifacts/experiments</span>
          {" "}without changing the ML pipeline.
        </p>
        <div className="flex flex-wrap gap-3 pt-4">
          <Badge variant="dim">{`${data.runs.length} indexed run${data.runs.length === 1 ? "" : "s"}`}</Badge>
          <CodeLink href="#runs">OPEN_RUN_INDEX</CodeLink>
          <CodeLink href="#comparison">OPEN_COMPARISON</CodeLink>
        </div>
      </section>

      <section id="runs" className="scroll-mt-32 space-y-12">
        <Eyebrow title="Experiment Index" count={`${data.artifactRoot} scan`} />

        {data.runs.length === 0 ? (
          <div className="rounded-xl border border-border-light bg-bg-panel p-10 text-center shadow-2xl">
            <div className="font-mono text-sm text-text-dim">[ NO_EXPERIMENT_RUNS_FOUND ]</div>
            <p className="mt-4 font-serif text-sm leading-relaxed text-text-dim">
              Train a model first, then refresh this page to inspect the saved metrics,
              metadata, predictions, and feature importances.
            </p>
          </div>
        ) : (
          <div className="grid grid-cols-1 gap-8 lg:grid-cols-[1.15fr_0.85fr]">
            <div className="space-y-3 rounded-xl border border-border-light bg-bg-panel p-6 shadow-2xl">
              <div className="mb-2 flex items-center justify-between">
                <Eyebrow
                  title="Available Runs"
                  count={`${data.runs.length} run${data.runs.length === 1 ? "" : "s"} available`}
                />
                <Badge variant="green">LIVE</Badge>
              </div>

              {data.runs.map((run) => (
                <DataCard
                  key={run.id}
                  label={run.label}
                  subtitle={`UPDATED ${formatTimestamp(run.updatedAt).toUpperCase()}`}
                  state={run.id === activeRun?.id ? "active" : "default"}
                  onClick={() => {
                    setActiveRunId(run.id);
                    setInspectedRunId(run.id);
                  }}
                  rightNode={
                    <Badge variant={run.id === activeRun?.id ? "gold" : "dim"}>
                      {run.id === activeRun?.id ? "SELECTED" : "OPEN"}
                    </Badge>
                  }
                />
              ))}
            </div>

            <div className="rounded-xl border border-border-light bg-bg-panel p-6 shadow-2xl">
              {activeRun ? (
                <div className="space-y-6">
                  <div className="flex flex-wrap items-center gap-3">
                    <Badge variant="gold">{activeRun.modelName}</Badge>
                    <Badge variant="green">{activeRun.datasetName}</Badge>
                    <Badge variant="dim">{formatTimestamp(activeRun.updatedAt)}</Badge>
                  </div>

                  <div>
                    <Eyebrow title="Selected Run" count="Inspector ready" />
                    <h2
                      className="font-serif text-4xl text-text-main"
                      style={{ fontFamily: "var(--font-glosa)" }}
                    >
                      {activeRun.datasetName}
                    </h2>
                    <p className="mt-3 max-w-xl font-serif leading-relaxed text-text-dim">
                      This run exposes the existing saved experiment bundle for review.
                      Use the sections below to inspect metrics, prediction previews,
                      feature importance previews, and cross-run comparisons.
                    </p>
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div className="rounded-xl border border-border-light/50 bg-bg-base p-4">
                      <p className="font-sans text-[10px] uppercase tracking-widest text-text-dim">
                        Metric Keys
                      </p>
                      <p className="mt-2 font-serif text-3xl text-text-main">
                        {Object.keys(activeRun.metrics).length}
                      </p>
                    </div>
                    <div className="rounded-xl border border-border-light/50 bg-bg-base p-4">
                      <p className="font-sans text-[10px] uppercase tracking-widest text-text-dim">
                        Prediction Rows
                      </p>
                      <p className="mt-2 font-serif text-3xl text-text-main">
                        {activeRun.predictions.totalCount}
                      </p>
                    </div>
                    <div className="rounded-xl border border-border-light/50 bg-bg-base p-4">
                      <p className="font-sans text-[10px] uppercase tracking-widest text-text-dim">
                        Preview Rows
                      </p>
                      <p className="mt-2 font-serif text-3xl text-text-main">
                        {activeRun.predictions.rows.length}
                      </p>
                    </div>
                    <div className="rounded-xl border border-border-light/50 bg-bg-base p-4">
                      <p className="font-sans text-[10px] uppercase tracking-widest text-text-dim">
                        Importance Rows
                      </p>
                      <p className="mt-2 font-serif text-3xl text-text-main">
                        {activeRun.featureImportance.totalCount}
                      </p>
                    </div>
                  </div>

                  <div className="flex flex-wrap gap-3">
                    <CodeLink href="#metrics">OPEN_METRICS</CodeLink>
                    <CodeLink href="#predictions">OPEN_PREDICTIONS</CodeLink>
                    <CodeLink href="#importance">OPEN_IMPORTANCE</CodeLink>
                    <button
                      type="button"
                      onClick={() => setInspectedRunId(activeRun.id)}
                      className="rounded-md border border-accent-red/20 bg-accent-red/10 px-3 py-1 font-mono text-xs uppercase tracking-widest text-accent-red transition-colors hover:bg-accent-red/20"
                    >
                      OPEN_INSPECTOR
                    </button>
                  </div>
                </div>
              ) : null}
            </div>
          </div>
        )}
      </section>

      <section id="metrics" className="scroll-mt-32 space-y-12">
        <Eyebrow title="Run Metrics" count={activeRun ? activeRun.label : "No run selected"} />
        <div className="grid gap-8 xl:grid-cols-[0.95fr_1.05fr]">
          <div className="rounded-xl border border-border-light bg-bg-panel p-6 shadow-2xl">
            {!activeRun ? (
              <div className="py-8 text-center font-mono text-sm text-text-dim">
                [ METRICS_OFFLINE ]
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="min-w-full border-separate border-spacing-0 text-left">
                  <thead>
                    <tr>
                      <th className="border-b border-border-light px-3 py-3 font-sans text-[10px] uppercase tracking-widest text-text-dim">
                        metric
                      </th>
                      <th className="border-b border-border-light px-3 py-3 font-sans text-[10px] uppercase tracking-widest text-text-dim">
                        value
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(activeRun.metrics).map(([metric, value]) => (
                      <tr key={metric}>
                        <td className="border-b border-border-light/40 px-3 py-3 font-mono text-xs text-text-main">
                          {metric}
                        </td>
                        <td className="border-b border-border-light/40 px-3 py-3 font-mono text-xs text-text-main">
                          {formatMetricValue(value)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
          <ConfusionMatrixChart run={activeRun} />
        </div>
        <CurveComparisonPanel runs={data.runs} />
      </section>

      <section id="predictions" className="scroll-mt-32 space-y-12">
        <Eyebrow
          title="Prediction Preview"
          count={
            activeRun
              ? `${activeRun.predictions.rows.length} of ${activeRun.predictions.totalCount} rows`
              : "No run selected"
          }
        />
        <TablePanel
          title="Predictions"
          count={
            activeRun
              ? `${activeRun.predictions.rows.length} preview rows`
              : "Preview unavailable"
          }
          columns={activeRun?.predictions.columns ?? []}
          rows={activeRun?.predictions.rows ?? []}
          emptyLabel="[ PREDICTION_PREVIEW_EMPTY ]"
        />
      </section>

      <section id="importance" className="scroll-mt-32 space-y-12">
        <Eyebrow
          title="Feature Importance"
          count={
            activeRun
              ? `${activeRun.featureImportance.rows.length} of ${activeRun.featureImportance.totalCount} rows`
              : "No run selected"
          }
        />
        <FeatureImportanceChart run={activeRun} />
        <TablePanel
          title="Importance"
          count={
            activeRun
              ? `${activeRun.featureImportance.rows.length} preview rows`
              : "Preview unavailable"
          }
          columns={activeRun?.featureImportance.columns ?? []}
          rows={activeRun?.featureImportance.rows ?? []}
          emptyLabel="[ FEATURE_IMPORTANCE_EMPTY ]"
        />
      </section>

      <section id="comparison" className="scroll-mt-32 space-y-12">
        <Eyebrow title="Cross-Run View" count="Side-by-side metric matrix" />
        <MetricsGraph runs={data.runs} />
        <ComparisonTable runs={data.runs} />
      </section>

      <RunInspector
        run={inspectedRun}
        isOpen={inspectedRun !== null}
        onClose={() => setInspectedRunId(null)}
      />
    </div>
  );
}
