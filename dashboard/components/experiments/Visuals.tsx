import { Eyebrow } from "@/components/ui/TacticalUI";
import type {
  ConfusionMatrixCell,
  CurvePoint,
  ExperimentRun,
  FeatureImportanceDatum,
  MetricValue,
} from "@/lib/experiments";

const PERFORMANCE_METRICS = [
  "accuracy",
  "precision",
  "recall",
  "f1",
  "roc_auc",
  "average_precision",
] as const;

const RUN_COLOR_CLASSES = [
  "bg-tree-blue",
  "bg-tree-green",
  "bg-tree-yellow",
  "bg-tree-rust",
  "bg-tree-purple",
  "bg-tree-grey",
] as const;

const RUN_COLORS = [
  "#7C9FB4",
  "#79A773",
  "#D1A556",
  "#C17C5F",
  "#9C7EB8",
  "#8A857F",
] as const;

const CHART_WIDTH = 420;
const CHART_HEIGHT = 280;
const CHART_PADDING = 28;

export function formatMetricValue(value: MetricValue) {
  if (typeof value === "number") {
    if (Number.isInteger(value)) {
      return value.toLocaleString();
    }

    return value.toFixed(4);
  }

  if (typeof value === "boolean") {
    return value ? "true" : "false";
  }

  if (value === null) {
    return "null";
  }

  return value;
}

function toNumericMetric(value: MetricValue) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

export function formatCurrencyValue(value: MetricValue) {
  const numericValue = toNumericMetric(value);
  if (numericValue === null) {
    return "n/a";
  }

  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(numericValue);
}

export function formatPercentValue(value: MetricValue) {
  const numericValue = toNumericMetric(value);
  if (numericValue === null) {
    return "n/a";
  }

  return `${(numericValue * 100).toFixed(2)}%`;
}

function formatMetricLabel(metric: string) {
  if (metric === "f1") {
    return "F1";
  }

  if (metric === "roc_auc") {
    return "ROC AUC";
  }

  if (metric === "average_precision") {
    return "Average Precision";
  }

  return metric.replace(/_/g, " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

function getPerformanceMetricKeys(runs: ExperimentRun[]) {
  return PERFORMANCE_METRICS.filter((metric) =>
    runs.some((run) => typeof run.metrics[metric] === "number"),
  );
}

function buildLinePath(points: CurvePoint[]) {
  if (points.length === 0) {
    return "";
  }

  const drawableWidth = CHART_WIDTH - CHART_PADDING * 2;
  const drawableHeight = CHART_HEIGHT - CHART_PADDING * 2;

  return points
    .map((point, index) => {
      const x = CHART_PADDING + point.x * drawableWidth;
      const y = CHART_HEIGHT - CHART_PADDING - point.y * drawableHeight;
      return `${index === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(" ");
}

function renderGridLines() {
  const marks = [0, 0.25, 0.5, 0.75, 1];
  const drawableWidth = CHART_WIDTH - CHART_PADDING * 2;
  const drawableHeight = CHART_HEIGHT - CHART_PADDING * 2;

  return marks.map((mark) => {
    const x = CHART_PADDING + mark * drawableWidth;
    const y = CHART_HEIGHT - CHART_PADDING - mark * drawableHeight;

    return (
      <g key={`grid-${mark}`}>
        <line
          x1={CHART_PADDING}
          y1={y}
          x2={CHART_WIDTH - CHART_PADDING}
          y2={y}
          stroke="rgba(255,255,255,0.08)"
          strokeDasharray="3 6"
        />
        <line
          x1={x}
          y1={CHART_PADDING}
          x2={x}
          y2={CHART_HEIGHT - CHART_PADDING}
          stroke="rgba(255,255,255,0.08)"
          strokeDasharray="3 6"
        />
      </g>
    );
  });
}

function buildValueLinePath(
  points: ExperimentRun["visuals"]["portfolioCurve"],
  minValue: number,
  maxValue: number,
) {
  if (points.length === 0) {
    return "";
  }

  const drawableWidth = CHART_WIDTH - CHART_PADDING * 2;
  const drawableHeight = CHART_HEIGHT - CHART_PADDING * 2;
  const range = Math.max(maxValue - minValue, 1);

  return points
    .map((point, index) => {
      const x = CHART_PADDING + point.x * drawableWidth;
      const normalizedY = (point.portfolioValue - minValue) / range;
      const y = CHART_HEIGHT - CHART_PADDING - normalizedY * drawableHeight;
      return `${index === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(" ");
}

function renderValueGridLines(minValue: number, maxValue: number) {
  const marks = [0, 0.25, 0.5, 0.75, 1];
  const drawableWidth = CHART_WIDTH - CHART_PADDING * 2;
  const drawableHeight = CHART_HEIGHT - CHART_PADDING * 2;
  const range = Math.max(maxValue - minValue, 1);

  return marks.map((mark) => {
    const x = CHART_PADDING + mark * drawableWidth;
    const y = CHART_HEIGHT - CHART_PADDING - mark * drawableHeight;
    const labelValue = minValue + mark * range;

    return (
      <g key={`value-grid-${mark}`}>
        <line
          x1={CHART_PADDING}
          y1={y}
          x2={CHART_WIDTH - CHART_PADDING}
          y2={y}
          stroke="rgba(255,255,255,0.08)"
          strokeDasharray="3 6"
        />
        <line
          x1={x}
          y1={CHART_PADDING}
          x2={x}
          y2={CHART_HEIGHT - CHART_PADDING}
          stroke="rgba(255,255,255,0.08)"
          strokeDasharray="3 6"
        />
        <text
          x={CHART_PADDING - 6}
          y={y + 4}
          textAnchor="end"
          fill="#7A7873"
          fontSize="9"
        >
          {labelValue.toFixed(0)}
        </text>
      </g>
    );
  });
}

function CurveChart({
  title,
  xLabel,
  yLabel,
  runs,
  type,
  showDiagonal = false,
}: {
  title: string;
  xLabel: string;
  yLabel: string;
  runs: ExperimentRun[];
  type: "rocCurve" | "prCurve";
  showDiagonal?: boolean;
}) {
  return (
    <div className="rounded-xl border border-border-light bg-bg-panel p-6 shadow-2xl">
      <Eyebrow title={title} count={`${runs.length} run overlay`} />

      <svg viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`} className="h-auto w-full">
        <rect
          x={CHART_PADDING}
          y={CHART_PADDING}
          width={CHART_WIDTH - CHART_PADDING * 2}
          height={CHART_HEIGHT - CHART_PADDING * 2}
          fill="#151412"
          stroke="rgba(255,255,255,0.08)"
        />

        {renderGridLines()}

        {showDiagonal ? (
          <line
            x1={CHART_PADDING}
            y1={CHART_HEIGHT - CHART_PADDING}
            x2={CHART_WIDTH - CHART_PADDING}
            y2={CHART_PADDING}
            stroke="rgba(122,120,115,0.6)"
            strokeDasharray="5 6"
          />
        ) : null}

        {runs.map((run, index) => {
          const points = run.visuals[type];
          const path = buildLinePath(points);

          if (!path) {
            return null;
          }

          return (
            <path
              key={`${type}-${run.id}`}
              d={path}
              fill="none"
              stroke={RUN_COLORS[index % RUN_COLORS.length]}
              strokeWidth="3"
              strokeLinejoin="round"
              strokeLinecap="round"
            />
          );
        })}

        <text
          x={CHART_WIDTH / 2}
          y={CHART_HEIGHT - 4}
          textAnchor="middle"
          fill="#7A7873"
          fontSize="10"
          style={{ letterSpacing: "0.18em", textTransform: "uppercase" }}
        >
          {xLabel}
        </text>
        <text
          x={14}
          y={CHART_HEIGHT / 2}
          textAnchor="middle"
          fill="#7A7873"
          fontSize="10"
          transform={`rotate(-90 14 ${CHART_HEIGHT / 2})`}
          style={{ letterSpacing: "0.18em", textTransform: "uppercase" }}
        >
          {yLabel}
        </text>
      </svg>
    </div>
  );
}

export function MetricsGraph({ runs }: { runs: ExperimentRun[] }) {
  const metricKeys = getPerformanceMetricKeys(runs);

  if (runs.length === 0 || metricKeys.length === 0) {
    return (
      <div className="rounded-xl border border-border-light bg-bg-panel p-6 shadow-2xl">
        <Eyebrow title="Metrics Graph" count="Performance metrics unavailable" />
        <div className="py-8 text-center font-mono text-sm text-text-dim">
          [ METRIC_GRAPH_OFFLINE ]
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-border-light bg-bg-panel p-6 shadow-2xl">
      <Eyebrow
        title="Metrics Graph"
        count={`${metricKeys.length} performance metrics across ${runs.length} runs`}
      />

      <div className="mb-8 flex flex-wrap gap-3">
        {runs.map((run, index) => (
          <div
            key={run.id}
            className="inline-flex items-center gap-2 rounded-full border border-border-light bg-bg-base px-3 py-1"
          >
            <span
              className={`h-2.5 w-2.5 rounded-full ${RUN_COLOR_CLASSES[index % RUN_COLOR_CLASSES.length]}`}
            />
            <span className="font-mono text-[10px] uppercase tracking-widest text-text-main">
              {run.label}
            </span>
          </div>
        ))}
      </div>

      <div className="space-y-8">
        {metricKeys.map((metric) => (
          <div key={metric} className="space-y-3">
            <div className="flex items-center justify-between gap-4">
              <span className="font-sans text-xs uppercase tracking-widest text-text-dim">
                {formatMetricLabel(metric)}
              </span>
              <span className="font-mono text-[10px] uppercase tracking-widest text-text-dim">
                scale 0.00 to 1.00
              </span>
            </div>

            <div className="rounded-xl border border-border-light/50 bg-bg-base p-4">
              <div className="space-y-3">
                {runs.map((run, index) => {
                  const rawValue = run.metrics[metric];
                  const numericValue =
                    typeof rawValue === "number"
                      ? Math.max(0, Math.min(rawValue, 1))
                      : 0;

                  return (
                    <div
                      key={`${metric}-${run.id}`}
                      className="grid gap-3 md:grid-cols-[190px_1fr_64px] md:items-center"
                    >
                      <span className="font-mono text-[10px] uppercase tracking-widest text-text-dim">
                        {run.modelName} / {run.datasetName}
                      </span>

                      <div className="relative h-3 overflow-hidden rounded-full border border-border-light bg-bg-panel">
                        <div className="absolute inset-0 grid grid-cols-4">
                          <div className="border-r border-border-light/40" />
                          <div className="border-r border-border-light/40" />
                          <div className="border-r border-border-light/40" />
                          <div />
                        </div>
                        <div
                          className={`absolute inset-y-0 left-0 rounded-full ${RUN_COLOR_CLASSES[index % RUN_COLOR_CLASSES.length]}`}
                          style={{ width: `${numericValue * 100}%` }}
                        />
                      </div>

                      <span className="text-right font-mono text-xs text-text-main">
                        {formatMetricValue(rawValue ?? null)}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export function ConfusionMatrixChart({ run }: { run: ExperimentRun | null }) {
  const cells = run?.visuals.confusionMatrix ?? [];

  if (!run || cells.length === 0) {
    return (
      <div className="rounded-xl border border-border-light bg-bg-panel p-6 shadow-2xl">
        <Eyebrow title="Confusion Matrix" count="No run selected" />
        <div className="py-8 text-center font-mono text-sm text-text-dim">
          [ CONFUSION_MATRIX_OFFLINE ]
        </div>
      </div>
    );
  }

  const maxValue = Math.max(...cells.map((cell) => cell.value), 1);
  const negatives = cells.slice(0, 2);
  const positives = cells.slice(2, 4);
  const total = cells.reduce((sum, cell) => sum + cell.value, 0);

  const renderCell = (cell: ConfusionMatrixCell) => {
    const alpha = 0.18 + (cell.value / maxValue) * 0.52;
    const isCorrect =
      cell.id === "true_negatives" || cell.id === "true_positives";
    const rgb = isCorrect ? "121,167,115" : "173,84,75";

    return (
      <div
        key={cell.id}
        className="rounded-xl border p-4"
        style={{
          borderColor: `rgba(${rgb}, 0.35)`,
          backgroundColor: `rgba(${rgb}, ${alpha})`,
        }}
      >
        <p className="font-sans text-[10px] uppercase tracking-widest text-text-main">
          {cell.label}
        </p>
        <p className="mt-3 font-serif text-4xl text-text-main">{cell.value}</p>
        <p className="mt-2 font-mono text-[10px] uppercase tracking-widest text-text-dim">
          {total > 0 ? `${((cell.value / total) * 100).toFixed(1)}% of cases` : "0.0% of cases"}
        </p>
      </div>
    );
  };

  return (
    <div className="rounded-xl border border-border-light bg-bg-panel p-6 shadow-2xl">
      <Eyebrow title="Confusion Matrix" count={run.label} />
      <div className="mb-5 flex flex-wrap items-center gap-3">
        <div className="rounded-full border border-border-light bg-bg-base px-3 py-1 font-mono text-[10px] uppercase tracking-widest text-text-main">
          total {total}
        </div>
        <div className="rounded-full border border-border-light bg-bg-base px-3 py-1 font-mono text-[10px] uppercase tracking-widest text-text-main">
          accuracy {formatMetricValue(run.metrics.accuracy ?? null)}
        </div>
      </div>

      <div className="grid gap-3 md:grid-cols-[130px_repeat(2,minmax(0,1fr))]">
        <div />
        <div className="px-2 text-center font-sans text-[10px] uppercase tracking-widest text-text-dim">
          Predicted Negative
        </div>
        <div className="px-2 text-center font-sans text-[10px] uppercase tracking-widest text-text-dim">
          Predicted Positive
        </div>

        <div className="self-center pr-2 font-sans text-[10px] uppercase tracking-widest text-text-dim">
          Actual Negative
        </div>
        {negatives.map(renderCell)}

        <div className="self-center pr-2 font-sans text-[10px] uppercase tracking-widest text-text-dim">
          Actual Positive
        </div>
        {positives.map(renderCell)}
      </div>
    </div>
  );
}

export function CurveComparisonPanel({ runs }: { runs: ExperimentRun[] }) {
  if (runs.length === 0) {
    return (
      <div className="rounded-xl border border-border-light bg-bg-panel p-6 shadow-2xl">
        <Eyebrow title="Probability Curves" count="No runs available" />
        <div className="py-8 text-center font-mono text-sm text-text-dim">
          [ CURVE_PANEL_OFFLINE ]
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <div className="flex flex-wrap gap-3">
        {runs.map((run, index) => (
          <div
            key={run.id}
            className="inline-flex items-center gap-2 rounded-full border border-border-light bg-bg-base px-3 py-1"
          >
            <span
              className={`h-2.5 w-2.5 rounded-full ${RUN_COLOR_CLASSES[index % RUN_COLOR_CLASSES.length]}`}
            />
            <span className="font-mono text-[10px] uppercase tracking-widest text-text-main">
              {run.label}
            </span>
          </div>
        ))}
      </div>

      <div className="grid gap-8 xl:grid-cols-2">
        <CurveChart
          title="ROC Curve"
          xLabel="False Positive Rate"
          yLabel="True Positive Rate"
          runs={runs}
          type="rocCurve"
          showDiagonal
        />
        <CurveChart
          title="Precision Recall Curve"
          xLabel="Recall"
          yLabel="Precision"
          runs={runs}
          type="prCurve"
        />
      </div>
    </div>
  );
}

export function PortfolioWorthChart({ runs }: { runs: ExperimentRun[] }) {
  const portfolioRuns = runs.filter(
    (run) => run.visuals.portfolioCurve.length > 1,
  );

  if (portfolioRuns.length === 0) {
    return (
      <div className="rounded-xl border border-border-light bg-bg-panel p-6 shadow-2xl">
        <Eyebrow title="Portfolio Worth" count="Model trade curves unavailable" />
        <div className="py-8 text-center font-mono text-sm text-text-dim">
          [ PORTFOLIO_CURVE_OFFLINE ]
        </div>
      </div>
    );
  }

  const allValues = portfolioRuns.flatMap((run) =>
    run.visuals.portfolioCurve.map((point) => point.portfolioValue),
  );
  const minValue = Math.min(...allValues);
  const maxValue = Math.max(...allValues);
  const padding = Math.max((maxValue - minValue) * 0.08, 4);
  const chartMin = Math.max(0, minValue - padding);
  const chartMax = maxValue + padding;

  const dateValues = portfolioRuns.flatMap((run) => {
    const firstPoint = run.visuals.portfolioCurve[0];
    const lastPoint = run.visuals.portfolioCurve[run.visuals.portfolioCurve.length - 1];
    return [firstPoint?.date, lastPoint?.date].filter(
      (value): value is string => typeof value === "string" && value.length > 0,
    );
  });
  const sortedDates = [...dateValues].sort();
  const startDate = sortedDates[0] ?? "";
  const endDate = sortedDates[sortedDates.length - 1] ?? "";

  return (
    <div className="rounded-xl border border-border-light bg-bg-panel p-6 shadow-2xl">
        <Eyebrow
          title="Portfolio Worth"
          count={`${portfolioRuns.length} strategy curves from $100 starting capital`}
        />

      <div className="mb-8 flex flex-wrap gap-3">
        {portfolioRuns.map((run, index) => (
          <div
            key={run.id}
            className="inline-flex items-center gap-2 rounded-full border border-border-light bg-bg-base px-3 py-1"
          >
            <span
              className={`h-2.5 w-2.5 rounded-full ${RUN_COLOR_CLASSES[index % RUN_COLOR_CLASSES.length]}`}
            />
            <span className="font-mono text-[10px] uppercase tracking-widest text-text-main">
              {run.label}
            </span>
          </div>
        ))}
      </div>

      <svg viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`} className="h-auto w-full">
        <rect
          x={CHART_PADDING}
          y={CHART_PADDING}
          width={CHART_WIDTH - CHART_PADDING * 2}
          height={CHART_HEIGHT - CHART_PADDING * 2}
          fill="#151412"
          stroke="rgba(255,255,255,0.08)"
        />

        {renderValueGridLines(chartMin, chartMax)}

        {portfolioRuns.map((run, index) => {
          const points = run.visuals.portfolioCurve;
          const path = buildValueLinePath(points, chartMin, chartMax);
          const lastPoint = points[points.length - 1];
          const drawableWidth = CHART_WIDTH - CHART_PADDING * 2;
          const drawableHeight = CHART_HEIGHT - CHART_PADDING * 2;
          const normalizedY =
            (lastPoint.portfolioValue - chartMin) / Math.max(chartMax - chartMin, 1);
          const circleX = CHART_PADDING + lastPoint.x * drawableWidth;
          const circleY = CHART_HEIGHT - CHART_PADDING - normalizedY * drawableHeight;

          return (
            <g key={`portfolio-${run.id}`}>
              <path
                d={path}
                fill="none"
                stroke={RUN_COLORS[index % RUN_COLORS.length]}
                strokeWidth="3"
                strokeLinejoin="round"
                strokeLinecap="round"
              />
              <circle
                cx={circleX}
                cy={circleY}
                r="4"
                fill={RUN_COLORS[index % RUN_COLORS.length]}
                stroke="#151412"
                strokeWidth="2"
              />
            </g>
          );
        })}

        <text
          x={CHART_WIDTH / 2}
          y={CHART_HEIGHT - 4}
          textAnchor="middle"
          fill="#7A7873"
          fontSize="10"
          style={{ letterSpacing: "0.18em", textTransform: "uppercase" }}
        >
          {startDate && endDate ? `${startDate} to ${endDate}` : "Backtest Window"}
        </text>
        <text
          x={14}
          y={CHART_HEIGHT / 2}
          textAnchor="middle"
          fill="#7A7873"
          fontSize="10"
          transform={`rotate(-90 14 ${CHART_HEIGHT / 2})`}
          style={{ letterSpacing: "0.18em", textTransform: "uppercase" }}
        >
          Portfolio Value
        </text>
      </svg>

      <div className="mt-8 grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        {portfolioRuns.map((run, index) => {
          const endValue = run.visuals.portfolioCurve[run.visuals.portfolioCurve.length - 1]?.portfolioValue ?? 100;
          const totalReturn = endValue / 100 - 1;

          return (
            <div
              key={`${run.id}-summary`}
              className="rounded-xl border border-border-light/50 bg-bg-base p-4"
            >
              <div className="flex items-center gap-2">
                <span
                  className={`h-2.5 w-2.5 rounded-full ${RUN_COLOR_CLASSES[index % RUN_COLOR_CLASSES.length]}`}
                />
                <p className="font-mono text-[10px] uppercase tracking-widest text-text-main">
                  {run.modelName}
                </p>
              </div>
              <p className="mt-3 font-serif text-3xl text-text-main">
                {formatCurrencyValue(endValue)}
              </p>
              <p className="mt-2 font-mono text-[10px] uppercase tracking-widest text-text-dim">
                {formatPercentValue(totalReturn)} total return
              </p>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export function FeatureImportanceChart({ run }: { run: ExperimentRun | null }) {
  const rows = run?.visuals.featureImportance ?? [];

  if (!run || rows.length === 0) {
    return (
      <div className="rounded-xl border border-border-light bg-bg-panel p-6 shadow-2xl">
        <Eyebrow title="Feature Importance Graph" count="No run selected" />
        <div className="py-8 text-center font-mono text-sm text-text-dim">
          [ FEATURE_CHART_OFFLINE ]
        </div>
      </div>
    );
  }

  const maxImportance = Math.max(...rows.map((row) => row.importance), 1);

  return (
    <div className="rounded-xl border border-border-light bg-bg-panel p-6 shadow-2xl">
      <Eyebrow
        title="Feature Importance Graph"
        count={`Top ${rows.length} ranked features for ${run.label}`}
      />

      <div className="space-y-3">
        {rows.map((row, index) => {
          const width = (row.importance / maxImportance) * 100;
          const colorClass = RUN_COLOR_CLASSES[index % RUN_COLOR_CLASSES.length];

          return (
            <div
              key={row.feature}
              className="grid gap-3 md:grid-cols-[180px_1fr_64px] md:items-center"
            >
              <span className="font-mono text-[10px] uppercase tracking-widest text-text-dim">
                {row.feature}
              </span>
              <div className="relative h-3 overflow-hidden rounded-full border border-border-light bg-bg-base">
                <div
                  className={`absolute inset-y-0 left-0 rounded-full ${colorClass}`}
                  style={{ width: `${width}%` }}
                />
              </div>
              <span className="text-right font-mono text-xs text-text-main">
                {row.importance.toFixed(4)}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export function ComparisonTable({ runs }: { runs: ExperimentRun[] }) {
  const metricKeys = Array.from(
    new Set(runs.flatMap((run) => Object.keys(run.metrics))),
  ).sort();

  if (runs.length === 0 || metricKeys.length === 0) {
    return (
      <div className="rounded-xl border border-border-light bg-bg-panel p-6 shadow-2xl">
        <Eyebrow title="Run Comparison" count="Metrics unavailable" />
        <div className="py-8 text-center font-mono text-sm text-text-dim">
          [ NO_COMPARISON_DATA ]
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-border-light bg-bg-panel p-6 shadow-2xl">
      <Eyebrow title="Run Comparison" count={`${runs.length} runs indexed`} />
      <div className="overflow-x-auto">
        <table className="min-w-full border-separate border-spacing-0 text-left">
          <thead>
            <tr>
              <th className="border-b border-border-light px-3 py-3 font-sans text-[10px] uppercase tracking-widest text-text-dim">
                metric
              </th>
              {runs.map((run) => (
                <th
                  key={run.id}
                  className="border-b border-border-light px-3 py-3 font-sans text-[10px] uppercase tracking-widest text-text-dim"
                >
                  {run.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {metricKeys.map((metric) => (
              <tr key={metric}>
                <td className="border-b border-border-light/40 px-3 py-3 font-mono text-xs text-text-main">
                  {metric}
                </td>
                {runs.map((run) => (
                  <td
                    key={`${metric}-${run.id}`}
                    className="border-b border-border-light/40 px-3 py-3 font-mono text-xs text-text-main"
                  >
                    {formatMetricValue(run.metrics[metric] ?? null)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export function TradingComparisonChart({ runs }: { runs: ExperimentRun[] }) {
  const tradingRuns = runs
    .map((run) => ({
      run,
      netPnl: toNumericMetric(run.tradingSummary.net_pnl_dollars ?? null),
      totalReturn: toNumericMetric(run.tradingSummary.return_on_traded_capital ?? null),
      trades: toNumericMetric(run.tradingSummary.trades_executed ?? null),
    }))
    .filter((item) => item.netPnl !== null);

  if (tradingRuns.length === 0) {
    return (
      <div className="rounded-xl border border-border-light bg-bg-panel p-6 shadow-2xl">
        <Eyebrow title="Trading Comparison" count="Trading artifacts unavailable" />
        <div className="py-8 text-center font-mono text-sm text-text-dim">
          [ TRADING_COMPARISON_OFFLINE ]
        </div>
      </div>
    );
  }

  const maxAbsPnl = Math.max(
    ...tradingRuns.map((item) => Math.abs(item.netPnl ?? 0)),
    1,
  );

  return (
    <div className="rounded-xl border border-border-light bg-bg-panel p-6 shadow-2xl">
      <Eyebrow title="Trading Comparison" count={`${tradingRuns.length} run PnL scan`} />
      <div className="space-y-4">
        {tradingRuns.map((item, index) => {
          const width = `${(Math.abs(item.netPnl ?? 0) / maxAbsPnl) * 100}%`;
          const isPositive = (item.netPnl ?? 0) >= 0;
          const colorClass = isPositive
            ? RUN_COLOR_CLASSES[index % RUN_COLOR_CLASSES.length]
            : "bg-accent-red";

          return (
            <div
              key={item.run.id}
              className="grid gap-3 md:grid-cols-[220px_1fr_160px] md:items-center"
            >
              <div className="min-w-0">
                <p className="truncate font-mono text-[10px] uppercase tracking-widest text-text-main">
                  {item.run.label}
                </p>
                <p className="mt-1 font-mono text-[10px] uppercase tracking-widest text-text-dim">
                  {item.trades ?? 0} trades
                </p>
              </div>

              <div className="relative h-4 overflow-hidden rounded-full border border-border-light bg-bg-base">
                <div
                  className={`absolute inset-y-0 left-0 rounded-full ${colorClass}`}
                  style={{ width }}
                />
              </div>

              <div className="text-right">
                <p
                  className={`font-mono text-xs ${isPositive ? "text-accent-green" : "text-accent-red"}`}
                >
                  {formatCurrencyValue(item.netPnl)}
                </p>
                <p className="mt-1 font-mono text-[10px] uppercase tracking-widest text-text-dim">
                  {formatPercentValue(item.totalReturn)}
                </p>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
