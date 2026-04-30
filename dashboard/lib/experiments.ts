import { promises as fs } from "fs";
import path from "path";

import { parse } from "csv-parse/sync";

export type MetricValue = string | number | boolean | null;

export type JsonLike =
  | string
  | number
  | boolean
  | null
  | JsonLike[]
  | { [key: string]: JsonLike };

export interface TablePreview {
  columns: string[];
  rows: Array<Record<string, string>>;
  totalCount: number;
}

export interface CurvePoint {
  x: number;
  y: number;
  threshold: number;
}

export interface PortfolioPoint {
  x: number;
  date: string;
  label: string;
  portfolioValue: number;
  periodReturn: number;
}

export interface FeatureImportanceDatum {
  feature: string;
  importance: number;
}

export interface ConfusionMatrixCell {
  id: "true_negatives" | "false_positives" | "false_negatives" | "true_positives";
  label: string;
  rowLabel: string;
  columnLabel: string;
  value: number;
}

export interface ExperimentVisuals {
  confusionMatrix: ConfusionMatrixCell[];
  rocCurve: CurvePoint[];
  prCurve: CurvePoint[];
  featureImportance: FeatureImportanceDatum[];
  portfolioCurve: PortfolioPoint[];
}

export interface ExperimentRun {
  id: string;
  label: string;
  modelName: string;
  datasetName: string;
  updatedAt: string;
  metrics: Record<string, MetricValue>;
  metadata: Record<string, JsonLike>;
  predictions: TablePreview;
  featureImportance: TablePreview;
  tradeLog: TablePreview;
  tradingSummary: Record<string, MetricValue>;
  visuals: ExperimentVisuals;
}

export interface StrategyRun {
  id: string;
  label: string;
  strategyName: string;
  datasetName: string;
  updatedAt: string;
  metrics: Record<string, MetricValue>;
  metadata: Record<string, JsonLike>;
  predictions: TablePreview;
  tradeLog: TablePreview;
  tradingSummary: Record<string, MetricValue>;
  visuals: {
    portfolioCurve: PortfolioPoint[];
  };
}

export interface DashboardData {
  artifactRoot: string;
  runs: ExperimentRun[];
  strategyArtifactRoot: string;
  strategyRuns: StrategyRun[];
}

interface LoadDashboardDataOptions {
  datasetName?: string;
}

const REPO_ROOT = path.resolve(process.cwd(), "..");
const EXPERIMENT_ROOT = path.join(REPO_ROOT, "artifacts", "experiments");
const STRATEGY_ANALYSIS_ROOT = path.join(REPO_ROOT, "artifacts", "strategy_analysis");
const PREDICTION_PREVIEW_ROWS = 50;
const FEATURE_PREVIEW_ROWS = 25;
const TRADE_PREVIEW_ROWS = 50;
const FEATURE_CHART_ROWS = 12;
const PORTFOLIO_STARTING_CAPITAL = 100;

type CsvRow = Record<string, string>;

interface PredictionRow {
  actualLabel: number;
  predictedProbability: number;
}

interface TradeObservation {
  period: string;
  sortValue: number;
  returnContribution: number;
  active: boolean;
  aggregationMode: "average" | "sum";
}

async function exists(targetPath: string) {
  try {
    await fs.access(targetPath);
    return true;
  } catch {
    return false;
  }
}

async function readJsonFile<T>(filePath: string, fallback: T): Promise<T> {
  if (!(await exists(filePath))) {
    return fallback;
  }

  const raw = await fs.readFile(filePath, "utf8");
  return JSON.parse(raw) as T;
}

async function readCsvRows(filePath: string): Promise<CsvRow[]> {
  if (!(await exists(filePath))) {
    return [];
  }

  const raw = await fs.readFile(filePath, "utf8");
  return parse(raw, {
    columns: true,
    skip_empty_lines: true,
    trim: true,
  }) as CsvRow[];
}

function buildTablePreview(rows: CsvRow[], limit: number): TablePreview {
  return {
    columns: rows[0] ? Object.keys(rows[0]) : [],
    rows: rows.slice(0, limit),
    totalCount: rows.length,
  };
}

function toNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }

  if (typeof value === "string") {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }

  return null;
}

function buildConfusionMatrix(metrics: Record<string, MetricValue>): ConfusionMatrixCell[] {
  const confusionKeys = [
    "true_negatives",
    "false_positives",
    "false_negatives",
    "true_positives",
  ] as const;
  const hasConfusionData = confusionKeys.some((key) => toNumber(metrics[key]) !== null);
  if (!hasConfusionData) {
    return [];
  }

  return [
    {
      id: "true_negatives",
      label: "True Negatives",
      rowLabel: "Actual Negative",
      columnLabel: "Predicted Negative",
      value: toNumber(metrics.true_negatives) ?? 0,
    },
    {
      id: "false_positives",
      label: "False Positives",
      rowLabel: "Actual Negative",
      columnLabel: "Predicted Positive",
      value: toNumber(metrics.false_positives) ?? 0,
    },
    {
      id: "false_negatives",
      label: "False Negatives",
      rowLabel: "Actual Positive",
      columnLabel: "Predicted Negative",
      value: toNumber(metrics.false_negatives) ?? 0,
    },
    {
      id: "true_positives",
      label: "True Positives",
      rowLabel: "Actual Positive",
      columnLabel: "Predicted Positive",
      value: toNumber(metrics.true_positives) ?? 0,
    },
  ];
}

function buildFeatureImportance(rows: CsvRow[]): FeatureImportanceDatum[] {
  return rows
    .map((row) => ({
      feature: row.feature ?? "",
      importance: toNumber(row.importance) ?? 0,
    }))
    .filter((row) => row.feature.length > 0)
    .sort((left, right) => right.importance - left.importance)
    .slice(0, FEATURE_CHART_ROWS);
}

function buildPredictionRows(rows: CsvRow[], labelColumn: string): PredictionRow[] {
  return rows
    .map((row) => {
      const actualLabel = toNumber(row[labelColumn]);
      const predictedProbability = toNumber(row.predicted_probability);

      if (actualLabel === null || predictedProbability === null) {
        return null;
      }

      return {
        actualLabel,
        predictedProbability,
      } satisfies PredictionRow;
    })
    .filter((row): row is PredictionRow => row !== null);
}

function buildCurvePoints(rows: PredictionRow[]) {
  if (rows.length === 0) {
    return {
      rocCurve: [] as CurvePoint[],
      prCurve: [] as CurvePoint[],
    };
  }

  const sortedRows = [...rows].sort(
    (left, right) => right.predictedProbability - left.predictedProbability,
  );
  const positives = sortedRows.filter((row) => row.actualLabel === 1).length;
  const negatives = sortedRows.length - positives;

  if (positives === 0 || negatives === 0) {
    return {
      rocCurve: [],
      prCurve: [],
    };
  }

  const rocCurve: CurvePoint[] = [{ x: 0, y: 0, threshold: 1 }];
  const prCurve: CurvePoint[] = [{ x: 0, y: 1, threshold: 1 }];

  let truePositives = 0;
  let falsePositives = 0;

  for (let index = 0; index < sortedRows.length; index += 1) {
    const row = sortedRows[index];
    if (row.actualLabel === 1) {
      truePositives += 1;
    } else {
      falsePositives += 1;
    }

    const nextRow = sortedRows[index + 1];
    const thresholdChanged =
      !nextRow || nextRow.predictedProbability !== row.predictedProbability;

    if (!thresholdChanged) {
      continue;
    }

    const recall = truePositives / positives;
    const falsePositiveRate = falsePositives / negatives;
    const precision = truePositives / (truePositives + falsePositives);

    rocCurve.push({
      x: falsePositiveRate,
      y: recall,
      threshold: row.predictedProbability,
    });
    prCurve.push({
      x: recall,
      y: Number.isFinite(precision) ? precision : 1,
      threshold: row.predictedProbability,
    });
  }

  rocCurve.push({ x: 1, y: 1, threshold: 0 });
  prCurve.push({
    x: 1,
    y: positives / sortedRows.length,
    threshold: 0,
  });

  return { rocCurve, prCurve };
}

function parseTradeObservation(row: CsvRow): TradeObservation | null {
  const weekStart = row.week_start;
  if (weekStart) {
    const realizedAlpha = toNumber(row.realized_alpha);
    const position = toNumber(row.trade_position) ?? 0;
    const tradeTaken = (toNumber(row.trade_taken) ?? (position !== 0 ? 1 : 0)) !== 0;
    const sortValue = Date.parse(weekStart);

    if (realizedAlpha === null || Number.isNaN(sortValue)) {
      return null;
    }

    return {
      period: weekStart,
      sortValue,
      returnContribution:
        toNumber(row.period_return_contribution) ??
        (tradeTaken ? position * realizedAlpha : 0),
      active: tradeTaken,
      aggregationMode: toNumber(row.period_return_contribution) !== null ? "sum" : "average",
    };
  }

  const benchmarkDate = row.Date ?? row.date;
  const benchmarkReturn = toNumber(row.sp_return_1d ?? row.daily_return ?? row.market_return);
  const position = toNumber(row.position ?? row.trade_position) ?? 1;
  const tradeTaken = (toNumber(row.trade_taken) ?? 1) !== 0;
  const sortValue = benchmarkDate ? Date.parse(benchmarkDate) : Number.NaN;

  if (!benchmarkDate || benchmarkReturn === null || Number.isNaN(sortValue)) {
    return null;
  }

  return {
    period: benchmarkDate,
    sortValue,
    returnContribution:
      toNumber(row.period_return_contribution) ??
      (tradeTaken ? position * benchmarkReturn : 0),
    active: tradeTaken,
    aggregationMode: toNumber(row.period_return_contribution) !== null ? "sum" : "average",
  };
}

function groupTradeObservations(rows: CsvRow[]) {
  const grouped = new Map<
    string,
    {
      period: string;
      sortValue: number;
      activeCount: number;
      returnSum: number;
      aggregationMode: "average" | "sum";
    }
  >();

  rows
    .map(parseTradeObservation)
    .filter((observation): observation is TradeObservation => observation !== null)
    .forEach((observation) => {
      const current = grouped.get(observation.period) ?? {
        period: observation.period,
        sortValue: observation.sortValue,
        activeCount: 0,
        returnSum: 0,
        aggregationMode: observation.aggregationMode,
      };
      current.activeCount += observation.active ? 1 : 0;
      current.returnSum += observation.returnContribution;
      current.aggregationMode =
        current.aggregationMode === "sum" || observation.aggregationMode === "sum"
          ? "sum"
          : "average";
      grouped.set(observation.period, current);
    });

  return Array.from(grouped.values()).sort((left, right) => left.sortValue - right.sortValue);
}

function buildPortfolioCurve(
  rows: CsvRow[],
  startingCapital: number = PORTFOLIO_STARTING_CAPITAL,
): PortfolioPoint[] {
  const orderedPeriods = groupTradeObservations(rows);
  if (orderedPeriods.length === 0) {
    return [];
  }

  let portfolioValue = startingCapital;
  const points: PortfolioPoint[] = [
    {
      x: 0,
      date: orderedPeriods[0].period,
      label: orderedPeriods[0].period,
      portfolioValue: startingCapital,
      periodReturn: 0,
    },
  ];

  orderedPeriods.forEach((period, index) => {
    const periodReturn =
      period.aggregationMode === "sum"
        ? period.returnSum
        : period.activeCount > 0
          ? period.returnSum / period.activeCount
          : 0;
    portfolioValue *= 1 + periodReturn;
    points.push({
      x: (index + 1) / orderedPeriods.length,
      date: period.period,
      label: period.period,
      portfolioValue: Number(portfolioValue.toFixed(4)),
      periodReturn,
    });
  });

  return points;
}

async function listRunDirectories(modelPath: string, datasetNameFilter: string | null) {
  const datasetEntries = await fs.readdir(modelPath, { withFileTypes: true });
  return datasetEntries
    .filter((entry) => entry.isDirectory())
    .filter((entry) => !datasetNameFilter || entry.name === datasetNameFilter)
    .map((entry) => ({
      datasetName: entry.name,
      runPath: path.join(modelPath, entry.name),
    }));
}

async function loadRun(modelName: string, datasetName: string, runPath: string) {
  const metricsPath = path.join(runPath, "metrics.json");
  if (!(await exists(metricsPath))) {
    return null;
  }

  const [
    metrics,
    metadata,
    predictionRows,
    featureImportanceRows,
    tradingSummary,
    tradeLogRows,
    stats,
  ] = await Promise.all([
    readJsonFile<Record<string, MetricValue>>(metricsPath, {}),
    readJsonFile<Record<string, JsonLike>>(path.join(runPath, "metadata.json"), {}),
    readCsvRows(path.join(runPath, "predictions.csv")),
    readCsvRows(path.join(runPath, "feature_importance.csv")),
    readJsonFile<Record<string, MetricValue>>(path.join(runPath, "trading_summary.json"), {}),
    readCsvRows(path.join(runPath, "trade_log.csv")),
    fs.stat(metricsPath),
  ]);

  const labelColumn =
    typeof metadata.label_column === "string"
      ? metadata.label_column
      : "label_abs_alpha_gt_1pct";
  const predictionSeries = buildPredictionRows(predictionRows, labelColumn);
  const { rocCurve, prCurve } = buildCurvePoints(predictionSeries);
  const portfolioCurve = buildPortfolioCurve(tradeLogRows);

  return {
    id: `${modelName}/${datasetName}`,
    label: `${modelName} / ${datasetName}`,
    modelName,
    datasetName,
    updatedAt: stats.mtime.toISOString(),
    metrics,
    metadata,
    predictions: buildTablePreview(predictionRows, PREDICTION_PREVIEW_ROWS),
    featureImportance: buildTablePreview(featureImportanceRows, FEATURE_PREVIEW_ROWS),
    tradeLog: buildTablePreview(tradeLogRows, TRADE_PREVIEW_ROWS),
    tradingSummary,
    visuals: {
      confusionMatrix: buildConfusionMatrix(metrics),
      rocCurve,
      prCurve,
      featureImportance: buildFeatureImportance(featureImportanceRows),
      portfolioCurve,
    },
  } satisfies ExperimentRun;
}

async function loadStrategyRun(strategyName: string, datasetName: string, runPath: string) {
  const metricsPath = path.join(runPath, "metrics.json");
  if (!(await exists(metricsPath))) {
    return null;
  }

  const [metrics, metadata, predictionRows, tradingSummary, tradeLogRows, stats] =
    await Promise.all([
      readJsonFile<Record<string, MetricValue>>(metricsPath, {}),
      readJsonFile<Record<string, JsonLike>>(path.join(runPath, "metadata.json"), {}),
      readCsvRows(path.join(runPath, "predictions.csv")),
      readJsonFile<Record<string, MetricValue>>(path.join(runPath, "trading_summary.json"), {}),
      readCsvRows(path.join(runPath, "trade_log.csv")),
      fs.stat(metricsPath),
    ]);

  return {
    id: `${strategyName}/${datasetName}`,
    label: `${strategyName} / ${datasetName}`,
    strategyName,
    datasetName,
    updatedAt: stats.mtime.toISOString(),
    metrics,
    metadata,
    predictions: buildTablePreview(predictionRows, PREDICTION_PREVIEW_ROWS),
    tradeLog: buildTablePreview(tradeLogRows, TRADE_PREVIEW_ROWS),
    tradingSummary,
    visuals: {
      portfolioCurve: buildPortfolioCurve(tradeLogRows),
    },
  } satisfies StrategyRun;
}

export async function loadDashboardData(
  options: LoadDashboardDataOptions = {},
): Promise<DashboardData> {
  const datasetNameFilter = options.datasetName?.trim() || null;
  const experimentRootExists = await exists(EXPERIMENT_ROOT);
  const strategyRootExists = await exists(STRATEGY_ANALYSIS_ROOT);

  if (!experimentRootExists && !strategyRootExists) {
    return {
      artifactRoot: datasetNameFilter
        ? `artifacts/experiments/*/${datasetNameFilter}`
        : "artifacts/experiments",
      runs: [],
      strategyArtifactRoot: datasetNameFilter
        ? `artifacts/strategy_analysis/*/${datasetNameFilter}`
        : "artifacts/strategy_analysis",
      strategyRuns: [],
    };
  }

  const runsByModel = experimentRootExists
    ? await Promise.all(
        (await fs.readdir(EXPERIMENT_ROOT, { withFileTypes: true }))
          .filter((entry) => entry.isDirectory())
          .map(async (modelEntry) => {
            const modelPath = path.join(EXPERIMENT_ROOT, modelEntry.name);
            const runDirectories = await listRunDirectories(modelPath, datasetNameFilter);
            const loadedRuns = await Promise.all(
              runDirectories.map(({ datasetName, runPath }) =>
                loadRun(modelEntry.name, datasetName, runPath),
              ),
            );
            return loadedRuns.filter((run): run is ExperimentRun => run !== null);
          }),
      )
    : [];

  const runs = runsByModel.flat();
  const strategyRuns = strategyRootExists
    ? (
        await Promise.all(
          (
            await fs.readdir(STRATEGY_ANALYSIS_ROOT, { withFileTypes: true })
          )
            .filter((entry) => entry.isDirectory())
            .map(async (strategyEntry) => {
              const strategyPath = path.join(STRATEGY_ANALYSIS_ROOT, strategyEntry.name);
              const runDirectories = await listRunDirectories(strategyPath, datasetNameFilter);
              const loadedRuns = await Promise.all(
                runDirectories.map(({ datasetName, runPath }) =>
                  loadStrategyRun(strategyEntry.name, datasetName, runPath),
                ),
              );
              return loadedRuns.filter((run): run is StrategyRun => run !== null);
            }),
        )
      ).flat()
    : [];

  runs.sort((left, right) => right.updatedAt.localeCompare(left.updatedAt));
  strategyRuns.sort((left, right) => right.updatedAt.localeCompare(left.updatedAt));

  return {
    artifactRoot: datasetNameFilter
      ? `artifacts/experiments/*/${datasetNameFilter}`
      : "artifacts/experiments",
    runs,
    strategyArtifactRoot: datasetNameFilter
      ? `artifacts/strategy_analysis/*/${datasetNameFilter}`
      : "artifacts/strategy_analysis",
    strategyRuns,
  };
}
