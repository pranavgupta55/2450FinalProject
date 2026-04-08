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
  visuals: ExperimentVisuals;
}

export interface DashboardData {
  artifactRoot: string;
  runs: ExperimentRun[];
}

const REPO_ROOT = path.resolve(process.cwd(), "..");
const EXPERIMENT_ROOT = path.join(REPO_ROOT, "artifacts", "experiments");
const PREDICTION_PREVIEW_ROWS = 50;
const FEATURE_PREVIEW_ROWS = 25;
const FEATURE_CHART_ROWS = 12;

type CsvRow = Record<string, string>;

interface PredictionRow {
  actualLabel: number;
  predictedProbability: number;
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

async function loadRun(modelName: string, datasetName: string, runPath: string) {
  const metricsPath = path.join(runPath, "metrics.json");
  if (!(await exists(metricsPath))) {
    return null;
  }

  const [metrics, metadata, predictionRows, featureImportanceRows, stats] = await Promise.all([
    readJsonFile<Record<string, MetricValue>>(metricsPath, {}),
    readJsonFile<Record<string, JsonLike>>(path.join(runPath, "metadata.json"), {}),
    readCsvRows(path.join(runPath, "predictions.csv")),
    readCsvRows(path.join(runPath, "feature_importance.csv")),
    fs.stat(metricsPath),
  ]);

  const labelColumn =
    typeof metadata.label_column === "string"
      ? metadata.label_column
      : "label_abs_alpha_gt_1pct";
  const predictionSeries = buildPredictionRows(predictionRows, labelColumn);
  const { rocCurve, prCurve } = buildCurvePoints(predictionSeries);

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
    visuals: {
      confusionMatrix: buildConfusionMatrix(metrics),
      rocCurve,
      prCurve,
      featureImportance: buildFeatureImportance(featureImportanceRows),
    },
  } satisfies ExperimentRun;
}

export async function loadDashboardData(): Promise<DashboardData> {
  if (!(await exists(EXPERIMENT_ROOT))) {
    return {
      artifactRoot: "artifacts/experiments",
      runs: [],
    };
  }

  const modelEntries = await fs.readdir(EXPERIMENT_ROOT, { withFileTypes: true });
  const runs: ExperimentRun[] = [];

  for (const modelEntry of modelEntries) {
    if (!modelEntry.isDirectory()) {
      continue;
    }

    const modelPath = path.join(EXPERIMENT_ROOT, modelEntry.name);
    const datasetEntries = await fs.readdir(modelPath, { withFileTypes: true });

    for (const datasetEntry of datasetEntries) {
      if (!datasetEntry.isDirectory()) {
        continue;
      }

      const run = await loadRun(
        modelEntry.name,
        datasetEntry.name,
        path.join(modelPath, datasetEntry.name),
      );

      if (run) {
        runs.push(run);
      }
    }
  }

  runs.sort((left, right) => right.updatedAt.localeCompare(left.updatedAt));

  return {
    artifactRoot: "artifacts/experiments",
    runs,
  };
}
