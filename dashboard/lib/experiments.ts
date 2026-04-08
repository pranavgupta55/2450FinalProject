import { promises as fs } from "fs";
import path from "path";

import { parse } from "csv-parse/sync";

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

export interface ExperimentRun {
  id: string;
  label: string;
  modelName: string;
  datasetName: string;
  updatedAt: string;
  metrics: Record<string, string | number | boolean | null>;
  metadata: Record<string, JsonLike>;
  predictions: TablePreview;
  featureImportance: TablePreview;
}

export interface DashboardData {
  artifactRoot: string;
  runs: ExperimentRun[];
}

const REPO_ROOT = path.resolve(process.cwd(), "..");
const EXPERIMENT_ROOT = path.join(REPO_ROOT, "artifacts", "experiments");
const PREDICTION_PREVIEW_ROWS = 50;
const FEATURE_PREVIEW_ROWS = 25;

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

async function readCsvPreview(filePath: string, limit: number): Promise<TablePreview> {
  if (!(await exists(filePath))) {
    return {
      columns: [],
      rows: [],
      totalCount: 0,
    };
  }

  const raw = await fs.readFile(filePath, "utf8");
  const rows = parse(raw, {
    columns: true,
    skip_empty_lines: true,
    trim: true,
  }) as Array<Record<string, string>>;

  return {
    columns: rows[0] ? Object.keys(rows[0]) : [],
    rows: rows.slice(0, limit),
    totalCount: rows.length,
  };
}

async function loadRun(modelName: string, datasetName: string, runPath: string) {
  const metricsPath = path.join(runPath, "metrics.json");
  if (!(await exists(metricsPath))) {
    return null;
  }

  const [metrics, metadata, predictions, featureImportance, stats] = await Promise.all([
    readJsonFile<Record<string, string | number | boolean | null>>(metricsPath, {}),
    readJsonFile<Record<string, JsonLike>>(path.join(runPath, "metadata.json"), {}),
    readCsvPreview(path.join(runPath, "predictions.csv"), PREDICTION_PREVIEW_ROWS),
    readCsvPreview(
      path.join(runPath, "feature_importance.csv"),
      FEATURE_PREVIEW_ROWS,
    ),
    fs.stat(metricsPath),
  ]);

  return {
    id: `${modelName}/${datasetName}`,
    label: `${modelName} / ${datasetName}`,
    modelName,
    datasetName,
    updatedAt: stats.mtime.toISOString(),
    metrics,
    metadata,
    predictions,
    featureImportance,
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
