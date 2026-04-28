import type { Metadata } from "next";
import { notFound } from "next/navigation";

import { DashboardShell } from "@/components/experiments/DashboardShell";
import { CodeLink, Eyebrow } from "@/components/ui/TacticalUI";
import { loadDashboardData } from "@/lib/experiments";

export const dynamic = "force-dynamic";

function decodeDatasetName(rawValue: string) {
  return decodeURIComponent(rawValue);
}

function formatDatasetLabel(datasetName: string) {
  return datasetName.replace(/[_-]+/g, " ");
}

export async function generateMetadata({
  params,
}: {
  params: Promise<{ datasetName: string }>;
}): Promise<Metadata> {
  const { datasetName: rawDatasetName } = await params;
  const datasetName = decodeDatasetName(rawDatasetName);

  return {
    title: `${formatDatasetLabel(datasetName)} Results`,
  };
}

export default async function DatasetDashboardPage({
  params,
}: {
  params: Promise<{ datasetName: string }>;
}) {
  const { datasetName: rawDatasetName } = await params;
  const datasetName = decodeDatasetName(rawDatasetName);
  const data = await loadDashboardData({ datasetName });

  if (data.runs.length === 0) {
    notFound();
  }

  return (
    <div className="space-y-12">
      <section className="max-w-3xl space-y-5 pt-10">
        <Eyebrow
          title="Dataset Page"
          count={`${datasetName} experiment slice`}
        />
        <h1 className="font-serif text-4xl text-text-main md:text-5xl">
          {formatDatasetLabel(datasetName)}
        </h1>
        <p className="max-w-2xl font-serif text-base leading-relaxed text-text-dim">
          This page isolates every saved run for the selected dataset so we can review
          the longer-horizon training results without mixing them into the sample dashboard.
        </p>
        <div className="flex flex-wrap gap-3">
          <CodeLink href="/">OPEN_ALL_RUNS</CodeLink>
          <CodeLink href="#runs">OPEN_DATASET_RUNS</CodeLink>
        </div>
      </section>

      <DashboardShell data={data} />
    </div>
  );
}
