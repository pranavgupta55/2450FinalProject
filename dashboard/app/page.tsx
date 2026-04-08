import { DashboardShell } from "@/components/experiments/DashboardShell";
import { loadDashboardData } from "@/lib/experiments";

export const dynamic = "force-dynamic";

export default async function HomePage() {
  const data = await loadDashboardData();

  return <DashboardShell data={data} />;
}
