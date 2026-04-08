import type { ReactNode } from "react";
import Link from "next/link";

import { cn } from "@/lib/utils";

export function Eyebrow({ title, count }: { title: string; count?: string }) {
  return (
    <div className="mb-4 flex flex-col gap-1">
      <span className="font-sans text-[10px] font-bold uppercase tracking-[0.15em] text-accent-red">
        {title}
      </span>
      {count && <span className="font-mono text-[10px] uppercase text-text-dim">{count}</span>}
    </div>
  );
}

export function Badge({
  children,
  variant = "red",
}: {
  children: ReactNode;
  variant?: "red" | "gold" | "green" | "dim";
}) {
  const variants = {
    red: "border-accent-red/30 bg-accent-red/20 text-accent-red",
    green: "border-accent-green/20 bg-accent-green/10 text-accent-green",
    gold: "border-accent-gold/20 bg-accent-gold/10 text-accent-gold",
    dim: "border-white/10 bg-white/5 text-text-dim",
  };

  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full border px-2 py-0.5 font-sans text-[10px] uppercase tracking-wider",
        variants[variant],
      )}
    >
      {children}
    </span>
  );
}

interface DataCardProps {
  label: string;
  subtitle?: string;
  state?: "default" | "active";
  onClick?: () => void;
  rightNode?: ReactNode;
  type?: "button" | "submit";
}

export function DataCard({
  label,
  subtitle,
  state = "default",
  onClick,
  rightNode,
  type = "button",
}: DataCardProps) {
  const states = {
    default: "border-border-light bg-bg-panel text-text-main hover:bg-white/[0.04]",
    active: "border-accent-green bg-accent-green/10 text-white",
  };

  return (
    <button
      type={type}
      onClick={onClick}
      className={cn(
        "flex w-full items-center justify-between rounded-md border px-4 py-3 text-left transition-all",
        states[state],
      )}
    >
      <div className="min-w-0 flex flex-col">
        <span className="truncate font-sans text-sm">{label}</span>
        {subtitle && <span className="mt-1 font-mono text-[10px] text-text-dim">{subtitle}</span>}
      </div>
      {rightNode && <div className="ml-4 shrink-0">{rightNode}</div>}
    </button>
  );
}

interface CodeLinkProps {
  href: string;
  children: ReactNode;
  className?: string;
}

export function CodeLink({ href, children, className }: CodeLinkProps) {
  return (
    <Link
      href={href}
      className={cn(
        "inline-flex items-center rounded-md border border-border-light bg-bg-panel px-3 py-1 font-mono text-xs uppercase tracking-widest text-text-main transition-colors hover:bg-white/5 hover:text-accent-gold",
        className,
      )}
    >
      {children}
    </Link>
  );
}
