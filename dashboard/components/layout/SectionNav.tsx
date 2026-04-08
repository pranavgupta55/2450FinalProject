"use client";

import type React from "react";
import { useEffect, useRef, useState } from "react";

import { cn } from "@/lib/utils";

const SECTIONS = [
  { id: "overview", label: "Overview", number: "01" },
  { id: "runs", label: "Run Index", number: "02" },
  { id: "metrics", label: "Metrics", number: "03" },
  { id: "predictions", label: "Predictions", number: "04" },
  { id: "importance", label: "Importance", number: "05" },
  { id: "comparison", label: "Comparison", number: "06" },
] as const;

type SectionId = (typeof SECTIONS)[number]["id"];

export function SectionNav() {
  const [activeTab, setActiveTab] = useState<SectionId>(SECTIONS[0].id);
  const isClickScrolling = useRef(false);
  const targetSectionId = useRef<string | null>(null);
  const releaseTimeout = useRef<ReturnType<typeof setTimeout> | null>(null);
  const scrollEndTimeout = useRef<ReturnType<typeof setTimeout> | null>(null);
  const updateActiveFromScrollRef = useRef<() => void>(() => {});

  useEffect(() => {
    const getSectionElements = () =>
      SECTIONS.map((section) => document.getElementById(section.id)).filter(
        (element): element is HTMLElement => element !== null,
      );

    const clearTimer = (timer: React.MutableRefObject<ReturnType<typeof setTimeout> | null>) => {
      if (timer.current) {
        clearTimeout(timer.current);
        timer.current = null;
      }
    };

    const releaseClickLock = () => {
      isClickScrolling.current = false;
      targetSectionId.current = null;
      clearTimer(releaseTimeout);
      clearTimer(scrollEndTimeout);
    };

    const updateActiveFromScroll = () => {
      const sections = getSectionElements();

      if (sections.length === 0) {
        return;
      }

      const viewportAnchor = window.innerHeight * 0.4;

      if (isClickScrolling.current && targetSectionId.current) {
        const targetSection = document.getElementById(targetSectionId.current);

        if (!targetSection) {
          releaseClickLock();
        } else {
          const rect = targetSection.getBoundingClientRect();
          const targetReached = rect.top <= viewportAnchor + 24 && rect.bottom >= viewportAnchor - 24;
          const lastSectionId = SECTIONS[SECTIONS.length - 1]?.id;
          const scrolledToBottom =
            window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 4;

          if (!targetReached && !(scrolledToBottom && targetSection.id === lastSectionId)) {
            return;
          }

          releaseClickLock();
        }
      }

      let nextActiveSectionId: SectionId = SECTIONS[0].id;
      let closestDistance = Number.POSITIVE_INFINITY;

      for (const section of sections) {
        const rect = section.getBoundingClientRect();
        const distance =
          rect.top <= viewportAnchor && rect.bottom >= viewportAnchor
            ? 0
            : Math.min(Math.abs(rect.top - viewportAnchor), Math.abs(rect.bottom - viewportAnchor));

        if (distance < closestDistance) {
          closestDistance = distance;
          nextActiveSectionId = section.id as SectionId;
        }
      }

      setActiveTab((current) => (current === nextActiveSectionId ? current : nextActiveSectionId));
    };

    updateActiveFromScrollRef.current = updateActiveFromScroll;

    const handleScroll = () => {
      updateActiveFromScroll();

      if (isClickScrolling.current) {
        clearTimer(scrollEndTimeout);
        scrollEndTimeout.current = setTimeout(() => {
          releaseClickLock();
          updateActiveFromScroll();
        }, 120);
      }
    };

    updateActiveFromScroll();

    window.addEventListener("scroll", handleScroll, { passive: true });
    window.addEventListener("resize", updateActiveFromScroll);

    return () => {
      window.removeEventListener("scroll", handleScroll);
      window.removeEventListener("resize", updateActiveFromScroll);
      clearTimer(releaseTimeout);
      clearTimer(scrollEndTimeout);
    };
  }, []);

  const scrollTo = (id: SectionId) => {
    const element = document.getElementById(id);

    if (!element) {
      return;
    }

    if (releaseTimeout.current) {
      clearTimeout(releaseTimeout.current);
    }

    if (scrollEndTimeout.current) {
      clearTimeout(scrollEndTimeout.current);
    }

    isClickScrolling.current = true;
    targetSectionId.current = id;
    setActiveTab(id);
    element.scrollIntoView({ behavior: "smooth", block: "start" });

    releaseTimeout.current = setTimeout(() => {
      isClickScrolling.current = false;
      targetSectionId.current = null;
      updateActiveFromScrollRef.current();
    }, 2500);
  };

  return (
    <nav className="fixed top-1/2 left-8 z-40 hidden -translate-y-1/2 xl:flex xl:flex-col">
      <div className="flex w-64 flex-col gap-1 rounded-xl border border-border-light bg-white/[0.02] p-2 shadow-2xl backdrop-blur-md">
        {SECTIONS.map((section) => {
          const isActive = activeTab === section.id;

          return (
            <button
              key={section.id}
              type="button"
              onClick={() => scrollTo(section.id)}
              aria-current={isActive ? "page" : undefined}
              className="relative flex w-full items-center gap-3 rounded-md px-3 py-2 text-left text-sm font-sans transition-colors"
            >
              {isActive ? (
                <div
                  className="absolute inset-0 -z-10 rounded-md border border-accent-red/20 bg-accent-red/10"
                />
              ) : null}

              <span
                className={cn(
                  "font-mono text-xs",
                  isActive ? "text-accent-red" : "text-text-dim",
                )}
              >
                {section.number}
              </span>

              <span
                className={cn(
                  "truncate",
                  isActive ? "text-text-main" : "text-text-dim hover:text-white/70",
                )}
              >
                {section.label}
              </span>
            </button>
          );
        })}
      </div>
    </nav>
  );
}
