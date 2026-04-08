"use client";

import { X } from "lucide-react";
import { useEffect, useId, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";

import { cn } from "@/lib/utils";

interface BottomSheetProps {
  isOpen: boolean;
  onClose: () => void;
  title?: string;
  children: ReactNode;
  footer?: ReactNode;
  showCloseButton?: boolean;
  className?: string;
  contentClassName?: string;
}

export function BottomSheet({
  isOpen,
  onClose,
  title,
  children,
  footer,
  showCloseButton = true,
  className,
  contentClassName,
}: BottomSheetProps) {
  const [mounted, setMounted] = useState(false);
  const titleId = useId();

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    if (!mounted || !isOpen) {
      return;
    }

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onClose();
      }
    };

    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", handleKeyDown);

    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [isOpen, mounted, onClose]);

  if (!mounted) {
    return null;
  }

  return createPortal(
    isOpen ? (
      <>
        <button
          type="button"
          tabIndex={-1}
          aria-label="Close panel overlay"
          className="fixed inset-0 z-[70] bg-black/40 backdrop-blur-[2px]"
          onClick={onClose}
        />

        <div
          role="dialog"
          aria-modal="true"
          aria-labelledby={title ? titleId : undefined}
          aria-label={title ? undefined : "Bottom sheet panel"}
          className={cn(
            "fixed inset-x-0 bottom-0 z-[80] mx-auto flex w-full max-w-4xl flex-col rounded-t-xl border border-b-0 border-border-light bg-bg-inspector shadow-[0_-10px_40px_rgba(0,0,0,0.5)]",
            className,
          )}
        >
          {(title || showCloseButton) && (
            <div className="flex items-center justify-between px-6 pb-4 pt-6">
              {title ? (
                <h3 id={titleId} className="font-serif text-2xl font-bold text-text-main">
                  {title}
                </h3>
              ) : (
                <div />
              )}

              {showCloseButton ? (
                <button
                  type="button"
                  onClick={onClose}
                  aria-label="Close panel"
                  className="p-1 text-text-dim transition-colors hover:text-text-main"
                >
                  <X size={20} strokeWidth={1.5} />
                </button>
              ) : null}
            </div>
          )}

          <div
            className={cn(
              "max-h-[70vh] overflow-y-auto px-6 pb-6 font-serif text-text-main",
              contentClassName,
            )}
          >
            {children}
          </div>

          {footer ? (
            <div className="shrink-0 border-t border-border-light/70 px-6 py-4">{footer}</div>
          ) : null}
        </div>
      </>
    ) : null,
    document.body,
  );
}
