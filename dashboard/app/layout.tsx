import type { Metadata } from "next";
import type { ReactNode } from "react";
import localFont from "next/font/local";

import "./globals.css";
import { Background } from "@/components/layout/Background";
import { SectionNav } from "@/components/layout/SectionNav";

const glosa = localFont({
  src: "../public/fonts/Glosa-W01-Black.ttf",
  variable: "--font-glosa",
  display: "swap",
});

const slippery = localFont({
  src: [
    {
      path: "../public/fonts/SlipperyTrial-Regular.otf",
      weight: "400",
      style: "normal",
    },
    {
      path: "../public/fonts/SlipperyTrial-RegularItalic.otf",
      weight: "400",
      style: "italic",
    },
    {
      path: "../public/fonts/SlipperyTrial-Bold.otf",
      weight: "700",
      style: "normal",
    },
    {
      path: "../public/fonts/SlipperyTrial-BoldItalic.otf",
      weight: "700",
      style: "italic",
    },
  ],
  variable: "--font-slippery",
  display: "swap",
});

export const metadata: Metadata = {
  title: {
    default: "Alpha Signal Dashboard",
    template: "%s | Alpha Signal Dashboard",
  },
  description: "Tactical dashboard for browsing Alpha Signal experiment artifacts.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className={`${glosa.variable} ${slippery.variable} font-sans`}>
      <body className="relative min-h-screen overflow-x-hidden font-sans antialiased selection:bg-accent-gold/20 selection:text-accent-gold">
        <Background />
        <SectionNav />
        <main className="relative z-10 flex min-h-screen flex-col items-center px-6 py-24 xl:ml-32">
          <div className="w-full max-w-5xl">{children}</div>
        </main>
      </body>
    </html>
  );
}
