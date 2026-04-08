import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./lib/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        bg: {
          base: "var(--bg-base)",
          panel: "var(--bg-panel)",
          inspector: "var(--bg-inspector)",
        },
        accent: {
          gold: "var(--accent-gold)",
          red: "var(--accent-red)",
          green: "var(--accent-green)",
        },
        text: {
          main: "var(--text-main)",
          dim: "var(--text-dim)",
        },
        border: {
          light: "var(--border-light)",
          dashed: "var(--border-dashed)",
        },
        code: {
          text: "var(--color-code-text)",
          green: "var(--color-code-green)",
        },
        tree: {
          blue: "var(--tree-blue)",
          green: "var(--tree-green)",
          yellow: "var(--tree-yellow)",
          rust: "var(--tree-rust)",
          purple: "var(--tree-purple)",
          grey: "var(--tree-grey)",
        },
      },
      fontFamily: {
        serif: ["var(--font-glosa)", "serif"],
        sans: ["var(--font-slippery)", "sans-serif"],
        mono: ["var(--font-slippery)", "monospace"],
      },
      animation: {
        "grid-shift": "grid-shift 20s linear infinite",
      },
      keyframes: {
        "grid-shift": {
          "0%": { transform: "translateY(0) translateX(0)" },
          "100%": { transform: "translateY(32px) translateX(32px)" },
        },
      },
    },
  },
  plugins: [],
};

export default config;
