export function Background() {
  return (
    <div aria-hidden="true">
      <div className="fixed inset-0 -z-50 bg-bg-base" />

      <div className="fixed inset-0 -z-40 overflow-hidden pointer-events-none">
        <div
          className="absolute inset-[-100%] animate-grid-shift opacity-40"
          style={{
            backgroundImage: `
              linear-gradient(to right, var(--border-light) 1px, transparent 1px),
              linear-gradient(to bottom, var(--border-light) 1px, transparent 1px)
            `,
            backgroundSize: "32px 32px",
          }}
        />
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_80%_60%_at_50%_40%,transparent_0%,var(--bg-base)_100%)]" />
      </div>

      <div
        className="pointer-events-none fixed inset-0 z-50 h-full w-full opacity-[0.06]"
        style={{
          backgroundImage:
            'url("data:image/svg+xml,%3Csvg viewBox=\'0 0 400 400\' xmlns=\'http://www.w3.org/2000/svg\'%3E%3Cfilter id=\'noiseFilter\'%3E%3CfeTurbulence type=\'fractalNoise\' baseFrequency=\'0.85\' numOctaves=\'3\' stitchTiles=\'stitch\'/%3E%3C/filter%3E%3Crect width=\'100%25\' height=\'100%25\' filter=\'url(%23noiseFilter)\'/%3E%3C/svg%3E")',
        }}
      />
    </div>
  );
}
