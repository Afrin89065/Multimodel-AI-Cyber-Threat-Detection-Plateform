/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      colors: {
        soc: {
          bg:      "#0a0e1a",
          panel:   "#0f1629",
          border:  "#1e2d4a",
          accent:  "#1d4ed8",
          critical:"#ef4444",
          high:    "#f97316",
          medium:  "#eab308",
          low:     "#22c55e",
          text:    "#e2e8f0",
          muted:   "#64748b",
        }
      },
      fontFamily: {
        mono: ["JetBrains Mono", "Fira Code", "monospace"]
      }
    }
  },
  plugins: []
};