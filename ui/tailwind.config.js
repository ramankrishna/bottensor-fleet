/** @type {import('tailwindcss').Config} */
// Used by tooling / IDE integrations. Runtime config lives in src/styles/tokens.css @theme.
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        'bg-base':     'var(--bg-base)',
        'bg-elevated': 'var(--bg-elevated)',
        'accent-teal': 'var(--accent-teal)',
        'accent-gold': 'var(--accent-gold)',
        'text-primary':'var(--text-primary)',
        'text-muted':  'var(--text-muted)',
        border:        'var(--border)',
      },
      fontFamily: {
        display: ['var(--font-display)', 'sans-serif'],
        serif:   ['var(--font-serif)',   'serif'],
        mono:    ['var(--font-mono)',    'monospace'],
      },
    },
  },
  plugins: [],
}
