# Bottensor Design Tokens

All tokens are defined in `ui/src/styles/tokens.css` and available as both
CSS custom properties (`var(--token)`) and Tailwind utility classes
(`bg-bg-base`, `text-accent-teal`, etc.).

---

## Color palette

| Token | Hex | Tailwind class | Usage |
|---|---|---|---|
| `--bg-base` | `#0b0d12` | `bg-bg-base` | App background, canvas |
| `--bg-elevated` | `#161b26` | `bg-bg-elevated` | Cards, panels, header |
| `--accent-teal` | `#00dbb8` | `text-accent-teal` / `border-accent-teal` | Active state, CTAs, brand |
| `--accent-gold` | `#f5c842` | `text-accent-gold` | Tool-call / waiting state |
| `--text-primary` | `#e8ecf1` | `text-text-primary` | Body copy, code output |
| `--text-muted` | `#8a94a6` | `text-text-muted` | Labels, timestamps, hints |
| `--border` | `#232936` | `border-border` | Dividers, card borders |

### Semantic meanings

- **Teal** — activity, progress, brand identity. Used for active node glow,
  primary buttons, and the wordmark.
- **Gold** — waiting / blocked. Applied to nodes that have issued a tool call
  and are awaiting the result.
- **Muted** — idle or informational. Default node state; secondary text.
- **Red (`#f87171`)** — error state. Not a formal token; used inline for
  error nodes and kill buttons.

---

## Typography

| Token | Value | Tailwind class | Usage |
|---|---|---|---|
| `--font-display` | `"Outfit", sans-serif` | `font-display` | Headings, UI labels, buttons |
| `--font-serif` | `"Instrument Serif", serif` | `font-serif` | Wordmark only |
| `--font-mono` | `"DM Mono", monospace` | `font-mono` | Code, status badges, log entries |

All three families are loaded from Google Fonts in `tokens.css`.

### Weight scale (Outfit)

| Weight | Use |
|---|---|
| 300 | Secondary labels |
| 400 | Body / default |
| 500 | Emphasis |
| 600 | Section headers |
| 700 | Reserved / large display |

---

## Spacing and radius

| Token | Value | Usage |
|---|---|---|
| `--radius-sm` | `6px` | Inputs, badges, small buttons |
| `--radius-md` | `10px` | Cards, agent nodes |
| `--radius-lg` | `16px` | Modals, large panels |

---

## How the token system works

### Runtime (CSS custom properties)

`tokens.css` sets `:root { --token: value }` variables.  Any component can
reference them with `var(--token)`.  This is the source of truth used in
inline styles and global CSS.

### Tailwind v4 (`@theme {}`)

The `@theme {}` block in `tokens.css` exposes the same values as Tailwind
utility classes (e.g. `bg-accent-teal`, `border-border`).  Tailwind v4 reads
`@theme` at build time — no `tailwind.config.js` entries required for
runtime.

### `tailwind.config.js`

This file exists solely for IDE IntelliSense (VS Code Tailwind CSS
extension).  It mirrors the `@theme {}` values so autocomplete works.
It is **not** used at build time.

---

## Adding a new token

1. Add the CSS custom property in `tokens.css` under `:root`:
   ```css
   --color-danger: #f87171;
   ```
2. Expose it in the `@theme {}` block:
   ```css
   @theme {
     --color-danger: #f87171;
   }
   ```
3. Mirror it in `tailwind.config.js` for IDE support:
   ```js
   colors: { danger: 'var(--color-danger)' }
   ```

---

## Node status → visual mapping

| Status | Border color | Glow | Badge color |
|---|---|---|---|
| `idle` | `--border` | none | `--text-muted` |
| `running` | `--accent-teal` | teal 25 % opacity | `--accent-teal` |
| `done` | `--border` | none | `#3ecf8e` |
| `error` | `#f87171` | none | `#f87171` |
| `waiting` | `--accent-gold` | gold 20 % opacity | `--accent-gold` |
