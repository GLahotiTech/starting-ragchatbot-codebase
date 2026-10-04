# Frontend changes: code quality tooling

No files under `frontend/` (HTML/CSS/JS) were modified. Black is a Python
formatter, so this feature only touches the backend and dev workflow.

## What changed
- Added `black` as a dev dependency (`uv add --dev black`) and a `[tool.black]`
  section in `pyproject.toml` (line length 88, py313, excludes `chroma_db`).
- Added `scripts/format.sh` (auto-format `backend/` and `main.py`) and
  `scripts/check.sh` (`black --check` plus the pytest suite).
- Reformatted all Python files with black (16 files; formatting only).
- All 61 tests pass after reformatting.

## Usage
- `./scripts/format.sh`: format the code
- `./scripts/check.sh`: verify formatting and run the tests

Frontend formatting (e.g. Prettier for JS/CSS) was not added, since it wasn't requested.

# Frontend changes: dark/light theme toggle

## `frontend/index.html`
- Added a `#themeToggle` button (fixed, top-right) with inline sun and moon SVG icons. It has `aria-label` and `aria-pressed`, and is a native `<button>`, so Tab/Enter/Space work.
- Added a small inline script in `<head>` that applies the saved theme (`localStorage`) to `<html data-theme>` before first paint, which avoids a flash of the wrong theme.
- Bumped the CSS and JS cache-busting versions.

## `frontend/style.css`
- Added a `:root[data-theme="light"]` block that overrides the CSS variables: light backgrounds and surfaces, dark text, a darker teal primary (white text on it passes WCAG AA), and adjusted borders and focus ring.
- Replaced hardcoded colors with new variables so both themes work: `--on-primary`, `--on-user-message`, `--link-hover`, `--link-underline`, `--user-link-underline`, `--code-bg`, `--welcome-shadow`, `--error-*`, `--success-*`, `--button-glow`. Dark values equal the previous hardcoded ones, so the dark theme looks unchanged.
- Fixed the blockquote border, which referenced the undefined `--primary`.
- Added `.theme-toggle` styles: a circular button with hover and `:focus-visible` ring. The sun and moon icons cross-fade and rotate.
- Added an `html.theme-transition` rule that animates background, text and border colors for 0.35s.

## `frontend/script.js`
- Added `setupThemeToggle()`, called on load. A click flips `data-theme` on `<html>` between `light` and `dark`, saves it to `localStorage`, updates the button's aria attributes, and applies the temporary `theme-transition` class so the change is smooth. The class is removed afterwards so normal interactions aren't slowed.

## Verification (Playwright)
- Toggled by click and by keyboard (Enter), checked persistence across reloads, and checked desktop and 390px mobile layouts in both themes.
- Measured WCAG contrast for 20+ text/background pairs in both themes (messages, links, code, blockquote, sources, error/success, sidebar, buttons, input). All are at least 4.5:1 after these fixes:
  - `body` background had been accidentally set to `--on-primary` and is now `--background`.
  - Links in user messages now use `--on-user-message` (was 3.18:1 in light).
  - Light `--success-text` darkened to `#166534` (was 4.2:1).
