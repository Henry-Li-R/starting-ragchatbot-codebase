# Frontend Changes

## Code Quality Tooling

### What was added

**Prettier** was set up as the automatic code formatter for all frontend files (HTML, CSS, JS).

### New files

| File | Purpose |
|------|---------|
| `frontend/package.json` | npm manifest; declares Prettier as a dev dependency and exposes `format` / `format:check` scripts |
| `frontend/.prettierrc` | Prettier config: 100-char print width, 2-space indent, LF line endings, ES5 trailing commas |
| `frontend/.prettierignore` | Tells Prettier to skip `node_modules/` |
| `check-frontend.sh` | Top-level dev script — runs `prettier --check` on all frontend files and exits non-zero on any formatting violation |

### Files reformatted

`frontend/index.html`, `frontend/script.js`, and `frontend/style.css` were reformatted by Prettier to establish a consistent baseline.

### `.gitignore` additions

`frontend/node_modules/` and `frontend/package-lock.json` were added to `.gitignore`.

### Usage

```bash
# Check formatting (CI / pre-commit)
./check-frontend.sh

# Auto-fix formatting
cd frontend && npm run format

# Check only (no writes)
cd frontend && npm run format:check
```
