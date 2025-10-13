# Research Project Documentation

This repository contains three related LaTeX projects that support my master's research: the full dissertation, the original dissertation proposal, and a systematic review template. Each folder is self‑contained so they can be compiled independently.

## Folder Overview

### 1. `dissertation/`
My main thesis/dissertation manuscript.

**Highlights**
- Intended for the final submission document.
- Typically organized into chapters (e.g., `chapters/01-introduction.tex`, etc.). (Folder listing not shown here, but follow the same style as the proposal.)
- Central entry point usually named something like `main.tex` (create if not present yet) which includes the chapter files and sets global packages, formatting, and bibliography style.

### 2. `dissertation-proposal/`
The original proposal document (already compiled auxiliary files are present).

### 3. `systematic-review/`
A systematic literature review before full LaTeX integration.

## Bibliography Management
One `references.bib` per project.

## Troubleshooting Quick Notes
- Missing references: run the build tool enough times or ensure bibliography backend matches preamble.
- Encoding errors: confirm source files are UTF‑8 and set `\usepackage[utf8]{inputenc}` (if using pdfLaTeX) or switch to XeLaTeX/LuaLaTeX which natively handle UTF‑8.
- Figure not found: check relative path from the including `.tex` file; prefer `\graphicspath{{figures/}}`.
- Stale `.aux` causing weird cross‑ref issues: clean auxiliary files and rebuild.

