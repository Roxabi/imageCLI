# Retro: Bootstrapping Architecture Docs for a New Project

**Date:** 2026-03-08
**Project:** imageCLI (Python CLI, Typer + Diffusers)
**Goal:** Make the architect agent useful on a non-boilerplate project by filling `{standards.architecture}` with real content.

---

## Starting State

imageCLI had `stack.yml` fully wired:

```yaml
standards:
  architecture: docs/architecture/     # path existed, but thin content
  backend: docs/standards/backend-patterns.md
  testing: docs/standards/testing.md
  code_review: docs/standards/code-review.md
  contributing: docs/contributing.md
  # dev_process: NOT SET → agent warns but continues
```

The paths existed and pointed to real files, so the architect agent wouldn't fail. But the
content was **engine-specific** (lifecycle, quantization, registry) — nothing about
architectural principles, layers, dependency rules, or when to create abstractions.

**Result:** The architect agent technically worked but was flying blind on architecture
decisions. It had no guidance on layer boundaries, SOLID principles, or pattern vocabulary.

### What existed before

| File | Content | Gap |
|---|---|---|
| `docs/architecture/engines.md` | Engine lifecycle, registry, quantization, GPU matrix | Good but narrow — engine internals only |
| `docs/architecture/adr/001-*.mdx` | One ADR (hardware detection boundary) | Fine — ADRs grow organically |
| `docs/standards/backend-patterns.md` | Lazy loading, deferred imports, CPU offload | Engine-focused, no architectural principles |
| `docs/standards/testing.md` | Runner, what to test, GPU mocking | Basic, no mock boundaries or coverage priorities |
| `docs/standards/code-review.md` | VRAM safety, cleanup, imports, quantization | Narrow checklist, no architecture review items |

### What was missing

- No layer diagram or module map
- No documentation of clean/hexagonal patterns the code already followed
- No dependency rule documented
- No ubiquitous language / domain glossary
- No SOLID principles applied to the codebase
- No error handling strategy documented
- No anti-patterns list
- Standards docs too narrow for the architect agent to make informed decisions

---

## Process: How It Was Done

### Step 1: Audit the stack.yml → agent path mechanism

Verified how `{standards.architecture}` works in the dev-core architect agent:

```
stack.yml → {standards.architecture} = docs/architecture/
         → substituted into architect agent prompt
         → agent reads docs at that path before making decisions
```

Key findings:
- If `{standards.architecture}` is **undefined** → agent fails fast with error
- If path **exists but thin** → agent works but has no real guidance
- `{standards.dev_process}` was **not set** → agent warns and continues (no tier classification)
- All 9 dev-core agents share the same standards paths from stack.yml

### Step 2: Analyze the actual codebase architecture

Before writing any docs, ran a thorough analysis of `src/imagecli/` to map what patterns
the code **already uses** (not aspirational targets):

**Discovery method:** Read every source file, mapped imports, identified patterns.

**Findings:**
- Code already follows hexagonal architecture — `ImageEngine` ABC is a port, concrete
  engines are adapters
- Dependencies flow strictly inward (CLI → facade → domain → infrastructure)
- Zero circular dependencies
- Deferred imports create a hard boundary between domain (fast) and infrastructure (GPU)
- Strategy pattern via engine registry
- Facade pattern via `__init__.py:generate()`
- Template Method pattern in `ImageEngine.generate()` → `_load()` → `_build_pipe_kwargs()`

### Step 3: Study the Roxabi boilerplate convention

Read the boilerplate's architecture docs to understand the structure and format:

- `docs/architecture/backend-ddd-hexagonal.mdx` — the key reference
- `docs/architecture/index.mdx` — overview with diagrams
- `docs/architecture/ubiquitous-language.mdx` — domain glossary
- `docs/standards/backend-patterns.mdx` — coding standards with SOLID, anti-patterns
- `AGENTS.md` — architect agent domain boundaries

**Key convention:** Architecture docs describe **what the code already does**, not
aspirational targets. Each doc has a "when to adopt" table with observable signals, not
subjective judgment.

### Step 4: Adapt boilerplate structure to Python/CLI context

Created Python-specific versions of the boilerplate docs. **Did not copy-paste** — adapted
the structure and format but filled with imageCLI-specific content from the actual codebase
analysis.

| Boilerplate doc (TypeScript/NestJS) | imageCLI adaptation (Python/Typer) |
|---|---|
| `backend-ddd-hexagonal.mdx` — DDD entities, ports, NestJS DI | `patterns.md` — ABC as port, engines as adapters, deferred imports as boundary |
| `index.mdx` — monorepo structure, data flow | `index.md` — flat module map, generation data flow |
| `ubiquitous-language.mdx` — NestJS/RBAC terms | `ubiquitous-language.md` — engine/GPU/quantization terms |
| `backend-patterns.mdx` — NestJS DI, filters, guards | `backend-patterns.md` — template method, lazy loading, layer discipline |

### Step 5: Expand existing standards docs

Didn't replace — expanded in place:

- **backend-patterns.md**: Added architecture principles (dependency rule, SRP, DIP,
  open/closed), error handling layers table, layer discipline table, template method hooks,
  anti-patterns table, AI quick reference
- **testing.md**: Added mock boundaries table, AAA structure, test file naming, coverage
  priorities
- **code-review.md**: Added architecture checks, template method compliance, error handling
  section, testing requirements

---

## What Was Created

### New files (3)

| File | Purpose | Key sections |
|---|---|---|
| `docs/architecture/index.md` | System overview for the architect agent | Layer diagram, module table, dependency flow, data flow (single + batch), entry points |
| `docs/architecture/patterns.md` | Clean/hexagonal architecture mapping | When to apply table, layer model, port (ImageEngine), adapters (engines), strategy (registry), facade (__init__), template method, SOLID, deferred import boundary, anti-patterns |
| `docs/architecture/ubiquitous-language.md` | Domain glossary | Core concepts, generation params, capabilities, GPU/memory, quantization, performance, config, lifecycle states, common confusions |

### Expanded files (3)

| File | Before | After |
|---|---|---|
| `docs/standards/backend-patterns.md` | 67 lines, engine-focused | 170 lines, adds SOLID, layers, errors, anti-patterns, AI reference |
| `docs/standards/testing.md` | 45 lines, basic | 117 lines, adds mock boundaries, structure, priorities |
| `docs/standards/code-review.md` | 38 lines, VRAM checklist | 81 lines, adds architecture, template method, error handling, testing |

### Total: 735 lines added across 6 files

---

## What `/init` Should Learn From This

### Problem: `/init` creates paths but not content

The current `/init` sets up `stack.yml` with correct paths and creates stub files. But the
architect agent needs **content** at those paths, not just file existence. A project can pass
`/doctor` (paths exist, no errors) while the architect agent has nothing useful to read.

### Gap: No codebase analysis step

The missing step between `/init` and "architect agent works" is **analyzing what patterns
the code already uses**. This analysis was done manually here (read all source files, map
imports, identify patterns). `/init` could automate this.

### Suggested improvements for `/init`

1. **Codebase analysis pass.** After creating `stack.yml`, scan the source tree:
   - Map import graph (which modules import which)
   - Identify abstract base classes / interfaces (ports)
   - Identify concrete implementations (adapters)
   - Check for circular dependencies
   - Detect deferred import patterns
   - Identify entry points (CLI, library API, etc.)

2. **Generate architecture docs from analysis.** Use the scan results to populate:
   - `docs/architecture/index.md` — layer diagram, module map, dependency flow
   - `docs/architecture/patterns.md` — patterns found with evidence from code
   - `docs/architecture/ubiquitous-language.md` — extract domain terms from class/function names

3. **Expand standards docs beyond stubs.** The boilerplate has rich standards docs; `/init`
   on a new project should generate Python/TypeScript/Go-specific versions based on detected
   stack, not empty files.

4. **"When to adopt" tables per tech stack.** The boilerplate's `backend-ddd-hexagonal.mdx`
   has a table of observable signals → patterns. `/init` could generate a stack-appropriate
   version (Python patterns vs NestJS patterns vs Go patterns).

5. **Detect missing `stack.yml` entries.** `dev_process` was not set in imageCLI. `/init`
   should either set it or explicitly document that it's optional and what the agent does
   without it (warns and continues).

6. **Architecture doc completeness check.** After generation, verify that the architect agent
   would have answers to common questions:
   - "Where does this logic belong?" → layer discipline table
   - "Should I create an abstraction?" → when-to-adopt table
   - "What's the dependency rule?" → dependency flow diagram
   - "What do these domain terms mean?" → ubiquitous language

### What `/init` should NOT do

- Don't generate aspirational architecture docs — only document what exists
- Don't copy boilerplate docs verbatim — adapt to the actual tech stack
- Don't create docs for patterns the code doesn't use (no DDD entities doc for a CLI tool)
- Don't add `dev_process` standards if the project doesn't use the full dev workflow

---

## Verification

After the docs were created, the architect agent's `{standards.architecture}` resolves to
a directory containing:

```
docs/architecture/
├── index.md                 — layer diagram, module map
├── patterns.md              — clean/hexagonal mapping
├── ubiquitous-language.md   — domain glossary
├── engines.md               — engine lifecycle (pre-existing)
└── adr/
    ├── 001-*.mdx            — hardware detection ADR
    └── meta.json            — ADR index
```

The architect agent can now:
- Classify changes by layer (presentation / domain / infrastructure)
- Enforce the dependency rule
- Decide whether a new abstraction is warranted (when-to-adopt table)
- Create ADRs that reference established patterns
- Use correct domain vocabulary
