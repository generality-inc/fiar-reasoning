# Four in a Row

## Python Environment Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you haven't already.

Sync the environment (installs Python 3.11 and all pinned dependencies from `uv.lock`):

```bash
uv sync
```

You also need the Graphviz system binary for tree rendering. On Mac:

```bash
brew install graphviz
```

For other systems, see the [Graphviz download page](https://graphviz.org/download/).

## Data Download

The raw game logs (~1.1 GB compressed, ~3.1 GB uncompressed) are hosted on OSF.

```bash
# Download and extract game logs
wget -O aws-logs.tar.gz "https://osf.io/download/69fba9c72c2505747af2009d/?view_only=e2e531ce7af64cd5a0a9a889bd60bfc0"
tar -xzf aws-logs.tar.gz
rm aws-logs.tar.gz
```

This creates an `aws-logs/` directory with the following structure:

```
aws-logs/
├── four_in_a_row-fen/          # FEN notation variant (used in the paper)
└── four_in_a_row-standard/     # Standard notation variant
```

Each variant contains **351 matchup folders** for all pairings of 27 models (27 × 26 / 2 = 351). Each matchup has **4 game folders** — 2 games where model A moves first and 2 where model B moves first:

```
four_in_a_row-fen/
└── <model-a>_vs_<model-b>/
    ├── batch-1c82113b-...--20251018-160729-000392/
    ├── batch-.../
    ├── batch-.../
    └── batch-.../
```

Each game folder contains **4 files**:

| File | Description |
|------|-------------|
| `game_log.json` | Game event stream, metadata, and outcome |
| `game.log` | Debugging log |
| `<player-0-model>(0)-log.jsonl` | Full logs for player 0 (includes reasoning traces) |
| `<player-1-model>(1)-log.jsonl` | Full logs for player 1 (includes reasoning traces) |
