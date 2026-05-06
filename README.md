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

## AWS Setup

### 1. Install AWS CLI

Follow the [official AWS CLI installation guide](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) to install the AWS CLI.

### 2. Configure AWS IAM User

Run the following command to configure your credentials:

```bash
aws configure
```

When prompted, enter the following:

| Prompt | Value |
|--------|-------|
| **Access Key ID** | Shared with you on Slack |
| **Secret Access Key** | Shared with you on Slack |
| **Default region** | `us-west-1` |
| **Output format** | Press Enter to keep the default |

### 3. Syncing with S3

You have **read and write access** to the `game-arena-data` S3 bucket. Files are synced into the `aws-logs/` directory in this project.

Download from S3:

```bash
aws s3 sync s3://game-arena-data/replays/ ./aws-logs/
```

Upload to S3:

```bash
aws s3 sync ./aws-logs/ s3://game-arena-data/replays/
```

### 4. S3 Bucket Structure

```
replays/
├── four_in_a_row-fen/          # FEN variant
└── four_in_a_row-standard/     # Standard variant
```

Each variant contains **351 matchup folders** for all pairings of 27 models (27 × 26 / 2 = 351). Each matchup has **4 game folders** named by game ID — 2 games where model A is player 0 and 2 games where model B is player 0:

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
| `game_log.json` | Game event stream, metadata, and outcome of the game |
| `game.log` | Debugging log |
| `<player-0-model>(0)-log.jsonl` | Logs for player 0 |
| `<player-1-model>(1)-log.jsonl` | Logs for player 1 |
