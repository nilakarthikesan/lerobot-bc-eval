# EDA — `lerobot/pusht`

| stat | value |
|------|-------|
| episodes | 206 |
| frames | 25650 |
| fps | 10 |
| camera keys | observation.image |
| episode length | min 49 / mean 124.5 / max 246 frames |
| proposed split | train episodes 0-184, held-out 185-205 (21 test episodes) |

## Features

| feature | dtype | shape |
|---------|-------|-------|
| `observation.image` | video | (96, 96, 3) |
| `observation.state` | float32 | (2,) |
| `action` | float32 | (2,) |
| `episode_index` | int64 | (1,) |
| `frame_index` | int64 | (1,) |
| `timestamp` | float32 | (1,) |
| `next.reward` | float32 | (1,) |
| `next.done` | bool | (1,) |
| `next.success` | bool | (1,) |
| `index` | int64 | (1,) |
| `task_index` | int64 | (1,) |
