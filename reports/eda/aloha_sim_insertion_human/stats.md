# EDA — `lerobot/aloha_sim_insertion_human`

| stat | value |
|------|-------|
| episodes | 50 |
| frames | 25000 |
| fps | 50 |
| camera keys | observation.images.top |
| episode length | min 500 / mean 500.0 / max 500 frames |
| proposed split | train episodes 0-44, held-out 45-49 (5 test episodes) |

## Features

| feature | dtype | shape |
|---------|-------|-------|
| `observation.images.top` | video | (480, 640, 3) |
| `observation.state` | float32 | (14,) |
| `action` | float32 | (14,) |
| `episode_index` | int64 | (1,) |
| `frame_index` | int64 | (1,) |
| `timestamp` | float32 | (1,) |
| `next.done` | bool | (1,) |
| `index` | int64 | (1,) |
| `task_index` | int64 | (1,) |
