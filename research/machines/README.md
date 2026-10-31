# Machine snapshots

Code synchronization checks five sources every six hours. Only reviewed files
from fixed Git commits are published. These entries describe code availability,
not whether experiments are running or complete.

| Machine | Code location | Captured source commit | Availability |
| --- | --- | --- | --- |
| Lambda | [research root](../README.md) | `a55d268db97fac1957bcd309a23218db88dadfcb` | Published in this snapshot |
| P920 | [p920/](p920/README.md) | `ff41b78d9a9680c551b8819995a72d956631d5e3` | Published in this snapshot |
| 1202a | Reserved: `machines/1202a/` | None | No Git source repository found in the configured locations |
| 1202b | Reserved: `machines/1202b/` | None | Deployment copy found; no Git source repository yet |
| 1202c | Reserved: `machines/1202c/` | None | Deployment copy found; no Git source repository yet |

Last checked: 2026-10-31 14:26:51 (Asia/Shanghai).
Pending nodes are checked again on subsequent runs. Uncommitted deployment
files and environments are not automatically initialized, committed or uploaded.
Each published machine keeps its own manifest and directory.
