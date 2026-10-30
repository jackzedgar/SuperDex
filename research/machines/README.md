# Machine snapshots

Code synchronization checks five sources every six hours. Only reviewed files
from fixed Git commits are published. These entries describe code availability,
not whether experiments are running or complete.

| Machine | Code location | Captured source commit | Availability |
| --- | --- | --- | --- |
| Lambda | [research root](../README.md) | `69c392c26b697d43a5ace2abb9deb6dd14954786` | Published in this snapshot |
| P920 | [p920/](p920/README.md) | `ff41b78d9a9680c551b8819995a72d956631d5e3` | Published in this snapshot |
| 1202a | Reserved: `machines/1202a/` | None | Deployment copy found; no Git source repository yet |
| 1202b | Reserved: `machines/1202b/` | None | Deployment copy found; no Git source repository yet |
| 1202c | Reserved: `machines/1202c/` | None | Deployment copy found; no Git source repository yet |

Last checked: 2026-10-30 13:43:40 (Asia/Shanghai).
Pending nodes are checked again on subsequent runs. Uncommitted deployment
files and environments are not automatically initialized, committed or uploaded.
Each published machine keeps its own manifest and directory.
