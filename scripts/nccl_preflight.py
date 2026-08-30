"""One all_reduce, to find out whether this node's NCCL works at all.

Jobs 316457, 316497 and 316571 each spent three minutes loading 220 GiB of
weights onto 25a-hgpn143 and then hung forever on the first collective --
231 SU per run, three runs, and because each carried a different prefill
configuration each failure got blamed on the configuration. `init_process_group`
succeeds on that node; `all_reduce` never returns and NCCL's own watchdog does
not fire. Twenty seconds here turns that into a cheap abort.
"""

import datetime
import os

import torch
import torch.distributed as dist

rank = int(os.environ["RANK"])
world = int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=60))

buf = torch.ones(1 << 20, device="cuda") * (rank + 1)
dist.all_reduce(buf)
torch.cuda.synchronize()

expected = world * (world + 1) // 2
if rank == 0:
    got = buf[0].item()
    if abs(got - expected) > 1e-3:
        raise SystemExit(f"NCCL PREFLIGHT MISMATCH: got {got}, expected {expected}")
    print(f"    NCCL ok on {os.uname().nodename} ({world} ranks)", flush=True)
dist.barrier()
dist.destroy_process_group()
