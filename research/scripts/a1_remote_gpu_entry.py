"""GPU allocation cap applies before loading the registered experiment script."""
import ctypes,os,runpy,signal,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
assert len(sys.argv)>=2 and os.environ.get('A1_REMOTE_GRANT_SHA256')
assert ctypes.CDLL(None).prctl(1,signal.SIGKILL,0,0,0)==0
assert os.getppid()==int(os.environ['A1_REMOTE_PARENT_PID']),'Budget supervisor already exited'
signal.signal(signal.SIGALRM,signal.SIG_DFL)
signal.setitimer(signal.ITIMER_REAL,float(os.environ['A1_REMOTE_CHILD_WALL_SECONDS']))
import torch
import warp as wp
wp.config.kernel_cache_dir=os.environ['WARP_CACHE_PATH']
maximum=int(os.environ['A1_REMOTE_MAX_GPU_MIB']);assert 1024<=maximum<=4096
assert torch.cuda.device_count()==1
total=torch.cuda.get_device_properties(0).total_memory/(1024**2)
torch.cuda.set_per_process_memory_fraction((maximum-512)/total,0)
path=(ROOT/sys.argv[1]).resolve();assert path.is_relative_to(ROOT) and path.is_file()
sys.argv=sys.argv[1:];runpy.run_path(str(path),run_name='__main__')
