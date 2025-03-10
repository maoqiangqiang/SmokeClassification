import torch.distributed as dist
# import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os 
import torch 


def setup(args_dist_backend, args_init_method, args_world_size):
    ngpus_per_node = torch.cuda.device_count()
    """ This next line is the key to getting DistributedDataParallel working on SLURM:
		SLURM_NODEID is 0 or 1 in this example, SLURM_LOCALID is the id of the 
 		current process inside a node and is also 0 or 1 in this example."""
    
    local_rank = int(os.environ.get("SLURM_LOCALID"))
    print("local_rank: ", local_rank)
    rank = int(os.environ.get("SLURM_NODEID"))*ngpus_per_node + local_rank
    current_device = local_rank
    torch.cuda.set_device(current_device)

    """ this block initializes a process group and initiate communications
		between all processes running on all nodes """

    print('From Rank: {}, ==> Initializing Process Group...'.format(rank))
    #init the process group
    dist.init_process_group(backend=args_dist_backend, init_method=args_init_method, world_size=args_world_size, rank=rank)
    print("process group ready!")
    print('From Rank: {}, ==> Making model..'.format(rank))
    
    return rank, current_device

def cleanup():
    dist.destroy_process_group()



