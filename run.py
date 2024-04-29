
import torch
import os
import sys
import wandb
import os
import signal
import acdc
from tqdm import tqdm
from typing import Any
acdc.tqdm = tqdm
from transformer_lens.hook_points import HookPoint
from acdc import (
    Edge,
    ACDCConfig,
    LOG_LEVEL_INFO,
    LOG_LEVEL_DEBUG,
    run_acdc,
    ACDCEvalData,
    load_checkpoint,
    get_most_recent_checkpoint
)


if not 'JOB_NAME' in os.environ or os.environ['JOB_NAME'] is None or os.environ['JOB_NAME'] == "":
    raise ValueError(f"Need to set environment variable JOB_NAME")

index = int(os.environ['JOB_COMPLETION_INDEX'])

job_name_text = os.environ['JOB_NAME'] + str(index) + ".txt"

if os.path.exists(job_name_text):
    with open(job_name_text, "r") as f:
        wandb_id = f.read().strip()
    print(f"resuming from {wandb_id}")
    resume = True
else:
    wandb_id = wandb.util.generate_id()
    with open(job_name_text, "w") as f:
        f.write(wandb_id)
    print(f"starting new run {wandb_id}")
    resume = False

# requires
# pip install git+https://github.com/Phylliida/MambaLens.git

from mamba_lens import HookedMamba # this will take a little while to import

model_path = "state-spaces/mamba-370m"
model = HookedMamba.from_pretrained(model_path, device='cuda')


from acdc.data.ioi import ioi_data_generator, ABC_TEMPLATES, get_all_single_name_abc_patching_formats
from acdc.data.utils import generate_dataset

num_patching_pairs = 20
seed = 27
valid_seed = 28
constrain_to_answers = True
has_symmetric_patching = True

templates = ABC_TEMPLATES
patching_formats = list(get_all_single_name_abc_patching_formats())

data = generate_dataset(model=model,
                  data_generator=ioi_data_generator,
                  num_patching_pairs=num_patching_pairs,
                  seed=seed,
                  valid_seed=valid_seed,
                  constrain_to_answers=constrain_to_answers,
                  has_symmetric_patching=has_symmetric_patching, 
                  varying_data_lengths=True,
                  templates=templates,
                  patching_formats=patching_formats)

print(data.data[0])
print(data.valid_data[0])

from acdc import get_pad_token

print("printing example data points:")
for b in range(10):
    pad_token = get_pad_token(model.tokenizer)
    # because there is padding if lengths vary, this only fetches the tokens that are part of the sequence
    toks = data.data[b][:data.last_token_position[b]+1]
    print(model.tokenizer.decode(toks))
    for ind, tok in enumerate(data.correct[b]):
        if tok != pad_token:
            print(f"  correct answer: {repr(model.tokenizer.decode([tok.item()]))}")
    for ind, tok in enumerate(data.incorrect[b]):
        if tok != pad_token:
            print(f"  incorrect answer: {repr(model.tokenizer.decode([tok.item()]))}")

from acdc import accuracy_metric
from acdc import ACDCEvalData
from acdc import get_pad_token
def logging_incorrect_metric(data: ACDCEvalData):
    pad_token = get_pad_token(model.tokenizer)
    for data_subset in [data.patched]:
        batch, _ = data_subset.data.size()
        for b in range(batch):
            if not data_subset.top_is_correct[b].item():
                toks = data_subset.data[b][:data_subset.last_token_position[b]+1]
                print("failed on this data point:")
                print(model.tokenizer.decode(toks))
                print("correct prs:")
                for i, tok in enumerate(data_subset.correct[b]):
                    if tok.item() != pad_token:
                        print(data_subset.correct_prs[b,i].item(), model.tokenizer.decode([tok.item()]))
                print("incorrect prs:")
                for i, tok in enumerate(data_subset.incorrect[b]):
                    if tok.item() != pad_token:
                        print(data_subset.incorrect_prs[b,i].item(), model.tokenizer.decode([tok.item()]))
    return data.patched.top_is_correct

#top_is_correct = data.eval(model=model, batch_size=10, metric=logging_incorrect_metric)
#accuracy = f'{top_is_correct.sum().item()}/{top_is_correct.size()[0]}'
#print(f"accuracy: {accuracy}")

global storage
storage = {}
def storage_hook(
    x,
    hook: HookPoint,
    **kwargs,
):
    global storage
    storage[hook.name] = x
    return x

def resid_patching_hook(
    x,
    hook: HookPoint,
    input_hook_name: str,
    batch_start: int,
    batch_end: int,
):
    global storage
    x_uncorrupted = storage[input_hook_name][batch_start:batch_end:2]
    x_corrupted = storage[input_hook_name][batch_start+1:batch_end:2]
    x[batch_start:batch_end:2] = x[batch_start:batch_end:2] - x_uncorrupted + x_corrupted
    return x

def conv_patching_hook(
    x,
    hook: HookPoint,
    input_hook_name: str,
    batch_start: int,
    batch_end: int
):
    corrupted = x[batch_start+1:batch_end:2]
    x[batch_start:batch_end:2] = corrupted

layers = list(range(model.cfg.n_layers))
#layers = [0,1,11,12,14,15,18,19,20,21,24,25,26,27,29,31,32,33,34,35,36,37,38,39,40,43,47]

## Setup edges for ACDC
edges = []

B,L = data.data.size()

INPUT_HOOK = f'hook_embed'
INPUT_NODE = 'input'

last_layer = max(layers)
OUTPUT_HOOK = f'blocks.{last_layer}.hook_resid_post'
OUTPUT_NODE = 'output'

def layer_hook_name(layer):
    return f'blocks.{layer}.hook_conv'

def layer_node(layer):
    return f'{layer}'

# direct connection from embed to output

edges.append(Edge(
        input_node=INPUT_NODE,
        input_hook=(INPUT_HOOK, storage_hook),
        output_node=OUTPUT_NODE,
        output_hook=(OUTPUT_HOOK, resid_patching_hook),
))

for layer in layers:
    # edge from embed to layer input
    edges.append(Edge(
            input_node=INPUT_NODE,
            input_hook=(INPUT_HOOK, storage_hook),
            output_node=layer_node(layer),
            output_hook=(f'blocks.{layer}.hook_layer_input', resid_patching_hook),
    ))

    # edge from some other earlier layer to this layer
    for other_layer in layers:
        if other_layer < layer:
            edges.append(Edge(
                    input_node=layer_node(other_layer),
                    input_hook=(f'blocks.{other_layer}.hook_out_proj', storage_hook),
                    output_node=layer_node(layer),
                    output_hook=(f'blocks.{layer}.hook_layer_input', resid_patching_hook),
            ))

    # edge from layer output to final layer output
    edges.append(Edge(
            input_node=layer_node(layer),
            input_hook=(f'blocks.{layer}.hook_out_proj', storage_hook),
            output_node=OUTPUT_NODE,
            output_hook=(OUTPUT_HOOK, resid_patching_hook),
    ))

model_kwargs = {
    'fast_ssm': True,
    'fast_conv': True,
}



    

def normalized_logit_diff_metric(data: ACDCEvalData, printing=False):
    B,V = data.patched.logits.size()

    # [batch_size]
    patched_correct = data.patched.correct[:,0]
    #print(data.unpatched.logits.size(), data.patched.logits.size(), data.corrupted.logits.size())
    A_logits_unpatched = data.unpatched.logits[torch.arange(B), patched_correct]
    A_logits_patched = data.patched.logits[torch.arange(B), patched_correct]
    A_logits_corrupted = data.corrupted.logits[torch.arange(B), patched_correct]

    corrupted_correct = data.corrupted.correct[:,0]
    B_logits_unpatched = data.unpatched.logits[torch.arange(B), corrupted_correct]
    B_logits_patched = data.patched.logits[torch.arange(B), corrupted_correct]
    B_logits_corrupted = data.corrupted.logits[torch.arange(B), corrupted_correct]

    # A and B are two potential outputs
    # if A patched > B patched, we are correct
    # else we are incorrect

    # thus we could just return A_logits_patched - B_logits_patched

    # however it is useful to "normalize" these values

    # in the worst case, our patching causes us to act like corrupted, and our diff will be
    # A_logits_corrupted - B_logits_corrupted
    # this will result in a small, negative value
    
    # in the best case, our patching will do nothing (cause us to act like unpatched), and our diff will be
    # A_logits_unpatched - B_logits_unpatched
    # this will result in a large, positive value

    # thus we can treat those as the "min" and "max" and normalize accordingly
    
    min_diff = A_logits_corrupted - B_logits_corrupted
    max_diff = A_logits_unpatched - B_logits_unpatched

    possible_range = (max_diff-min_diff)
    possible_range[possible_range == 0] = 1.0 # prevent divide by zero
    
    diff = A_logits_patched - B_logits_patched
    normalized_diff = (diff-min_diff)/possible_range

    if printing:
        print(f"A corrupted {A_logits_corrupted}")
        print(f"B corrupted {B_logits_corrupted}")
        print(f"A unpatched {A_logits_unpatched}")
        print(f"B unpatched {B_logits_unpatched}")
        print(f"A patched {A_logits_patched}")
        print(f"B patched {B_logits_patched}")
        print(f"min diff {min_diff}")
        print(f"max diff {max_diff}")
        print(f"possible range {possible_range}")
        print(f"diff {diff}")
        print(f"normalized diff {normalized_diff}")
    # as described, 1.0 corresponds to acting like unpatched,
    # and 0.0 corresponds to acting like corrupted

    return torch.mean(normalized_diff)
from functools import partial
#normalized_diff = data.eval(model=model, batch_size=3, metric=partial(normalized_logit_diff_metric, printing=True), store_unpatched_logits=True)
#print(f"normalized diff {normalized_diff}")
cfg = ACDCConfig(
    ckpt_directory = wandb_id,
    thresh = 0.001,
    rollback_thresh = 0.001,
    metric=normalized_logit_diff_metric,
    # extra inference args
    model_kwargs=model_kwargs,
    # these are needed for doing graph pruning
    input_node=INPUT_NODE,
    output_node=OUTPUT_NODE,
    # batch size for evaluating data points
    batch_size=4,
    log_level=LOG_LEVEL_INFO,
    # if False, will be equivalent to batch_size=1
    batched = True,
    # set these two to false to use traditional ACDC
    # recursive will try patching multiple at a time (this is faster sometimes)
    recursive = True,
    # try_patching_multiple_at_same_time will evaluate many different patchings before commiting to any
    # and includes a rollback scheme if after patching one, the others get worse
    try_patching_multiple_at_same_time = True,
    ## if true, you metric will also have the logits from a run with no patching available
    # (useful for normalized logit diff)
    store_unpatched_logits = True,
)

wandb_resume = None
if resume:
    try:
        most_recent_checkpoint_path, is_done = get_most_recent_checkpoint(checkpoint_dir=cfg.ckpt_directory)
        if is_done:
            print(f"run {cfg.ckpt_directory} is completed")
        else:
            print(f"resuming from checkpoint {most_recent_checkpoint_path}")
            old_cfg, edges = load_checkpoint(path=most_recent_checkpoint_path)
            cfg.iter = old_cfg.iter
            wandb_resume = 'must'
    except FileNotFoundError:
        print(f"no checkpoints available at {cfg.ckpt_directory}, starting from scratch")
class InterruptedException(Exception):
    pass

def interrupt_callback(sig_num: Any, stack_frame: Any):
    raise InterruptedException()

try:
    wandb.init(
        project="mamba-acdc",
        config=cfg,
        name=job_name_text.replace(".txt", "") + f' thresh={cfg.thresh} rollback_thresh={cfg.rollback_thresh}',
        resume=wandb_resume,
        id=wandb_id,
    )
    # signal handlers (if preempted)
    signal.signal(signal.SIGINT, interrupt_callback)
    signal.signal(signal.SIGTERM, interrupt_callback)
    result_edges = run_acdc(model=model, data=data, cfg=cfg, edges=edges)
except (KeyboardInterrupt, InterruptedException):
    print("interrupted")