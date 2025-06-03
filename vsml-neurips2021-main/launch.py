import wandb
import mpi4py.MPI as MPI
import yaml
import logging
import os
import config
import numpy as np
import tensorflow as tf

from experiment import Experiment
from config import expand_dot_items, WandbMockConfig, WandbMockSummary, flatten_dot_items, DotDict, GLOBAL_CONFIG

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _warn_new_keys(config, existing_config):
    for k in config.keys() - existing_config.keys():
        if '.mpi_' not in k:
            logging.warning(f'Specified config key {k} does not exist.')


def _merge_config(config, config_files, task_config=None):
    derived_config = {}
    # Ensure default.yaml is loaded relative to script dir
    default_cfg_path = os.path.join(SCRIPT_DIR, 'configs/default.yaml')
    
    # Process other config files, making them relative to SCRIPT_DIR if they are not absolute
    processed_config_files = []
    for cfg_file in config_files:
        if not os.path.isabs(cfg_file):
            # Assuming config files passed as args are relative to project root (containing SCRIPT_DIR)
            # or directly under SCRIPT_DIR/configs if just a name like 'cifar10.yaml'
            # For simplicity, let's assume they are like 'configs/cifar10.yaml' or an absolute path
            # A more robust solution might check if cfg_file starts with 'configs/'
            if os.path.exists(os.path.join(SCRIPT_DIR, cfg_file)):
                 processed_config_files.append(os.path.join(SCRIPT_DIR, cfg_file))
            elif os.path.exists(cfg_file): # If it's a relative path from CWD that exists
                 processed_config_files.append(cfg_file)
            else: # Fallback to assuming it's meant to be in SCRIPT_DIR/configs
                 processed_config_files.append(os.path.join(SCRIPT_DIR, 'configs', os.path.basename(cfg_file)))
        else:
            processed_config_files.append(cfg_file)

    all_files_to_load = [default_cfg_path] + processed_config_files
    
    for cfg_path in all_files_to_load:
        try:
            with open(cfg_path, mode='r') as f:
                new_config = flatten_dot_items(yaml.safe_load(f))
                if len(derived_config) > 0:
                    _warn_new_keys(new_config, derived_config)
                derived_config.update(new_config)
        except FileNotFoundError:
            logging.warning(f"Configuration file {cfg_path} not found. Skipping.")
            # If default.yaml is missing, it's a critical error
            if cfg_path == default_cfg_path:
                logging.error(f"Critical error: Default configuration file {default_cfg_path} is missing.")
                raise
    if task_config is not None:
        _warn_new_keys(task_config, derived_config)
        derived_config.update(task_config)
    if config is not None:
        _warn_new_keys(config, derived_config)
        derived_config.update(config)
    derived_config = flatten_dot_items(derived_config)
    return derived_config


def _sync_config(comm, mpi_rank):
    config = wandb.config._items if mpi_rank == 0 else None
    config = comm.bcast(config, root=0)
    if mpi_rank > 0:
        # Create mock wandb objects because we didnt initialize wandb in this process
        wandb.config = WandbMockConfig(config)
        wandb.summary = WandbMockSummary()
        wandb.log = lambda *args, **kwargs: None
    return config


def _update_ranked_config(config: dict, mpi_rank: int):
    # TODO potentially add support for dictionaries within the mpi_split
    for k, v in filter(lambda it: 'mpi_split' in it[0], list(config.items())):
        base_key = k.replace('.mpi_split', '')
        repeat_key = f'{base_key}.mpi_repeat'
        repeat = config.pop(repeat_key, 1)
        idx = mpi_rank // repeat
        selected_option = v[idx % len(v)]
        config[base_key] = selected_option
        del config[k]
    return config


def _save_log_to_wandb():
    if 'SLURM_JOB_ID' in os.environ:
        if 'SLURM_ARRAY_JOB_ID' in os.environ:
            job_id = os.environ['SLURM_ARRAY_JOB_ID'] + '_' + os.environ['SLURM_ARRAY_TASK_ID']
            wandb.summary.slurm_array_jobid = os.environ['SLURM_ARRAY_JOB_ID']
        else:
            job_id = os.environ['SLURM_JOB_ID']
        wandb.summary.slurm_jobid = job_id
        log_src = f'slurm-{job_id}.out'
        log_dst = os.path.join(wandb.run.dir, f'slurm-{job_id}.txt')
        os.link('./' + log_src, log_dst)
        wandb.save(log_dst)
        with open(f'wandb-run-{job_id}', 'a') as f:
            f.write(f'{wandb.run.id}\n')


def _create_array_task(spec, mpi_rank, array_subset):
    task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    tasks = spec['array']
    if array_subset is not None and len(array_subset) > 0:
        task = tasks[array_subset[task_id - 1]]
    else:
        task = tasks[task_id - 1]
    if mpi_rank == 0:
        logging.info(f'Loading task {task_id}:\n{yaml.dump(task)}')
    tags = task.get('tags', [])
    config_files = task.get('config_files', [])
    config = task.get('config', {})
    config['task_id'] = task_id
    return tags, config_files, config


def _create_grid_task(spec, mpi_rank):
    task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    grid = spec['grid']
    task_count = np.prod([len(ax) for ax in grid])
    if task_id > task_count:
        raise ValueError(f'There are only {task_count} tasks, {task_id} was requested')
    selection = []
    i = task_id - 1  # One based task id
    for ax in grid:
        selection.append(ax[i % len(ax)])
        i //= len(ax)

    if mpi_rank == 0:
        logging.info(f'Loading grid selection {task_id} of {task_count}:\n{yaml.dump(selection)}')

    tags = []
    config_files = []
    config = dict(task_id=task_id)
    for ax in selection:
        tags.extend(ax.get('tags', []))
        config_files.extend(ax.get('config_files', []))
        config.update(flatten_dot_items(ax.get('config', {})))
    return tags, config_files, config


def _setup_config(mpi_rank, config, config_files, array_file, array_subset):
    tags = []
    config_files = config_files or [] # This is a list of strings now
    
    task_config = None
    if array_file is not None:
        processed_array_file = array_file
        if not os.path.isabs(array_file):
            # Try relative to SCRIPT_DIR first, then as given (relative to CWD)
            script_relative_path = os.path.join(SCRIPT_DIR, array_file)
            if os.path.exists(script_relative_path):
                processed_array_file = script_relative_path
            # If not found relative to script, assume it's relative to CWD or an already correct relative path
            elif not os.path.exists(array_file):
                 # As a last resort, if it's just a filename, assume it's in SCRIPT_DIR/configs/array
                potential_path = os.path.join(SCRIPT_DIR, 'configs', 'array', os.path.basename(array_file))
                if os.path.exists(potential_path):
                    processed_array_file = potential_path
                else:
                    # Log a warning if the file isn't found in expected relative locations
                    logging.warning(f"Array file {array_file} not found directly, nor relative to script dir {SCRIPT_DIR}, nor in {os.path.join(SCRIPT_DIR, 'configs', 'array')}. Trying to open as is.")
        
        try:
            with open(processed_array_file, mode='r') as f:
                spec = yaml.safe_load(f)
                if 'array' in spec:
                    t_tags, t_config_files_relative, t_config = _create_array_task(spec, mpi_rank, array_subset)
                elif 'grid' in spec:
                    t_tags, t_config_files_relative, t_config = _create_grid_task(spec, mpi_rank)
                else: # Handle case where array_file has unexpected structure
                    t_tags, t_config_files_relative, t_config = [], [], {}
                
                tags.extend(t_tags)
                # Process t_config_files_relative from array_file to be SCRIPT_DIR relative if needed
                for rel_cfg_file in t_config_files_relative:
                    if not os.path.isabs(rel_cfg_file):
                        # These paths from array spec are usually relative to 'configs/' or 'configs/array/'
                        # A common pattern is 'configs/something.yaml'
                        path_from_script_dir = os.path.join(SCRIPT_DIR, rel_cfg_file)
                        if os.path.exists(path_from_script_dir):
                           config_files.append(path_from_script_dir)
                        else:
                           # If not, assume it might be relative to the array_file's directory or just a name
                           # to be found under SCRIPT_DIR/configs/
                           base_dir_of_array_file = os.path.dirname(processed_array_file)
                           path_from_array_dir = os.path.join(base_dir_of_array_file, rel_cfg_file)
                           if os.path.exists(path_from_array_dir):
                               config_files.append(path_from_array_dir)
                           else:
                               # Fallback: treat as basename to be found in SCRIPT_DIR/configs
                               path_in_script_configs = os.path.join(SCRIPT_DIR, 'configs', os.path.basename(rel_cfg_file))
                               if os.path.exists(path_in_script_configs):
                                   config_files.append(path_in_script_configs)
                               else:
                                   logging.warning(f"Config file {rel_cfg_file} from array spec not found relative to script, array file, or in default configs. Trying as is.")
                                   config_files.append(rel_cfg_file) # Try as is if not found
                    else:
                        config_files.append(rel_cfg_file)
                task_config = t_config
        except FileNotFoundError:
            logging.error(f"Array file {processed_array_file} (derived from {array_file}) not found.")
            # Depending on desired behavior, you might want to raise an error or just proceed without it
            # For now, let's allow proceeding without it, task_config will remain None

    # The config_files list here now contains potentially absolute paths or paths made relative to SCRIPT_DIR
    # _merge_config will handle these. Note: _merge_config itself was also updated.
    # We pass an empty list for the 'config_files' argument to _merge_config initially,
    # as it prepends default.yaml and the paths from 'config_files' (now potentially absolute) are added inside.
    # The `config` argument to _merge_config is from args.config (command line --config KEY=VAL)
    # The `task_config` is from the array file.
    
    # The `config_files` list in `_setup_config` (which is `args.config_files`)
    # needs to be processed by `_merge_config` along with its internal default.yaml.
    # So, we pass `config_files` (from args) directly to `_merge_config`.
    # The list `t_config_files_relative` from array spec has been added to `config_files` already.
    
    merged_config_data = _merge_config(config, config_files, task_config)
    return merged_config_data, tags


def run(args):
    log_level = os.environ.get('LOGLEVEL', 'INFO').upper()
    tf_log_level = os.environ.get('TF_LOGLEVEL', 'WARN').upper()
    logging.basicConfig(level=log_level)
    tf.get_logger().setLevel(tf_log_level)
    logging.info('Launching')

    # Disable tensorflow GPU support
    tf.config.experimental.set_visible_devices([], "GPU")

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        config, tags = _setup_config(rank, args.config, args.config_files, args.array, args.subset)
        tags = tags + args.tags if args.tags else tags
        wandb.init(config=config, tags=tags, job_type=args.job_type)
        _save_log_to_wandb()
        wandb.summary.mpi_size = comm.Get_size()
    config = _sync_config(comm, rank)
    config = _update_ranked_config(config, rank)
    config = expand_dot_items(DotDict(config))
    GLOBAL_CONFIG.update(config)
    experiment = Experiment(config)
    entry_fn = getattr(experiment, config.call)
    entry_fn()


if __name__ == '__main__':
    run(config.parse_args())
