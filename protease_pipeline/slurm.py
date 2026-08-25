"""Rendering of SLURM array-job submission scripts.

Every heavy stage of the pipeline writes a plain-text file with one shell
command per line (a "task list"), then submits a SLURM array job in which each
array task runs a contiguous slice of that file. This module centralises that
launcher so the individual stage scripts only have to describe resources.

The generated ``#SBATCH`` directives target a Slurm cluster; edit the resource
arguments (partition names, ``--gres`` values, wall time) to match your site.
"""

from __future__ import annotations

import math
import os

_TEMPLATE = """\
#!/bin/bash
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
{mail_line}#SBATCH -p {partition}
{gres_line}#SBATCH --mem={mem}
#SBATCH -c {cpus}
#SBATCH -J {job_name}
#SBATCH -o {log_dir}/out.log%a
#SBATCH -e {log_dir}/error.log%a
#SBATCH -t {time}
#SBATCH -a 1-{n_runs}

# Each array task runs {tasks_per_job} command(s) from the task list.
start=$(( (SLURM_ARRAY_TASK_ID - 1) * {tasks_per_job} + 1 ))
end=$(( start + {tasks_per_job} - 1 ))

for i in $(seq $start $end); do
    CMD=$(sed -n "${{i}}p" {command_file})
    [ -z "${{CMD}}" ] && break
    echo "Running task $i: ${{CMD}}"
    echo "${{CMD}}" | bash
done
"""


def render_array_script(
    command_file,
    log_dir,
    n_commands,
    *,
    job_name,
    partition,
    time,
    tasks_per_job=1,
    cpus=1,
    mem="4g",
    gres=None,
    mail_user=None,
):
    """Return the text of a SLURM array script for ``n_commands`` tasks."""
    n_runs = math.ceil(n_commands / tasks_per_job) if n_commands else 1
    gres_line = f"#SBATCH --gres={gres}\n" if gres else ""
    mail_line = f"#SBATCH --mail-user={mail_user}\n" if mail_user else ""
    return _TEMPLATE.format(
        mail_line=mail_line,
        partition=partition,
        gres_line=gres_line,
        mem=mem,
        cpus=cpus,
        job_name=job_name,
        log_dir=log_dir,
        time=time,
        n_runs=n_runs,
        tasks_per_job=tasks_per_job,
        command_file=command_file,
    )


def write_array_script(path, command_file, log_dir, n_commands, **kwargs):
    """Render and write a SLURM array script; returns its path."""
    os.makedirs(log_dir, exist_ok=True)
    with open(path, "w") as handle:
        handle.write(
            render_array_script(command_file, log_dir, n_commands, **kwargs)
        )
    print(f"SLURM submit script saved to {path}")
    return path
