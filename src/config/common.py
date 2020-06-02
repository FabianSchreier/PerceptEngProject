# Imports


__author__ = 'Fabian Schreier'
__version__ = "0.1.0"
__status__ = "Prototype"

import os
from typing import Optional

import datasets


def get_job_id() -> Optional[int]:
    job_id_var = os.environ.get('SLURM_JOB_ID', None)
    if job_id_var is None:
        return None
    return int(job_id_var)


def default_dataset_root() -> str:
    job_id = get_job_id()
    if job_id is not None:
        return '/scratch/%d/ProcessedDatasets/' % job_id
    res = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', 'ProcessedDatasets'))
    return res


def default_output_folder() -> Optional[str]:
    job_id = get_job_id()
    if job_id is not None:
        return '/scratch/%d/Output/' % job_id
    return None

