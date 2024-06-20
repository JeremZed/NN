__version__ = "0.0.1"
__debug_verbose__ = 1
__env_keras__ = "tensorflow"

__base_modules__ = [
    { 'name' : 'nnz.tools', 'as' : 'tools' },
    { 'name' : 'nnz.dataset', 'as' : 'dataset' },
    { 'name' : 'nnz.project.project', 'as' : 'project' },
    { 'name' : 'platform' },
    { 'name' : 'os' },
    { 'name' : 'json' },
    { 'name' : 'pandas', 'as' : 'pd' },
    { 'name' : 'pkg_resources'}
]

__path_filename_workspace__ = "info.pkl"
__path_dir_runtime__ = "./runtime"
__path_dir_runtime_notebooks__ = "./runtime/notebooks"
__path_dir_resources__ = "./resources"
__path_dir_models__ = "./resources/models"
__path_dir_datasets__ = "./resources/datasets"