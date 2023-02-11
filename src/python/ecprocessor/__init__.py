submodules = [
    'tiltcorrections',
    'utils',
]

__all__ = submodules

def __dir__():
    return __all__

import importlib as _importlib
def __getattr__(name):
    if name in submodules:
        return _importlib.import_module(f'ecprocessor.{name}')
    else:
        try:
            return globals()[name]
        except KeyError:
            raise AttributeError(
                f"Module 'ecprocessor' has no attribute '{name}'"
            )
