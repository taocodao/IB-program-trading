# Models module
from .multi_tenant import User, IBAccount, UserWatchlist, UserSettings, MultiTenantDB

# Re-export from parent models.py for backwards compatibility
import sys
from pathlib import Path
# Import the models.py file directly (not the package)
import importlib.util
_models_file = Path(__file__).parent.parent / "models.py"
if _models_file.exists():
    _spec = importlib.util.spec_from_file_location("_old_models", _models_file)
    _old_models = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_old_models)
    # Export the functions
    get_beta = _old_models.get_beta
    VolatilityTracker = _old_models.VolatilityTracker
    OptionPosition = _old_models.OptionPosition
    PositionStatus = _old_models.PositionStatus
    BETA_TABLE = _old_models.BETA_TABLE

