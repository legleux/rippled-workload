import tomllib
import json
from pathlib import Path

pkg_root = Path(__file__).parent
config_file = pkg_root / "config.toml"
log_path = Path("logs")

# with config_file.open() as f_in:
#     conf_file = json.load(f_in)

cfg = tomllib.loads(Path(config_file).read_text())
fw = cfg["funding_account"]
cfg["funding_account"]["address"] = fw.get("address", "rHb9CJAWyB4rj91VRWn96DkukG4bwdtyTh" )
cfg["funding_account"]["seed"] = fw.get("seed", "snoPBrXtMeMyMHUVTgbuqAfg1SUTb" )
