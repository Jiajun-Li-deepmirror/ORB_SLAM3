import pickle
from pathlib import Path

from splg_slam.map.world_map import WorldMap


def save_map(world_map: WorldMap, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(world_map, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_map(path: str | Path) -> WorldMap:
    """Unpickles a map file with pickle.load, which can execute arbitrary code embedded
    in the file - only ever call this on a map you (or a trusted pipeline run) produced
    with save_map, never on one fetched from a shared drive, network location, or
    another party's run without first establishing it's trustworthy."""
    with open(path, "rb") as f:
        return pickle.load(f)
