import pickle
from pathlib import Path

from splg_slam.map.world_map import WorldMap


def save_map(world_map: WorldMap, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(world_map, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_map(path: str | Path) -> WorldMap:
    with open(path, "rb") as f:
        return pickle.load(f)
