import inspect
import json
from pathlib import Path
import importlib


class Base:
    def __init__(self, **kwargs):
        self._config = {}

    def capture_config(self, extra_config: dict | None = None) -> None:
        frame = inspect.currentframe().f_back
        _, _, _, values = inspect.getargvalues(frame)
        self._config = {"parameters": {}}
        for k, v in values.items():
            if k == "self" or k.startswith("_"):
                continue
            elif isinstance(v, dict):
                for k2, v2 in v.items():
                    self._config["parameters"][k2] = v2
            else:
                self._config["parameters"][k] = v
        self._config["class"] = self.__class__.__name__
        self._config["module"] = self.__class__.__module__
        self._config["config_name"] = None
        self._config.update(extra_config or {})

    @property
    def config(self):
        return self._config

    def base_save_config(self, config_name: str, save_dir: str | Path) -> None:
        assert isinstance(config_name, str) and len(config_name) > 0, (
            f"{config_name} is not a valid name, it should be a string with at least one character"
        )

        # 1. Resolve to get absolute path to guarantee correct directory creation
        save_path = Path(save_dir).resolve()
        save_path.mkdir(parents=True, exist_ok=True)
        file_path = save_path / f"{config_name}.json"

        # 2. Make the path relative to the project root (Current Working Directory)
        project_root = Path.cwd().resolve()
        try:
            # This strips the absolute system path and leaves just the project-relative path
            rel_file_path = file_path.relative_to(project_root)
            self._config["path"] = str(rel_file_path).replace("\\", "/") # Normalize slashes for JSON
        except ValueError:
            # Fallback to absolute if the save directory is completely outside the project folder
            self._config["path"] = str(file_path).replace("\\", "/")

        if (
            self._config["config_name"] is None
            and self._config["config_name"] != config_name
        ):
            self._config["config_name"] = config_name

        def json_serializer(obj):
            if hasattr(obj, "config"):
                if obj.config.get("path", None):
                    return obj.config["path"]
                raise ValueError(f"Serialization Failed: {obj} lacks a saved 'path' in its config.")
            if hasattr(obj, "__name__"):
                return obj.__name__
            return str(obj)

        if file_path.exists():
            with open(file_path, "r") as f:
                if json.load(f) == json.loads(
                    json.dumps(self._config, default=json_serializer)
                ):
                    return

        with open(file_path, "w") as f:
            json.dump(self._config, f, indent=4, default=json_serializer)

    @staticmethod
    def _load_obj(parent_cls: type['Base'], path: str | Path):
        config = json.loads(Path(path).read_text())
        module = importlib.import_module(".".join(config["module"].split(".")[:-1]))
        cls = getattr(module, config["class"])
        params = parent_cls._load_params(config["parameters"])
        obj = cls(**params)
        return obj

    @classmethod
    def _load_params(cls, val):
        """Recursively parses structures and loads objects from JSON paths."""
        if isinstance(val, str) and val.endswith(".json") and Path(val).exists():
            try:
                return cls._load_obj(cls, val)
            except Exception:
                return val  # Not a config path, just a standard string ending in .json
        elif isinstance(val, list):
            return [cls._load_params(item) for item in val]
        elif isinstance(val, dict):
            return {k: cls._load_params(v) for k, v in val.items()}
        return val

    @classmethod
    def from_config(cls, path: str | Path):
        config = json.loads(Path(path).read_text())
        params = cls._load_params(config["parameters"])
        return cls(**params)

    def save_config(self, config_name: str, save_dir: str | Path) -> None:
        self.base_save_config(config_name, save_dir)


class BaseConfig(Base):
    def __getitem__(self, item):
        return self._config["parameters"][item]
