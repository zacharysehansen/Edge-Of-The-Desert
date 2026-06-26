import importlib.util as ipl
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
FUNC_NAME = "main"


def import_and_run_function(file_path: Path) -> None:
    module_name = file_path.stem
    spec = ipl.spec_from_file_location(module_name, file_path)
    module = ipl.module_from_spec(spec)
    spec.loader.exec_module(module)
    func = getattr(module, FUNC_NAME)
    func()


def main() -> None:
    for file in sorted(SCRIPT_DIR.iterdir()):
        if file.suffix != ".py" or file.name in {"run_phase1.py", "region.py"}:
            continue
        import_and_run_function(file)
        print(f"{file.name} completed")


if __name__ == "__main__":
    main()
