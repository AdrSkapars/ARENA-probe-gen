from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = REPO_ROOT / "data"


class SmartPath:
    def __init__(self, base_path):
        self.path = base_path
        self.path.mkdir(parents=True, exist_ok=True)

    def __truediv__(self, other):
        return str(self.path / other)

    def __str__(self):
        return str(self.path)


class data:
    probe_gen = SmartPath(REPO_ROOT / "src" / "probe_gen")
    data = SmartPath(DATA_DIR)
    
    refusal = SmartPath(DATA_DIR / "refusal")
    jailbreaks = SmartPath(DATA_DIR / "jailbreaks")
    
    lists = SmartPath(DATA_DIR / "lists")
    metaphors = SmartPath(DATA_DIR / "metaphors")
    science = SmartPath(DATA_DIR / "science")

    sycophancy_short = SmartPath(DATA_DIR / "sycophancy_short")
    sycophancy = SmartPath(DATA_DIR / "sycophancy")

    # # Unused commented so that doesnt create folders in data directory
    # formality = SmartPath(DATA_DIR / "formality")
    # lists_brazil = SmartPath(DATA_DIR / "lists_brazil")
    # refusal_brazil = SmartPath(DATA_DIR / "refusal_brazil")
    # metaphors_brazil = SmartPath(DATA_DIR / "metaphors_brazil")
    # science_brazil = SmartPath(DATA_DIR / "science_brazil")
    

