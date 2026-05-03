import pathlib
import dotenv
from ip.utils.env_utils import OpenAIKeyRing, load_oai_keys

# Initialize directories
ROOT_DIR = pathlib.Path(__file__).parent.parent.resolve()
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATASETS_DIR = ROOT_DIR / "datasets"
DATASETS_DIR.mkdir(parents=True, exist_ok=True)
JOBS_DIR = ROOT_DIR / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENTS_DIR = ROOT_DIR / "experiments"
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

# Load API keys and create OpenAIKeyRing
oai_key_ring = OpenAIKeyRing(load_oai_keys(ROOT_DIR))

# Load env variables
dotenv.load_dotenv(ROOT_DIR / ".env")