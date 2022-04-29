import os
BASE_DIR0 = "/home/zlc/cll/code/peclr"
BASE_DIR = "/home/zlc/cll/code/peclr_cbg"
# Data paths
DATA_PATH = BASE_DIR0+"/data/raw/"
FREIHAND_DATA = os.path.join(DATA_PATH, "freihand_dataset")
YOUTUBE_DATA = os.path.join(DATA_PATH, "youtube_3d_hands", "data")
YOUTUBE_DATA = "/media/d3-ai/E/cll/Dataset/YT3dHands"
# config paths
CONFIG_PATH = os.path.join(BASE_DIR, "src", "experiments", "config")
TRAINING_CONFIG_PATH = os.path.join(CONFIG_PATH, "training_config.json")
SUPERVISED_CONFIG_PATH = os.path.join(CONFIG_PATH, "supervised_config.json")
SIMCLR_CONFIG = os.path.join(CONFIG_PATH, "simclr_config.json")

SSL_CONFIG = os.path.join(CONFIG_PATH, "semi_supervised_config.json")
HYBRID2_CONFIG = os.path.join(CONFIG_PATH, "hybrid2_config.json")

DOWNSTREAM_CONFIG = os.path.join(CONFIG_PATH, "downstream_config.json")

ANGLES = [i for i in range(10, 360, 10)]
SAVED_MODELS_BASE_PATH = BASE_DIR+"/data/models/peclr"
SAVED_META_INFO_PATH =BASE_DIR+ "/data/models"
STD_LOGGING_FORMAT = "%(name)s -%(levelname)s - %(message)s"
COMET_KWARGS = {
    "api_key": '6c13icO732zNn3OMiPfrUcFSm',
    # "project_name":'Hybrid2-FreiYT3d',
    # "project_name":'Hybrid2-Frei',
    "project_name":'Hybrid2-Frei-cgbgr-debug',
    "workspace": 'peclr',
    "save_dir": SAVED_META_INFO_PATH,
}

# MANO mesh to joint matrix
MANO_MAT = os.path.join(
    BASE_DIR, "src", "data_loader", "mano_mesh_to_joints_mat.pth"
)

# BASE_DIR = os.environ.get("BASE_PATH")
# # Data paths
# DATA_PATH = os.environ.get("DATA_PATH")
# FREIHAND_DATA = os.path.join(DATA_PATH, "freihand_dataset")
# YOUTUBE_DATA = os.path.join(DATA_PATH, "youtube_3d_hands", "data")

# # config paths
# CONFIG_PATH = os.path.join(BASE_DIR, "src", "experiments", "config")
# TRAINING_CONFIG_PATH = os.path.join(CONFIG_PATH, "training_config.json")
# SUPERVISED_CONFIG_PATH = os.path.join(CONFIG_PATH, "supervised_config.json")
# SIMCLR_CONFIG = os.path.join(CONFIG_PATH, "simclr_config.json")

# SSL_CONFIG = os.path.join(CONFIG_PATH, "semi_supervised_config.json")
# HYBRID2_CONFIG = os.path.join(CONFIG_PATH, "hybrid2_config.json")

# DOWNSTREAM_CONFIG = os.path.join(CONFIG_PATH, "downstream_config.json")

# ANGLES = [i for i in range(10, 360, 10)]
# SAVED_MODELS_BASE_PATH = os.environ.get("SAVED_MODELS_BASE_PATH")
# SAVED_META_INFO_PATH = os.environ.get("SAVED_META_INFO_PATH")
# STD_LOGGING_FORMAT = "%(name)s -%(levelname)s - %(message)s"
# COMET_KWARGS = {
#     "api_key": os.environ.get("COMET_API_KEY"),
#     "project_name": os.environ.get("COMET_PROJECT"),
#     "workspace": os.environ.get("COMET_WORKSPACE"),
#     "save_dir": SAVED_META_INFO_PATH,
# }

# # MANO mesh to joint matrix
# MANO_MAT = os.path.join(
#     BASE_DIR, "src", "data_loader", "mano_mesh_to_joints_mat.pth"
# )
