from pathlib import Path
import os
from shutil import copyfile
import re
from tqdm import tqdm


BASE_DIR = "B:\\MIT Places dataset\\test_256" # Places365-Standard test set path
TRAIN_DIR = Path("./images/train/class")
TEST_DIR = Path("./images/test/class")

TRAIN_RANGE = (41328, 50000)
TEST_RANGE = (60000, 75000)

all_files = os.listdir(BASE_DIR)
id_regex = re.compile(r'^Places365_test_(\d+).jpg$')
def get_id(file_name):
    id = re.match(id_regex, file_name)
    if id:
        return int(id.group(1).lstrip('0'))
    return None

print("Creating train dataset...")
TRAIN_DIR.parent.mkdir(parents=True, exist_ok=True)
TRAIN_DIR.mkdir(parents=True, exist_ok=True)
for file_name in tqdm(all_files[TRAIN_RANGE[0]:TRAIN_RANGE[1]+1], desc="Copying train files"):
    id = get_id(file_name)
    if id is not None and id >= TRAIN_RANGE[0] and id <= TRAIN_RANGE[1]:
        src_path = os.path.join(BASE_DIR, file_name)
        dst_path = os.path.join(TRAIN_DIR, file_name)
        copyfile(src_path, dst_path)


print("Creating test dataset...")
TEST_DIR.parent.mkdir(parents=True, exist_ok=True)
TEST_DIR.mkdir(parents=True, exist_ok=True)
for file_name in tqdm(all_files[TEST_RANGE[0]-1:TEST_RANGE[1]+1], desc="Copying test files"):
    id = get_id(file_name)
    if id is not None and id >= TEST_RANGE[0] and id <= TEST_RANGE[1]:
        src_path = os.path.join(BASE_DIR, file_name)
        dst_path = os.path.join(TEST_DIR, file_name)
        copyfile(src_path, dst_path)