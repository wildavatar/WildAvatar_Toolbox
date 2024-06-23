import os
import json

def read_meta(subject, root_path):
    metadata = os.path.join(root_path, subject, "metadata.json")
    with open(metadata, "r+") as f:
        metadata = json.load(f)
    return metadata