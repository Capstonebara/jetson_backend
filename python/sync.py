import os
import shutil
import requests
import hashlib
import zipfile
import time

JETSON_EMBED_PATH = "/home/jetson/FaceRecognitionSystem/jetson/backend/embeddings/"
SERVER_SYNC_API = "http://api.fptuaiclub.me/sync-metadata"
SERVER_DOWNLOAD_API = "http://api.fptuaiclub.me/download-embeds/"
TIME_INTERVAL = 3  # seconds

# ANSI màu
def color(text, code):
    return f"\033[{code}m{text}\033[0m"

GREEN = "32"
YELLOW = "33"
RED = "31"
CYAN = "36"

def calc_local_md5(folder_path):
    md5 = hashlib.md5()
    for root, _, files in os.walk(folder_path):
        for file in sorted(files):
            path = os.path.join(root, file)
            with open(path, 'rb') as f:
                chunk = f.read(8192)
                while chunk:
                    md5.update(chunk)
                    chunk = f.read(8192)
    return md5.hexdigest()

def get_local_state():
    state = {}
    for folder in os.listdir(JETSON_EMBED_PATH):
        folder_path = os.path.join(JETSON_EMBED_PATH, folder)
        if os.path.isdir(folder_path):
            try:
                id_part = folder.split("_")[0]
                md5 = calc_local_md5(folder_path)
                state[id_part] = {
                    "folder": folder,
                    "md5": md5
                }
            except Exception as e:
                print(color(f"Error reading {folder}: {e}", YELLOW))
    return state

def download_and_extract(id, name):
    zip_name = f"{id}_[{name}].zip"
    folder_name = f"{id}_[{name}]"
    folder_path = os.path.join(JETSON_EMBED_PATH, folder_name)

    # XÓA THƯ MỤC CŨ TRƯỚC KHI GIẢI NÉN (nếu tồn tại)
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(color(f"Same id, same name, other MD5 -> Remove: {folder_name}", RED))

    url = f"{SERVER_DOWNLOAD_API}{id}" 
    r = requests.get(url)
    if r.status_code == 200:
        with open(zip_name, "wb") as f:
            f.write(r.content)
        with zipfile.ZipFile(zip_name, "r") as zip_ref:
            zip_ref.extractall(JETSON_EMBED_PATH)
        os.remove(zip_name)
        print(color(f"Downloaded & extracted: {zip_name}", GREEN))
    else:
        print(color(f"Failed to download {zip_name}: {r.status_code}", RED))

def sync():
    print(color("Starting sync...", CYAN))
    os.makedirs(JETSON_EMBED_PATH, exist_ok=True)

    server_data = requests.get(SERVER_SYNC_API).json()
    local_data = get_local_state()

    server_ids = {item["id"] for item in server_data}
    local_ids = set(local_data.keys())

    to_download = []
    to_rename = []
    to_delete = []

    for item in server_data:
        id = item["id"]
        name = item["name"]
        md5 = item["md5"]
        new_folder_name = f"{id}_[{name}]"

        if id not in local_data:
            to_download.append((id, name))
        else:
            local_md5 = local_data[id]["md5"]
            local_folder = local_data[id]["folder"]
            if local_md5 != md5:
                if local_folder != new_folder_name:
                    to_delete.append(local_folder)
                to_download.append((id, name))
            elif local_folder != new_folder_name:
                to_rename.append((local_folder, new_folder_name))

    for folder in to_delete:
        path = os.path.join(JETSON_EMBED_PATH, folder)
        if os.path.exists(path):
            shutil.rmtree(path)
            print(color(f"Deleted (due to change): {folder}", RED))

    for id, name in to_download:
        download_and_extract(id, name)

    for old_name, new_name in to_rename:
        os.rename(
            os.path.join(JETSON_EMBED_PATH, old_name),
            os.path.join(JETSON_EMBED_PATH, new_name)
        )
        print(color(f"Renamed: {old_name} -> {new_name}", YELLOW))

    removed_ids = list(local_ids - server_ids)
    for id in removed_ids:
        old_name = local_data[id]["folder"]
        shutil.rmtree(os.path.join(JETSON_EMBED_PATH, old_name))
        print(color(f"Deleted: {old_name}", RED))

    print(color("Sync complete.", GREEN))

if __name__ == "__main__":
    try:
        print(color("Starting sync loop. Press Ctrl+C to exit.", CYAN))
        while True:
            sync()
            print(color(f"Next sync in {TIME_INTERVAL} seconds...", CYAN))
            time.sleep(TIME_INTERVAL)
    except KeyboardInterrupt:
        print(color("\nSync loop stopped by user.", YELLOW))
    except Exception as e:
        print(color(f"Error in sync loop: {e}", RED))
