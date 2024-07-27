import os
import subprocess
import argparse
import json
import shutil

def try_download_all(data_root='data/WildAvatar', ytdl="/bin/yt-dlp", output_root='data/WildAvatar/videos',raw=False):
    os.makedirs(output_root, exist_ok=True)
    with open("human_list.txt", "r+") as f:
        human_list = f.readlines()
    human_list = [sub.strip() for sub in human_list]
    
    for video_id in human_list:
        ret = subprocess.run(
            [
                ytdl,
                "https://www.youtube.com/watch?v=" + video_id,
                "-f",
                "best",
                "-f",
                "mp4",
                "-o",
                os.path.join(output_root, video_id+'.mp4'),
            ]
        )
        if (not ret.returncode) and (not raw):
            meta_json = os.path.join(data_root, video_id, 'metadata.json')
            with open(meta_json, 'r', encoding='utf-8') as file:
                json_str = file.read()  
            data = json.loads(json_str)  
            start = data['start']
            end = start + 20
            
            cmd = 'ffmpeg -i {videoname} -q:v 1 -ss {scene_start} -to {scene_end} -c copy {videoclipname}.mp4'.format(
                    videoname=os.path.join(output_root, video_id+'.mp4'),
                    scene_start=start,
                    scene_end=start+20,
                    videoclipname=os.path.join(output_root, "{}+{:06}+{:06}.mp4".format(video_id, start, end)),
                )
            os.system(cmd)
            os.remove(os.path.join(output_root, video_id+'.mp4'))
            
            
def parse_args():
    parser = argparse.ArgumentParser(description='parse_args')
    
    parser.add_argument('--ytdl', type=str, default="/bin/yt-dlp")
    parser.add_argument('--data_root', type=str, default="./data/WildAvatar")
    parser.add_argument('--output_root', type=str, default="./data/WildAvatar-videos")
    parser.add_argument('--raw', action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    try_download_all(data_root=args.data_root, ytdl=args.ytdl, output_root=args.output_root, raw=args.raw)