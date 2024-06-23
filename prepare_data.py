import os
import subprocess
import json  
import shutil

def try_download_and_extract(data_root='data/WildAvatar', ytdl="lib/yt-dlp.exe"):
    splits = ["train.txt", "test.txt", "val.txt"]
    save_root = data_root
    
    for split in splits:
        human_list = os.path.join(data_root)
        with open(os.path.join(data_root, split), "r+") as f:
            human_list = f.readlines()
        human_list = [sub.strip() for sub in human_list]
        
        for video_id in human_list:
            meta_json = os.path.join(data_root, video_id, 'metadata.json')
            with open(meta_json, 'r', encoding='utf-8') as file:
                json_str = file.read()  
            data = json.loads(json_str)  
            start = data['start']
            img_ids = list(data.keys())[:-1]

            save_dir = os.path.join(save_root, video_id)
            ret = subprocess.run(
                [
                    ytdl,
                    "https://www.youtube.com/watch?v=" + video_id,
                    "-f",
                    "best",
                    "-f",
                    "mp4",
                    "-o",
                    os.path.join(save_dir, video_id+'.mp4'),
                ]
            )
            if not ret.returncode:
                frames_dir = os.path.join(save_dir, 'frames')
                os.makedirs(frames_dir, exist_ok=True)
                cmd = 'ffmpeg -i {videoname} -q:v 1 -ss {scene_start} -to {scene_end} -start_number {start_number} {outdir}/%06d.jpg'.format(
                        videoname=os.path.join(save_dir, video_id+'.mp4'),
                        scene_start=start,
                        scene_end=start+20,
                        start_number=0,
                        outdir=frames_dir
                    )
                os.system(cmd)
                
                img_name = [f'{img}.jpg' for img in img_ids]
                imgs_dir = os.path.join(save_dir, 'images')
                os.makedirs(imgs_dir, exist_ok=True)
                for img in img_name:
                    f1 = os.path.join(frames_dir,img)
                    f2 = os.path.join(imgs_dir,img)
                    shutil.copyfile(f1, f2) 
                shutil.rmtree(frames_dir)
            
def generate_new_splits(data_root='data/WildAvatar'):
    splits = ["train.txt", "test.txt", "val.txt"]
    for split in splits:
        human_list = os.path.join(data_root)
        with open(os.path.join(data_root, split), "r+") as f:
            human_list = f.readlines()
        human_list = [sub.strip() for sub in human_list]
        
        with open(os.path.join(data_root, "new_" + split), "w+") as f:
            for video_id in human_list:
                image_dir = os.path.join(data_root, video_id, "images")
                if not os.path.exists(image_dir):
                    continue
                if len(os.listdir(image_dir)) != 20:
                    continue
                f.writelines([video_id + "\n"])
            
if __name__ == "__main__":
    data_root='data/WildAvatar'
    try_download_and_extract(data_root=data_root)
    generate_new_splits(data_root=data_root)