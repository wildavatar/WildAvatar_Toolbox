# Environments
```bash
conda create -n wildavatar python=3.9
conda activate wildavatar
pip install -r requirements.txt
pip install pyopengl==3.1.4
```

# Prepare Dataset
1. Download [WildAvatar.zip](https://zenodo.org/record/11526806/files/WildAvatar.zip)
2. Put the **WildAvatar.zip** under [./data/WildAvatar/](./data/WildAvatar/).
3. Unzip **WildAvatar.zip**
4. Install [yt-dlp](https://github.com/yt-dlp/yt-dlp)
1. Download and Extract images from YouTube, by running
```bash
python prepare_data.py --ytdl ${/path/to/yt-dlp}$
```

# Visualization
1. Put the [SMPL_NEUTRAL.pkl](https://smpl.is.tue.mpg.de/) under [./assets/](./assets/).
2. Run the following script to visualize the smpl overlay of the human subject of ${youtube_ID}
```bash
python vis_smpl.py --subject "${youtube_ID}"
```
3. The SMPL mask and overlay visualization can be found in [data/WildAvatar/\${youtube_ID}/smpl](data/WildAvatar/${youtube_ID}/smpl) and [data/WildAvatar/\${youtube_ID}/smpl_masks](data/WildAvatar/${youtube_ID}/smpl_masks)

For example, if you run
```bash
python vis_smpl.py --subject "__-ChmS-8m8"
```
The SMPL mask and overlay visualization can be found in [data/WildAvatar/__-ChmS-8m8/smpl](data/WildAvatar/__-ChmS-8m8/smpl) and [data/WildAvatar/__-ChmS-8m8/smpl_masks](data/WildAvatar/__-ChmS-8m8/smpl_masks)


# Using WildAvatar
We currently provide several examples to load our WildAvatar for [humannerf](./lib/humannerf), [gauhuman](./lib/gauhuman), [animatable_nerf](./lib/animatable_nerf) and [sherf](./lib/sherf).
