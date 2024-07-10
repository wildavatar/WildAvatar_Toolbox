<h2 align="center" width="100%">
WildAvatar: Web-scale In-the-wild Video Dataset for 3D Avatar Creation
</h2>
<div>
<div align="center">
    <a href='https://inso-13.github.io/' target='_blank'>Zihao Huang</a><sup>1</sup>&emsp;
    <a href='https://skhu101.github.io/' target='_blank'>Shoukang Hu</a><sup>2</sup>&emsp;
    <a href='https://wanggcong.github.io/' target='_blank'>Guangcong Wang</a><sup>3</sup>&emsp;
    <a href='http://tqtqliu.github.io/' target='_blank'>Tianqi Liu</a><sup>1</sup><br>
    <a href='https://yuhangzang.github.io/' target='_blank'>Yuhang Zang</a><sup>4</sup>&emsp;
    <a href='http://faculty.hust.edu.cn/caozhiguo1/en/index.htm/' target='_blank'>Zhiguo Cao</a><sup>1</sup>&emsp;
    <a href='https://weivision.github.io/' target='_blank'>Wei Li</a><sup>2</sup>&emsp;
    <a href='https://liuziwei7.github.io/' target='_blank'>Ziwei Liu</a><sup>2</sup>
</div>
<div>
<div align="center">
    <sup>1</sup>Huazhong University of Science and Technology&emsp;
    <sup>2</sup>Nanyang Technological University<br>
    <sup>3</sup>Great Bay University&emsp;
    <sup>4</sup>Shanghai AI Laboratory
</div>

<p align="center">
  <a href="https://arxiv.org/pdf/2407.02165v2" target='_blank'>
    <img src="http://img.shields.io/badge/cs.CV-arXiv%3A2407.02165-B31B1B.svg" alt="ArXiv">
  </a>
  <a href="https://wildavatar.github.io/" target='_blank'>
    <img src="https://img.shields.io/badge/Project Page-%F0%9F%93%9a-lightblue" alt="Project Page">
  </a>
  <a href="https://youtu.be/ViAcrsq9Al8">
    <img src="https://img.shields.io/badge/YouTube-%23FF0000.svg?logo=YouTube&logoColor=white">
  </a>
  <a href="#">
    <img src="https://visitor-badge.laobi.icu/badge?page_id=wildavatar.WildAvatar_Toolbox" alt="Visitors">
  </a>
</p>

>**TL;DR**: <em>WildAvatar is a large-scale dataset from YouTube with 10,000+ human subjects, designed to address the limitations of existing laboratory datasets for avatar creation.</em>

## Environments
```bash
conda create -n wildavatar python=3.9
conda activate wildavatar
pip install -r requirements.txt
pip install pyopengl==3.1.4
```

## Prepare Dataset
1. Download [WildAvatar.zip](https://zenodo.org/record/11526806/files/WildAvatar.zip)
2. Put the **WildAvatar.zip** under [./data/WildAvatar/](./data/WildAvatar/).
3. Unzip **WildAvatar.zip**
4. Install [yt-dlp](https://github.com/yt-dlp/yt-dlp)
1. Download and Extract images from YouTube, by running
```bash
python prepare_data.py --ytdl ${PATH_TO_YT-DLP}$
```

## Visualization
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


## Using WildAvatar
For training and testing on WildAvatar, we currently provide the adapted code for [HumanNeRF](./lib/humannerf) and [GauHuman](./lib/gauhuman). 
