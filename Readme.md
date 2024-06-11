# Environments
```bash
conda create -n WA-tools python=3.9
conda activate WA-tools
```

# Prepare Dataset
1. Download the WildAvatar from [here](https://zenodo.org/record/11526806/files/WildAvatar.zip)
2. Put the *WildAvatar.zip* under the *./data*.
3. Unzip *WildAvatar.zip*
4. Download and Extract images from YouTube, using
```bash
python prepare_data.py
```

