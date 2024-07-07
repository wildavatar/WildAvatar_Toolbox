# WildAvatar for HumanNeRF
This repo is for WildAvatar, adapted from offical repo [HumanNeRF: Free-viewpoint Rendering of Moving People from Monocular Video (CVPR 2022)](https://github.com/chungyiweng/humannerf). Please prepare the environment as in their [offical instruction](https://github.com/chungyiweng/humannerf?tab=readme-ov-file#prerequisite).

## Additional Instruction
Below we take the subject **2WVreEPoLlg** as a running example.

1. Link the WildAvatar dataset, by running
```bash
    ln -s /path/to/WildAvatar ./datasets/WildAvatar
```
2. Training
```bash
    python train.py --cfg configs/human_nerf/WildAvatar/2WVreEPoLlg.yaml
```
3. Testing
```bash
    python run.py --cfg configs/human_nerf/WildAvatar/2WVreEPoLlg.yaml --type movement
```