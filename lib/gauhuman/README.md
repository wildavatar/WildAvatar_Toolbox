# WildAvatar for GauHuman
This repo is for WildAvatar, adapted from offical repo [GauHuman: Articulated Gaussian Splatting from Monocular Human Videos](https://github.com/skhu101/GauHuman). Please prepare the environment as in their [offical instruction](https://github.com/skhu101/GauHuman?tab=readme-ov-file#desktop_computer-requirements).

## Additional Instruction
Below we take the subject **2WVreEPoLlg** as a running example.

1. Link the WildAvatar dataset, by running
```bash
    ln -s /path/to/WildAvatar ./data/WildAvatar
```
2. Training and Testing
```bash
    python train.py -s data/WildAvatar/2WVreEPoLlg
```