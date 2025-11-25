# il-opensource-template
![GitHub License](https://img.shields.io/github/license/IntelLabs/il-opensource-template)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/IntelLabs/il-opensource-template/badge)](https://scorecard.dev/viewer/?uri=github.com/IntelLabs/il-opensource-template)
<!-- UNCOMMENT AS NEEDED
[![Unit Tests](https://github.com/IntelLabs/ConvAssist/actions/workflows/run_unittests.yaml/badge.svg?branch=covassist-cleanup)](https://github.com/IntelLabs/ConvAssist/actions/workflows/run_unittests.yaml)
[![pytorch](https://img.shields.io/badge/PyTorch-v2.4.1-green?logo=pytorch)](https://pytorch.org/get-started/locally/)
![python-support](https://img.shields.io/badge/Python-3.12-3?logo=python)
-->

# LiveAvatar 

## Installation

Minimum tested requirements (older versions might still work):
- python3.12
- cuda12.6
- torch2.7
- Ubuntu24.04

### Install pyton packages
```
pip install -r requirements.txt
```

### Install INRIA diff-gaussian-renderer

https://github.com/graphdeco-inria/diff-gaussian-rasterization


### Install PyTorch3D
Follow the installation guide on https://pytorch3d.org.
```
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d && pip install .
```

If you encounter problems during install, try:
```
pip install . --no-build-isolation
```


### Training data setup

#### CelebV-HQ

Run face tracker to perform frame extraction, video matting, facial keypoint detection, and camera pose estimation on input videos in 'CelebV-HQ/35666':
```
python prepare_celebvhq.py --st 0 --None --show False --root ./data/datasets/CelebV-HQ --output_dir ./data/datasets/CelebV-HQ/processed_celebvhq
```

The directory *./CelebV-HQ* should have the following structure

```
CelebV-HQ/
- celebvhq_info.json
- 35666/
- processed_celebvhq/
    - frames/
    - alpha/
    - seg/
    - poses/
```

#### Option 1
Use config parameter 'celebvhq_root' to specify dataset location.

```
train.py --celebvhq_root=/mydata/CelebV-HQ
```


#### Option 2
Place training data in ./data/datasets or create symlink to pointing to training data location:
```
ln -s </foo/bar/data> ./data
```

with CelebV-HQ located in /foo/bar/data/datasets/CelebV-HQ.
