# RL Algorithms + other baselines for Frozen Lake

## RL Algorithms
We setup [d3rlpy](https://github.com/takuseno/d3rlpy/tree/master) to train various offline RL algorithms for the frozen lake dataset.

### Setup

---
There were no clear instructions on how to setup d3rlpy to work with old atari datasets. Here's how I got around
to getting it finally work on Linux (Still haven't figured out on Windows)
1. Create conda environment with python=3.7
    ```conda create -n rl_baselines python=3.7```
2. Install gym==0.19.0 and atari_py==0.2.5. These specific versions are needed to successfully load the Atari data. 
This is also why we need python3.7 since it was not possible to install an older version of gym on python>=3.9. I also
could not install the older version of gym even with python3.7 on Windows
    ```pip install gym==0.19.0 atari_py==0.2.5 ale_py```
3. Install [d4rl-atari](https://github.com/takuseno/d4rl-atari) so that you can successfully load the data:
    ```pip install git+https://github.com/takuseno/d4rl-atari```
4. To install d3rlpy, use the conda install (the pip install did not work out in Linux but worked for Windows):
    ```conda install -c conda-forge d3rlpy```
5. Please note that the above command also installs Torch by default. However, it installs the cpu version by default. 
To override this installation and use the gpu version, install the following (after pip uninstall torch):
    ```pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113```
6. I specifically found that torch=1.12.1 version works the best with d3rlpy and the cuda version is 11.3 since currently that's the
cuda version on the Spocks-Brain server.