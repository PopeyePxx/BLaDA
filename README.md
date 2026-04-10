# BLADA
[**BLaDA: Bridging Language to Functional Dexterous Actions within 3DGS Fields**](https://arxiv.org/pdf/2604.08410)
## Installation

1. **Create the environment**
   
   Set up a conda/mamba/micromamba environment for the project:
   ```bash
   micromamba create -n grasp_splats python=3.10 -c conda-forge
   micromamba activate grasp_splats
   ```

2. **Install part-level feature splatting**

   Clone the repository and install the required components for part-level feature splatting:
   ```bash
   git clone --recursive https://github.com/vuer-ai/feature-splatting-inria.git
   cd feature-splatting-inria
   git checkout roger/graspsplats_part

   # Install PyTorch and Torchvision with CUDA 11.8 support
   pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
   
   # Install CUDA Toolkit 11.8
   micromamba install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
   ```

   Next, set up the submodules and required dependencies:
   ```bash
   # Install diff-gaussian-rasterization submodule
   cd submodules/diff-gaussian-rasterization
   pip install -e .

   # Install simple-knn submodule
   cd ../..
   cd submodules/simple-knn
   pip install -e .

   # Install remaining requirements
   cd ../..
   pip install -r requirements.txt
   ```

   If any errors occur, try the following fixes:
   ```bash
   pip install numpy==1.23.5         # Downgrade to 'numpy<2' if there are compatibility issues
   pip install setuptools==69.5.1    # Resolves 'ImportError: cannot import name 'packaging' from 'pkg_resources''
   ```

4. **Install additional dependencies for grasping by query and visualization**

   Install the necessary Python packages for grasping by query and visualization:
   ```bash
   pip install viser==0.1.10 roboticstoolbox-python transforms3d 
   pip install panda_python   # Choose the version based on your Franka robot setup; any version works for UI-based runs
   ```

## Usage

1. **Compute features and train the model**
   
   To compute object part features and perform feature splatting training:
   ```bash
   python feature-splatting-inria/compute_obj_part_feature.py -s scene_data/example_data
   python feature-splatting-inria/train.py -s scene_data/example_data -m outputs/example_data --iterations 3000 --feature_type "clip_part"
   ```
   Increasing the number of iterations can improve reconstruction quality, but higher iteration counts are not required for successful grasping.

2. **Static scene grasping**

   For static grasping, run the following command:
   ```bash
   python realbot_ui.py -m outputs/example_data
   ```

   Then the UI would be on [http://0.0.0.0:8080](http://0.0.0.0:8080). Now you can use the UI to do text query and grasp sampling.
   - Input texts and click "Query" to segment the objects.
   - Click "Generate Global Grasps" to sample grasps in the whole scene, and then use "Filter with Gaussian" to clean grasps.
   - Click "Generate Object Grasps" to directly get grasps near the object by cropping the gaussians first.

## Custom Data

To use custom data, refer to [colmap_handeye](https://github.com/jimazeyu/colmap_handeye). This repository provides tools for dataset preparation and robot arm calibration. After obtaining the `world2base` transformation matrix, copy it into the code to align the point cloud or Gaussian splats with the robot’s coordinate frame:
   ```python
   world2base = np.array([
       [-0.4089165231525215, -0.8358961766325012, 0.3661486842582114, 0.42083348316217706],
       [-0.9105881302403995, 0.34730407737749247, -0.22407394962882685, 0.20879287837427596],
       [0.060137626808399375, -0.4250381861999404, -0.9031755123527864, 0.5594013590398528],
       [0.0, 0.0, 0.0, 1.0],
   ])
   ```
   This transformation converts the point cloud and Gaussian splats to the robot’s frame of reference for grasping tasks.

## Acknowledgements

The 3DGS code has been adapted from [GraspSplats]([https://github.com/atenpas/gpd](https://github.com/jimazeyu/GraspSplats)).

## Citation

If you find this project useful, please consider citing the following paper:
```bibtex
@misc{yang2026bladabridginglanguagefunctional,
      title={BLaDA: Bridging Language to Functional Dexterous Actions within 3DGS Fields}, 
      author={Fan Yang and Wenrui Chen and Guorun Yan and Ruize Liao and Wanjun Jia and Dongsheng Luo and Kailun Yang and Zhiyong Li and Yaonan Wang},
      year={2026},
      eprint={2604.08410},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2604.08410}, 
}
```
