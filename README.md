# SAGA
The official implementation of SAGA (Segment Any 3D GAussians). The paper is at [this url](https://jumpat.github.io/SAGA/SAGA_paper.pdf). Please refer to our [project page](https://jumpat.github.io/SAGA/) for more information. The code will be released soon.
<br>
<br>
<div align=center>
<img src="./imgs/teaser.png" width="500px">

SAGA can perform fine-grained interactive segmentation for 3D Gaussians within milliseconds.  
</div>
<br>
<br>
<div align=center>
<img src="./imgs/pipe.png" width="900px">
</div>
Given a pre-trained 3DGS model and its training set, we attach a low-dimensional 3D feature to each Gaussian in the model. For every image within the training set, we employ SAM to extract 2D features and a set of masks. Then we render 2D feature maps through the differentiable rasterization and train the attached features with two losses: i.e., the SAM-guidance loss and the correspondence loss. The former adopts SAM features to guide the 3D features to learn 3D segmentation from the ambiguous 2D masks. The latter distills the point-wise correspondence derived from the masks to enhance feature compactness.

# Installation
The installation of SAGA is similar to [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting).
```bash
git clone git@github.com:Jumpat/SegAnyGAussians.git
```
or
```bash
git clone https://github.com/Jumpat/SegAnyGAussians.git
```
Then install the dependencies:
```bash
conda env create --file environment.yml
conda activate gaussian_splatting
```


> **Note:** The code is tested on Ubuntu 18.04 with Python 3.7 and PyTorch 1.7.1.

## Citation
If you find this project helpful for your research, please consider citing the report and giving a ⭐.
```BibTex
@article{cen2023saga,
      title={Segment Any 3D Gaussians}, 
      author={Jiazhong Cen and Jiemin Fang and Chen Yang and Lingxi Xie and Xiaopeng Zhang and Wei Shen and Qi Tian},
      year={2023},
      journal={arXiv preprint arXiv:2312.00860},
}

