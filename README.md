<!-- PROJECT LOGO -->

<p align="center">
  <p align="center">
    <img src="docs/logo.png"" alt="Logo" width="30%">
  </p>
  <h2 align="center">Generalizable Multi-view Hand Mesh Recovery</h2>
  <h3 align="center">Multi-view Hand Reconstruction with a Point-Embedded Transformer </h3>
  <p align="center">
    <a href="https://lixiny.github.io"><strong>Lixin Yang</strong></a>
    ·
    <a href="https://zlicheng.com"><strong>Licheng Zhong</strong></a>
    ·
    <a href="https://jubsteven.github.io"><strong>Pengxiang Zhu</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=WurpqEMAAAAJ&hl=en"><strong>Xinyu Zhan</strong></a>
    ·
    <a href=""><strong>Junxiao Kong</strong></a>
    .
    <a href="https://xjhaoren.github.io"><strong>Jian Xu</strong></a>
    .
    <a href="https://mvig.org"><strong>Cewu Lu</strong></a>
  </p>

  <p align="center">
    <a href="https://arxiv.org/abs/2408.10581"><img src='https://img.shields.io/badge/TPAMI-v2-blue?style=flat&logo=IEEE&logoWidth=20&logoColor=blue&labelColor=a2d2ff&color=792ee5' alt='TPAMI Paper'></a> 
    <a href='https://arxiv.org/abs/2304.04038'><img src='https://img.shields.io/badge/CVPR-v1-green?style=flat&logo=ieee&logoWidth=20&logoColor=white&labelColor=66cc00&color=94DD15' alt='CVPR Paper'></a>
  </p>
</p>


<table width="100%" cellpadding="0" cellspacing="0">
  <tr>
    <td colspan="2" align="center">
      <img src="./docs/demo_both.png" alt="both" width="100%" />
      <p><em>Supports reconstruction of both left and right hand.</em></p>
    </td>
  </tr>
  <tr>
    <td width="58%" align="center">
      <img src="docs/demo_single.png" alt="demo_single" width="100%" />
      <p><em>Absolute metric output and occlusion-robust.</em></p>
    </td>
    <td width="42%" align="center">
      <img src="docs/teleop_sim.gif" alt="teleop" width="100%" />
      <p><em>It support Human-hand teleoperation.</em></p>
    </td>
  </tr>
</table>

### What‘s _POEM-v2_?  
POEM (**PO**int-**EM**bed Multi-view Transformer) v2 is a generalizable **multi-view** hand mesh recovery model designed for seamless use in real-world hand MoCap & teleoperation. 

### What is _POEM-v2_'s advantage?
**It is flexible:** Works with any number, order or arrangement of cameras, as long as:  
  * share overlapping views, 
  * see the hand in at least some cameras, 
  * have calibrated extrinsics

**It is robust to occlusion:** 
It can handle occlusion and partial visibility by leveraging views where the hand remains visible.

**It produces absolute hand position:**  It directly recovers hand‐surface vertices in real‐world (meter) units, referenced to the first camera’s coordinate system.. 


**It supports both left and right hands:**: 
Although trained on right-hand data, it can still also handle left hand 
by a _world-mirroring process_ (horizon-tally flipping all images and mirroring camera extrinsics along the first camera's Y-Z plane)

### :joystick: Try me
We provide a **real-world demonstration** for running our model.   

Download the example data from [huggingface](https://huggingface.co/kelvin34501/POEM-v2_example_data/blob/main/example_data.tar.xz). 
The tarball includes **multi-view video of manipulaiton** captured in a laboratory setting, along with the corresponding camera **instrinsics and extrinsics**, hand poses, and hand's side information.

In the file `tool/infer_hand.py`, modify the path prefix (`/prefix/data/`) to the full path of the directory where the data has been extracted.

```python
DATA_FILEDIR = "/prefix/data/data" # Modify /prefix/data to where example data is extracted
MASK_FILEDIR = "/prefix/data/human_mask_hand"
CALIB_FILEDIR = "/prefix/data/calib/calib__2025_0319_1534_41"
HAND_SIDE_FILEPATH = "/prefix/data/hand_labels.json"
```

The visualize command (you need to install our env first)
```bash
python -m tool.infer_hand -c config/release/eval_single.yaml --reload ./checkpoints/medium.pth.tar -g 0
```
As a multiview method, camera **extrinsics mat** is crucial for POEM-v2 making prediction. 
In the `tool/infer_hand.py`, we require the N extrinsics matrices $T_{cw}$ in the SE(3) form: 
```math
P_c = T_{cw} \cdot P_w
```
where `c` indicats the camera coordinate system, `w` indicates the world, and $\mathbf{P}_c$ is the 3D points in camera system.

--- 

## :notebook: Instructions

- See [docs/installation.md](docs/installation.md) to setup the environment and install all the required packages.
- See [docs/datasets.md](docs/datasets.md) to download all the datasets and additional assets required.

&nbsp;

## :runner: Training and Evaluation

### Available models

We provide four models with different configurations for training and evaluation. We have evaluated the models on multiple datasets.

- set `${MODEL}` as one in `[small, medium, medium_MANO, large]`.
- set `${DATASET}` as one in `[HO3D, DexYCB, Arctic, Interhand, Oakink, Freihand]`.

Download the pretrained checkpoints at :link:[ckpt_release](https://drive.google.com/drive/folders/16BRH8zJ7fbR7QNluHHEshZMJc1wMRr_k?usp=drive_link) and move the contents to `./checkpoints`.

### Command line arguments

- `-g, --gpu_id`, visible GPUs for training, e.g. `-g 0,1,2,3`. evaluation only supports single GPU.
- `-w, --workers`, num_workers in reading data, e.g. `-w 4`.
- `-p, --dist_master_port`, port for distributed training, e.g. `-p 60011`, set different `-p` for different training processes.
- `-b, --batch_size`, e.g. `-b 32`, default is specified in config file, but will be overwritten if `-b` is provided.
- `--cfg`, config file for this experiment, e.g. `--cfg config/release/train_${MODEL}.yaml`.
- `--exp_id` specify the name of experiment, e.g. `--exp_id ${EXP_ID}`. When `--exp_id` is provided, the code requires that no uncommitted change is remained in the git repo. Otherwise, it defaults to 'default' for training and 'eval\_{cfg}' for evaluation. All results will be saved in `exp/${EXP_ID}*{timestamp}`.
- `--reload`, specify the path to the checkpoint (.pth.tar) to be loaded.

### Compare POEM-v2 vs Single-view methods on HO3D

To provide a holistic benchmark, we compare POEM-v2 with state-of-the-art **single-view** 3D hand recon-
struction frameworks. 
Since the absolute position of hands
is ambiguous in a single-view setting, we only report the
MPJPE and MPVPE under the Procrustes Alignment.

We perform this comparison on the **official HO3D test set** v2 and v3, now the testset GT can be download from the [official repo](https://github.com/shreyashampali/ho3d) (Update - Nov 3rd, 2024).
```shell
├── HO3D_v2 
├── HO3D_v2_official_gt 
│   ├── evaluation_verts.json
│   └── evaluation_xyz.json
├── HO3D_v3 
├── HO3D_v3_manual_test_gt 
    ├── evaluation_verts.json
    └── evaluation_xyz.json
```

Then run the following command to get the results:
```shell
# HO3D_VERSION can be set to 2 or 3,
$ python scripts/eval_ho3d_official.py  --ho3d-v ${HO3D_VERSION}
                                        --cfg config/release/eval_single.yaml 
                                        --model large  
                                        --reload ${PATH_TO_POEM_LARGE_CKPT} 
                                        --eval_extra ho3d_offi 
```
Then you can get the results reported in the paper: 
<table width="100%" cellpadding="0" cellspacing="0">
  <tr>
    <td width="50%" align="center" valign="top">
      <img src="docs/res/ho3dv2_res.png" alt="demo_single" width="100%" />
    </td>
    <td width="50%" align="center" valign="top">
      <img src="docs/res/ho3dv3_res.png" alt="teleop" width="100%" />
    </td>
  </tr>
</table>





### Evaluation

Specify the `${PATH_TO_CKPT}` to `./checkpoints/${MODEL}.pth.tar`. Then, run the following command. Note that we essentially modify the config file in place to suit different configuration settings. `view_min` and `view_max` specify the range of views fed into the model. Use `--draw` option to render the results, note that it is incompatible with the computation of `auc` metric.

```shell
$ python scripts/eval_single.py --cfg config/release/eval_single.yaml
                                -g ${gpu_id}
                                --reload ${PATH_TO_CKPT}
                                --dataset ${DATASET}
                                --view_min ${MIN_VIEW}
                                --view_max ${MAX_VIEW}
                                --model ${MODEL}
```

The evaluation results will be saved at `exp/${EXP_ID}_{timestamp}/evaluations`.

### Training

We have used the mixature of multiple datasets packed by webdataset for training. Excecute the following command to train a specific model on the provided dataset.

```shell
$ python scripts/train_ddp_wds.py --cfg config/release/train_${MODEL}.yaml -g 0,1,2,3 -w 4
```

### Tensorboard

```shell
$ cd exp/${EXP_ID}_{timestamp}/runs/
$ tensorboard --logdir .
```

### Checkpoint

All the checkpoints during training are saved at `exp/${EXP_ID}_{timestamp}/checkpoints/`, where `../checkpoints/checkpoint` records the most recent checkpoint.

&nbsp;


## License

This code and model are available for non-commercial scientific research purposes as defined in the [LICENSE](LICENSE) file. By downloading and using the code and model you agree to the terms in the [LICENSE](LICENSE).

## Citation

```bibtex
@misc{yang2024multiviewhandreconstructionpointembedded,
      title={Multi-view Hand Reconstruction with a Point-Embedded Transformer}, 
      author={Lixin Yang and Licheng Zhong and Pengxiang Zhu and Xinyu Zhan and Junxiao Kong and Jian Xu and Cewu Lu},
      year={2024},
      eprint={2408.10581},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.10581}, 
}
```

For more questions, please contact Lixin Yang: siriusyang@sjtu.edu.cn
