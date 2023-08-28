# Requirements

- Python=3.9
- pip=21.2
- pytorch=1.10
- torchvision=0.11
- cudatoolkit=11.3
- cupy=9.5
- h5py=3.3
- scikit-image=0.18
- scipy=1.7.1
- setuptools=59.5.0
- configargparse=1.5.3
- segmentation-models-pytorch=0.2
- tqdm=4.62
- tensorboard

# Dataset

To reproduce our results, create a data directory (e.g. `./data`) with the three datasets:

- **NYUv2**: Download the labeled dataset from [[here\]](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) and place the `nyu_depth_v2_labeled.mat` in `./data/NYU Depth v2`.
- **Middlebury**: Download the 2005-2014 scenes (full size, two-view) from [[here\]](https://vision.middlebury.edu/stereo/data/) and place the extracted scenes in `./data/Middlebury/<year>/<scene>`. For the 2005 dataset, make sure to only put the scenes for which ground truth is available. The data splits are defined in code.
- **DIML**: Download the indoor data sample from [[here\]](https://dimlrgbd.github.io/) and extract it into `./data/DIML/{train,test}` respectively. Then run `python scripts/create_diml_npy.py ./data/DIML` to create numpy binary files for faster data loading.

# Checkpoints

Our pretrained model checkpoints which were used for the numbers in our paper, for all three datasets and upsampling factors, are uploaded in the path`./checkpoint` of the repository.

# Train

As mentioned in the paper, the training of this model is divided into two steps:

For step 1:

```bash
python run_train.py --dataset <...> --data-dir <...> --scaling <...> --save-dir <...>
```

for step 2:

```bash
python run_train.py --dataset <...> --data-dir <...> --scaling <...> --save-dir <...> --step1_path <...> --step2
```



# Evaluation

For test set evaluation:

```bash
python run_eval.py --checkpoint <...> --dataset <...> --data-dir <...> --sc
```

