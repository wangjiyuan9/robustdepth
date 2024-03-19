# Self-supervised Monocular Depth Estimation: Let's Talk About The Weather (ICCV23)


 >**Self-supervised Monocular Depth Estimation: Let's Talk About The Weather
 >
 >[[Arxiv](https://arxiv.org/pdf/2307.08357.pdf)] [[Project Page](https://kieran514.github.io/Robust-Depth-Project/)]



<p align="center">
  <img src="assets/movie.gif" alt="Robustness" width="600" />
</p>

If you find our work useful in your research, kindly consider citing our paper:

```
@InProceedings{Saunders_2023_ICCV,
    author    = {Saunders, Kieran and Vogiatzis, George and Manso, Luis J.},
    title     = {Self-supervised Monocular Depth Estimation: Let's Talk About The Weather},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {8907-8917}
}
```


## Installation Setup

The models were trained using CUDA 11.1, Python 3.7.4 (conda environment), and PyTorch 1.8.0.

Create a conda environment with the PyTorch library:

```bash
conda env create --file environment.yml
conda activate robustdepth
```

## Datasets

### Training
We use the [KITTI dataset](http://www.cvlibs.net/download.php?file=raw_data_downloader.zip) and follow the downloading/preprocessing instructions set out by [Monodepth2](https://github.com/nianticlabs/monodepth2).
Download from scripts;
```
wget -i scripts/kitti_archives_to_download.txt -P data/KITTI_RAW/
```
Then unzip the downloaded files;
```
cd data/KITTI_RAW
unzip "*.zip"
cd ..
cd ..
```
Then convert all images to jpg;
```
find data/KITTI_RAW/ -name '*.png' | parallel 'convert -quality 92 -sampling-factor 2x2,1x1,1x1 {.}.png {.}.jpg && rm {}'
```

## Creating Augmentations For Any Dataset
Here, we have the flexibility to create any augmentations we desire before commencing the training process. Once we have generated the augmented data using the steps outlined below, we can proceed to train using only those augmented images.

**Creating augmentations can be extremely time-consuming, each augmentation section has an approximated time for the processes. Certain augmentations can be skipped to save time. However, these arguments must be removed in Robust-Depth/experiments/train_all.sh. For example, if you choose not to create the rain augmentation, --do_rain must be removed from Robust-Depth/experiments/train_all.sh**



#### File Format
```
├── KITTI_RAW
    ├── 2011_09_26
    ├── 2011_09_28
    │   ├── 2011_09_28_drive_0001_sync
    │   ├── 2011_09_28_drive_0002_sync
    |   │   ├── image_00
    |   │   ├── image_01
    |   │   ├── image_02
    |   │   |   ├── B
    |   │   |   ├── blur
    |   │   |   ├── data
    |   │   |   ├── dawn
    |   │   |   ├── dawn+fog
    |   │   |   ├── dawn+rain
    |   │   |   ├── dawn+rain+fog
    |   │   |   ├── defocus_blur
    |   │   |   ├── depth
    |   │   |   ├── dusk
    |   │   |   ├── dusk+fog
    |   │   |   ├── dusk+rain
    |   │   |   ├── dusk+rain+fog
    |   │   |   ├── elastic_transform
    |   │   |   ├── fog
    |   │   |   ├── fog+night
    |   │   |   ├── frost
    |   │   |   ├── G
    |   │   |   ├── gaussian_noise
    |   │   |   ├── glass_blur
    |   │   |   ├── greyscale
    |   │   |   ├── ground_snow
    |   │   |   ├── impulse_noise
    |   │   |   ├── jpeg_compression
    |   │   |   ├── night
    |   │   |   ├── pixelate
    |   │   |   ├── R
    |   │   |   ├── rain
    |   │   |   ├── rain+fog
    |   │   |   ├── rain+fog+night
    |   │   |   ├── rain_gan
    |   │   |   ├── rain+night
    |   │   |   ├── shot_noise
    |   │   |   ├── snow
    |   │   |   ├── zoom_blur
    |   │   |   ├── timestamps.txt
    |   |   ├── image_03
    |   │   ├── oxts
    |   │   ├── velodyne_points
    │   ├── calib_cam_to_cam.txt
    │   ├── calib_imu_to_velo.txt
    │   ├── calib_velo_to_cam.txt
    ├── 2011_09_29
    ├── 2011_09_30
    └── 2011_10_03
```
## Pretrained Models

| Model Name          | *Sunny* Abs_Rel | *Bad Weather* Abs_Rel | Model resolution  | Model  |
|-------------------------|-------------------|--------------------------|-----------------|------|
| [`ViT`](https://drive.google.com/drive/folders/13vtwYM2OHu4iBl6fSpdxXyjFZiI7K8_b?usp=sharing)         | 0.100 | 0.114 | 640 x 192                | ViT        |
| [`Resnet18`](https://drive.google.com/drive/folders/1x1dGzKhMY7k6Fq-47yU6Qa0bYtv7AgEH?usp=sharing)      | 0.115 | 0.133 | 640 x 192                |  Resnet18          |

## KITTI Ground Truth 

We must prepare ground truth files for validation and training.
```
python Robust-Depth/export_gt_depth.py --data_path data/KITTI_RAW --split eigen
python Robust-Depth/export_gt_depth.py --data_path KITTI_RAW --split eigen_zhou
python Robust-Depth/export_gt_depth.py --data_path KITTI_RAW --split eigen_benchmark
```

## Training

The models can be trained on the KITTI dataset by running: 
```
bash Robust-Depth/experiments/train_all.sh
```
The hyperparameters are defined in the script file and set at their defaults as stated in the paper.

To train with the vision transformer please add --ViT to train_all.sh and see MonoViT's repository for any issues.

Feel free to vary which augmentations are used.

### Adding your own augmentations

Finally, as Robust-Depth can have many further applications, we provide a simple step-by-step solution to train with one's own augmentations. Here we will add a near-infrared augmentation as an example. 

1. First create the augmentation on the entire KITTI dataset in the same format as above (in this case called NIR)
2. Enter Robust-Depth/options.py and add self.parser.add_argument("--do_NIR", help="NIR augmentation", action="store_true")
3. Inside Robust-Depth/trainer.py, add do_NIR_aug = self.opt.NIR to line 155 and line 181. Then add NIR:{self.opt.NIR} to line 233
4. Inside Robust-Depth/datasets/mono_dataset.py, add do_NIR_aug=False to line 70 and self.do_NIR_aug = do_NIR_aug to line 110
5. Inside Robust-Depth/datasets/mono_dataset.py, add 'NIR':self.do_NIR_aug to line 303 (where 'NIR' is the augmented images folder name)
6. Now inside the Robust-Depth/experiments/train_all.sh split add --do_NIR (removing other augmentations if you wish) and proceed with training

## Evaluation
We provide the evaluation for the KITTI dataset. If a ViT model is used as the weights, please use --ViT when evaluating below.

#### KITTI 

```
python Robust-Depth/evaluate_depth.py --load_weights_folder {weights_directory} --eval_mono --data_path data/KITTI_RAW --eval_split eigen
```
#### KITTI Benchmark

```
python Robust-Depth/evaluate_depth.py --load_weights_folder {weights_directory} --eval_mono --data_path data/KITTI_RAW --eval_split eigen_benchmark
```

#### KITTI Robust

```
python Robust-Depth/evaluate_depth.py --load_weights_folder {weights_directory} --eval_mono --data_path data/KITTI_RAW --eval_split eigen --robust_test
```

#### KITTI Benchmark Robust

```
python Robust-Depth/evaluate_depth.py --load_weights_folder {weights_directory} --eval_mono --data_path data/KITTI_RAW --eval_split eigen_benchmark --robust_test
```

#### KITTI Robust specific

```
python Robust-Depth/evaluate_depth.py --load_weights_folder {weights_directory} --eval_mono --data_path data/KITTI_RAW --eval_split eigen --robust_test --robust_augment blur
```
#### KITTI Benchmark Robust specific

```
python Robust-Depth/evaluate_depth.py --load_weights_folder {weights_directory} --eval_mono --data_path data/KITTI_RAW --eval_split eigen_benchmark --robust_test --robust_augment blur
```

### Out-Of-Distribution data

#### DrivingStereo
Download the "Different weathers" from the [DrivingStereo](https://drivingstereo-dataset.github.io/) into a folder called DrivingStereo. Specifically, download the depth-map-full-size and left-image-full-size. These extracted files should be placed inside of the weather condition folder, e.g. sunny. 

Next, we create ground truth depth data for the sunny weather conditions (there are four choices sunny, rainy, cloudy and foggy):
```
python Robust-Depth/export_gt_depth.py --data_path data/DrivingStereo --split sunny
```
Now we can run the evaluation:
```
python Robust-Depth/evaluate_depth.py --eval_mono --load_weights_folder {weights_directory} --data_path data/DrivingStereo --eval_split sunny
```

##### Testing
Evaluation for Cityscape Foggy and NuScenes-Night coming soon. 



## References

* [Monodepth2](https://github.com/nianticlabs/monodepth2) (ICCV 2019)
* [MonoViT](https://github.com/zxcqlf/MonoViT) 
* [Rain-Rendering](https://github.com/astra-vision/rain-rendering)
* [CoMoGAN](https://github.com/astra-vision/CoMoGAN)
* [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
* [robustness](https://github.com/hendrycks/robustness)
* [AutoMold](https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library)

