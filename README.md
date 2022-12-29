# MS-Lesion-Segmentation

This repository contains the implementation of our paper [__A Dense Residual U-Net for Multiple Sclerosis Lesions Segmentation from Multi-Sequence 3D MR Images__](https://authors.elsevier.com/a/1gK2Q4xGJ-51nl).

In this study, we proposed a novel dense residual U-Net model which combines attention gate (AG), efficient channel attention (ECA), and Atrous Spatial Pyramid Pooling (ASPP) to enhance the performance of the automatic MS lesions segmentation. In addition, 3D MR images of FLAIR, T1-w, and T2-w were exploited jointly to perform better MS lesion segmentation.

_isbi_dense_res_u_net_ag_eca_aspp_summary.txt_ shows the model summary in detail.  The proposed model can be accessed in the _models_ folder as the JSON file.

Training weights for both datasets are available in the _weights_ folder.

The results obtained from both datasets are available in the _results_ folder.

## Generated NPY files
Generated NPY files from the ISBI2015 and MSSEG2016 training datasets should be placed to the dataset folder as follows:
```
dataset
|───isbi2015
|   | rater1_images_224.npy
|   | rater1_masks_224.npy
|   | rater2_images_224.npy
|   | rater2_masks_224.npy
|───msseg2016
|   | train_images_224.npy
|   | train_masks_224.npy
```
## How to train
To get all options, use -h or --help
```
python train.py -h
```
Here is the training example for the ISBI2015 dataset. Model name can be any of the model json file located in the _models_ folder.
```
python train.py --dataset="isbi2015" --model_name="isbi_dense_res_u_net_ag_eca_aspp" --epochs=300 --lr=0.0001
```

## How to cite:
If you use this repository, please cite this study as given:
```
  @article{sarica2022dense,
    title={A Dense Residual U-Net for Multiple Sclerosis Lesions Segmentation from Multi-Sequence 3D MR Images},
    author={Sarica, Beytullah and Seker, Dursun Zafer and Bayram, Bulent},
    journal={International Journal of Medical Informatics},
    pages={104965},
    year={2022},
    publisher={Elsevier}
  }
```
