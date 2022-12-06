# MS-Lesion-Segmentation

This repository contains the implementation of our paper [__A Dense Residual U-Net for Multiple Sclerosis Lesions Segmentation from Multi-Sequence 3D MR Images__](link to paper).

In this study, we proposed a novel dense residual U-Net model which combines attention gate (AG), efficient channel attention (ECA), and Atrous Spatial Pyramid Pooling (ASPP) to enhance the performance of the automatic MS lesions segmentation. In addition, 3D MR images of FLAIR, T1-w, and T2-w were exploited jointly to perform better MS lesion segmentation.

_isbi_dense_res_u_net_ag_eca_aspp_summary.txt_ shows the model summary in detail.  The proposed model can be accessed in the _models_ folder as the JSON file.

Training weights for both datasets are available in the _weights_ folder.

The results obtained from both datasets are available in the _results_ folder.

## How to cite:
If you use this repository, please cite this study as given:
```
  @article{author,
    title={},
    author={},
    journal={},
    volume={},
    year={},
    publisher={}
  }
```
