# Offroad_terrain_segmentation
![Graphical Abstract](https://github.com/user-attachments/assets/39daa23b-5e58-4268-ace1-f65f0ac18cd8)

Deep semantic segmentation model for identifying traversable terrain in off-road autonomous driving scenarios using a novel encoder-decoder architecture.

**Overview**
This repository presents a novel encoder-decoder structure for off-road semantic segmentation, specifically designed to identify traversable terrain for autonomous driving in challenging off-road environments. The work consists of two main components:

1. Enhanced Dataset: Derived from the Yamaha-CMU Off-Road dataset (YCOR) with comprehensive augmentation and enhancement. The dataset consists of 8 classes. The class labels are provided in the "class_dict_no_non_traversable_vegetation.csv" file.
2. Deep Learning Model: A custom encoder-decoder architecture optimized for off-road terrain classification

**Dataset Download**
The enhanced dataset can be downloaded from: https://purdue0-my.sharepoint.com/:u:/g/personal/arahi_purdue_edu/Efbz4TytH-NFs6VSgrqENKkBlXwNm82fsWaL21vFqYmbiA?e=bIuQXg

**Model Testing and Training**
The trained model can be tested directly using the "segmented_yamaha_augmented_8_class_sampled_data_512_3_resnet34.h5" weight file. The test.py file can also be used to test the model. Please ensure that the paths to the model file and the labels file are properly defined. 

If you are interested in training the model from scratch, you can follow the "resnet18+unet_augmented_combined_terrain.ipynb" notebook.


This is part of a published work (https://ieeexplore.ieee.org/abstract/document/10742323/). Detailed results and comparisons are presented in our published paper. 

If you use our work in your research, please cite our paper:

@article{rahi2024deep,
  title={Deep Semantic Segmentation for Identifying Traversable Terrain in Off-Road Autonomous Driving},
  author={Rahi, Adibuzzaman and Elgeoushy, Omar and Syed, Shazeb H and El-Mounayri, Hazim and Wasfy, Hatem and Wasfy, Tamer and Anwar, Sohel},
  journal={IEEE Access},
  year={2024},
  publisher={IEEE}
}
