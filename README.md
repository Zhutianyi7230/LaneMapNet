# LaneMapNet
The code implementation of LaneMapNet,  a unified end-to-end model designed for both HD Map Construction and Lane Network Recognization using a BEV perception framework. The project is based on mmdetection3d and MapTR.

| Method | Backbone | 2D to BEV | Config | Checkpoint |
| ------ | ------- |  -------- | -------- | -------- |
| LaneMapNet | ResNet50 | GKT | [config][1]| [model][2] |

[1]:https://github.com/Zhutianyi7230/LaneMapNet/blob/master/projects/configs/lanemapnet/lanemapnet_tiny_r50_24e_.py
[2]:https://pan.baidu.com/s/1_JJRWWOklYyfFwoYj7rKHA?pwd=c9va

# Visualization

![Fig.6](/Fig/Figure6.png "")
Visualization results for two scenarios on the NuScenes val split. Columns 2-4 represent road divider (red), road boundary (green), and pedestrian crossing (blue). Columns 5-8 use green and red dots to indicate the beginning and end of a lane line.

![Fig.7](/Fig/Figure7.png "")

A comprehensive presentation of HD map and lane network. The output of LaneMapNet is shown on the left and GT on the right.
