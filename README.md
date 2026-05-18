# Efficient Orthomosaicking

Efficient Large-Scale Orthomosaicking: Integrating Learning-based Visual Odometry with GNSS for Real-time Consistency

# Prerequisites
We have tested the library in **Ubuntu 20.04**, with the following hardware and software configurations:

- **CPU**: Intel Core i7-10700K
- **GPU**: NVIDIA GeForce RTX 3080
- **CUDA Version**: 11.8



## Pangolin
We use [Pangolin](https://github.com/stevenlovegrove/Pangolin) for visualization and user interface. Dowload and install instructions can be found at: https://github.com/stevenlovegrove/Pangolin.

## OpenCV
**Required at leat 3.0. Tested with OpenCV 3.4.1**.

## Eigen3
Required by g2o (see below). Download and install instructions can be found at: http://eigen.tuxfamily.org. **Required at least 3.1.0**.

## ONNXRuntime
**Required onnxruntime-linux-x64-gpu-1.16.3** and Modify line 63 of the CmakeLists.txt to the current location of ONNXRuntime library.




## Download Dbow File
Download ["voc_binary_tartan_8u_6.zip"](https://pan.baidu.com/s/1dd6k_Gf8mEjbiyli31_Yeg?pwd=p5y8), and unzip in Vocabulary/ .

# Building

Clone the repository:
```
git clone https://github.com/xiaozzzzzz-a/Efficient_Orthomosaicking.git
```


```
cd Efficient_Orthomosaicking
mkdir build
cd build
cmake ..
make -j12
```


# Running 

## Real-Time Orthomosaicking

The real-time orthomosaicking example is provided by `Examples/Monocular/mono_map_gnss`. It takes the visual vocabulary, a YAML configuration file, and an image/GNSS dataset directory:

```
./Examples/Monocular/mono_map_gnss \
    /path/to/Vocabulary/voc_binary_tartan_8u_6.yml.gz \
    Examples/Monocular/map2d_npu.yaml \
    /path/to/dataset/phantom3-npu-unified
```

### NPU Dataset Preparation

For the NPU dataset, the GNSS timestamps are not directly synchronized with the image timestamps, and the dataset does not provide public synchronization parameters. Before running the GNSS-assisted orthomosaicking pipeline, we first run ORB-SLAM3 in pure visual mode to estimate a visual trajectory and save `KeyFrameTrajectory.txt`.

Then, we align the GNSS measurements to the image frames using:

```
python3 script/align_gps_to_frames.py \
    --kf-traj KeyFrameTrajectory.txt \
    --gps /path/to/dataset/phantom3-npu-unified/gps.txt \
    --frames /path/to/dataset/phantom3-npu-unified/frames.txt \
    --output /path/to/dataset/phantom3-npu-unified/frames_with_gps_aligned.txt
```

The script searches for the best temporal offset between the visual keyframe trajectory and the GNSS trajectory, then interpolates GNSS measurements for the image frames and writes `frames_with_gps_aligned.txt`. The real-time mapping program reads this aligned file during startup, so the final pipeline is:

1. Run pure visual SLAM and save `KeyFrameTrajectory.txt`.
2. Generate `frames_with_gps_aligned.txt` with `script/align_gps_to_frames.py`.
3. Run `Examples/Monocular/mono_map_gnss` with `Examples/Monocular/map2d_npu.yaml`.

### Supplementary Video

Due to GitHub’s file size limitation, the supplementary video has been uploaded to Baidu Netdisk:

🔗 [Download Link](通过网盘分享的文件：Supplement video.mp4
链接: https://pan.baidu.com/s/15U_7hJ3vOTkgmle15L-v7A?pwd=xgpx 提取码: xgpx 复制这段内容后打开百度网盘手机App，操作更方便哦)




# Acknowledgments

The completion of this project would not have been possible without the support and contributions of the following open-source projects and tools. We extend our sincere gratitude to:

1. **ORB-SLAM3**  
   

2. **AIRVO**  
  
3. **SP-Loop**  

4. **ORB_SLAM3_detailed_comments**  

5. **SuperPoint_SLAM**

6. **Map2DFusion**
