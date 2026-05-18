/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/


#include "LocalMapping.h"
#include "System.h"
#include "LoopClosing.h"
#include "ORBmatcher.h"
#include "Matchers/SPmatcher.h"
#include "Optimizer.h"
#include "Converter.h"
#include "GeometricTools.h"
#include <opencv2/core/eigen.hpp>
#include<mutex>
#include<chrono>
#include <Eigen/Dense>
#include "../map2DFusion/RANSAC.h"
#include "../map2DFusion/DataTrans.h"
#include <torch/script.h>
#include <torch/torch.h>

#include <algorithm>
#include <iomanip>
namespace ORB_SLAM3
{

/**
 * @brief 局部地图线程构造函数
 * @param pSys 系统类指针
 * @param pAtlas atlas
 * @param bMonocular 是否是单目 (bug)用float赋值了
 * @param bInertial 是否是惯性模式
 * @param _strSeqName 序列名字，没用到
 */
LocalMapping::LocalMapping(System* pSys, Atlas *pAtlas, const float bMonocular, bool bInertial, const string &_strSeqName):
    mpSystem(pSys), mbMonocular(bMonocular), mbInertial(bInertial), mbResetRequested(false), mbResetRequestedActiveMap(false), mbFinishRequested(false), mbFinished(true), mpAtlas(pAtlas), bInitializing(false),
    mbAbortBA(false), mbStopped(false), mbStopRequested(false), mbNotStop(false), mbAcceptKeyFrames(true),
    mIdxInit(0), mScale(1.0),mspmatcher(0.0), mInitSect(0), mbNotBA1(true), mbNotBA2(true), mIdxIteration(0), infoInertial(Eigen::MatrixXd::Zero(9,9))
{

    /*
     * mbStopRequested:    外部线程调用，为true，表示外部线程请求停止 local mapping
     * mbStopped:          为true表示可以并终止localmapping 线程
     * mbNotStop:          true，表示不要停止 localmapping 线程，因为要插入关键帧了。需要和 mbStopped 结合使用
     * mbAcceptKeyFrames:  true，允许接受关键帧。tracking 和local mapping 之间的关键帧调度
     * mbAbortBA:          是否流产BA优化的标志位
     * mbFinishRequested:  请求终止当前线程的标志。注意只是请求，不一定终止。终止要看 mbFinished
     * mbResetRequested:   请求当前线程复位的标志。true，表示一直请求复位，但复位还未完成；表示复位完成为false
     * mbFinished:         判断最终LocalMapping::Run() 是否完成的标志。
     */
    mnMatchesInliers = 0;

    mbBadImu = false;

    mTinit = 0.f;
    //mspmatcher = SPmatcher(0.9);
    mNumLM = 0;
    mNumKFCulling=0;

    LoadSegmentationModel("/media/xiao/data2/image_stitching/mmsegmentation/testpt/segformer.pt");

    if(mpSystem && mpSystem->settings_ && mpSystem->UseGlobalMeas())
    {
        cout << "LocalMapping GNSS alignment frames init/finish: "
             << mpSystem->settings_->globalAlignInitFrames() << "/"
             << mpSystem->settings_->globalAlignFinishMeasurements() << endl;
    }
    

#ifdef REGISTER_TIMES
    nLBA_exec = 0;
    nLBA_abort = 0;
#endif

}

cv::Mat LocalMapping::RunSegmentation(const cv::Mat& img_bgr) {

  if (!mbModelLoaded) {
        std::cerr << "[ERROR] Segmentation model not loaded!" << std::endl;
        return cv::Mat();
    }
    
    cv::Mat img_rgb;
    cv::cvtColor(img_bgr, img_rgb, cv::COLOR_BGR2RGB);//到时候还要查查传进来的是不是bgr
    int orig_h = img_rgb.rows, orig_w = img_rgb.cols;
    cv::Mat resized_img;
    cv::resize(img_rgb, resized_img, cv::Size(512, 288));

    resized_img.convertTo(resized_img, CV_32FC3, 1.0 / 255.0);
    cv::Mat mean = (cv::Mat_<float>(1,3) << 123.675, 116.28, 103.53) / 255.0;
    cv::Mat std  = (cv::Mat_<float>(1,3) << 58.395, 57.12, 57.375) / 255.0;

    cv::Mat channels[3];
    cv::split(resized_img, channels);
    for(int i = 0; i < 3; ++i)
        channels[i] = (channels[i] - mean.at<float>(0,i)) / std.at<float>(0,i);
    cv::merge(channels, 3, resized_img);

    torch::Tensor tensor_image = torch::from_blob(resized_img.data, {1, 288, 512, 3}, torch::kFloat32).clone();
    tensor_image = tensor_image.permute({0, 3, 1, 2});

    // (5) Forward
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(tensor_image);
    at::Tensor output = mSegmentationModel.forward(inputs).toTensor();

     output = torch::nn::functional::interpolate(
        output,
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{288, 512})
            .mode(torch::kBilinear)
            .align_corners(false)
    );

    // (7) Upsample to original image size
    output = torch::nn::functional::interpolate(
        output,
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{orig_h, orig_w})
            .mode(torch::kBilinear)
            .align_corners(false)
    );

    // (8) Argmax
    at::Tensor pred = output.argmax(1).squeeze().to(torch::kU8).cpu();

    // (9) Convert to cv::Mat
    cv::Mat result(orig_h, orig_w, CV_8UC1, pred.data_ptr());
    result = result.clone();  // clone to detach from tensor memory

    return result;

}

bool LocalMapping::LoadSegmentationModel(const std::string& path) {

    try {
        mSegmentationModel = torch::jit::load(path);
        mSegmentationModel.eval();
        std::cout << "[INFO] Segmentation model loaded from " << path << std::endl;
        mbModelLoaded = true;
        return true;
    }

    catch (const c10::Error& e) {
        std::cerr << "[ERROR] Failed to load segmentation model: " << e.what() << std::endl;
        return false;
    }

}
/**
 * @brief 设置回环类指针
 * @param pLoopCloser 回环类指针
 */
void LocalMapping::SetLoopCloser(LoopClosing* pLoopCloser)
{
    mpLoopCloser = pLoopCloser;
}

/**
 * @brief 设置跟踪类指针
 * @param pLoopCloser 回环类指针
 */
void LocalMapping::SetTracker(Tracking *pTracker)
{
    mpTracker=pTracker;
}

/**
 * @brief 局部地图线程主函数
 */
void LocalMapping::Run()
{
    // 标记状态，表示当前run函数正在运行，尚未结束
    mbFinished = false;
    Eigen::Matrix3f Rg0w;
    int inum =0;
    bool initPlane = false;
    // 主循环
    while(1)
    {
        
        // Tracking will see that Local Mapping is busy
        // Step 1 告诉Tracking，LocalMapping正处于繁忙状态，请不要给我发送关键帧打扰我
        // LocalMapping线程处理的关键帧都是Tracking线程发过来的
        SetAcceptKeyFrames(false);
        // if(inum %50 ==0)
        //     cout<<"SetAcceptKeyFrames(false)"<<endl;
        // inum++;
        // Check if there are keyframes in the queue
        // 等待处理的关键帧列表不为空 并且imu正常
        if(CheckNewKeyFrames() && !mbBadImu)
        {
            //cout<<"localmapping begin0"<<endl;
#ifdef REGISTER_TIMES
            double timeLBA_ms = 0;
            double timeKFCulling_ms = 0;

            std::chrono::steady_clock::time_point time_StartProcessKF = std::chrono::steady_clock::now();
#endif
            // BoW conversion and insertion in Map
            // Step 2 处理列表中的关键帧，包括计算BoW、更新观测、描述子、共视图，插入到地图等
           //cout<<"ProcessNewKeyFrame()111"<<endl;
            ProcessNewKeyFrame();
            //cout<<"ProcessNewKeyFrame()222"<<endl;
#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_EndProcessKF = std::chrono::steady_clock::now();

            double timeProcessKF = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndProcessKF - time_StartProcessKF).count();
            vdKFInsert_ms.push_back(timeProcessKF);
#endif

            // Check recent MapPoints
            // Step 3 根据地图点的观测情况剔除质量不好的地图点
            
            MapPointCulling();
            //cout<<"step3: "<<mlNewKeyFrames.size()<<endl;
#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_EndMPCulling = std::chrono::steady_clock::now();

            double timeMPCulling = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndMPCulling - time_EndProcessKF).count();
            vdMPCulling_ms.push_back(timeMPCulling);
#endif

            // Triangulate new MapPoints
            // Step 4 当前关键帧与相邻关键帧通过三角化产生新的地图点，使得跟踪更稳
            CreateNewMapPoints();
            
            //cout<<"step4: "<<mlNewKeyFrames.size()<<endl;
            // 注意orbslam2中放在了函数SearchInNeighbors（用到了mbAbortBA）后面，应该放这里更合适
            mbAbortBA = false;

            // 已经处理完队列中的最后的一个关键帧
            if(!CheckNewKeyFrames())
            {
                // Find more matches in neighbor keyframes and fuse point duplications
                //  Step 5 检查并融合当前关键帧与相邻关键帧帧（两级相邻）中重复的地图点
                // 先完成相邻关键帧与当前关键帧的地图点的融合（在相邻关键帧中查找当前关键帧的地图点），
                // 再完成当前关键帧与相邻关键帧的地图点的融合（在当前关键帧中查找当前相邻关键帧的地图点）
                SearchInNeighbors();
            }
            //cout<<"step5: "<<mlNewKeyFrames.size()<<endl;
#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_EndMPCreation = std::chrono::steady_clock::now();

            double timeMPCreation = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndMPCreation - time_EndMPCulling).count();
            vdMPCreation_ms.push_back(timeMPCreation);
#endif

            bool b_doneLBA = false;
            int num_FixedKF_BA = 0;
            int num_OptKF_BA = 0;
            int num_MPs_BA = 0;
            int num_edges_BA = 0;
            
            
            // 已经处理完队列中的最后的一个关键帧，并且闭环检测没有请求停止LocalMapping
            if(!CheckNewKeyFrames() && !stopRequested())
            {

                //计算拟合平面
                std::vector<MapPoint*> mapPoints = mpCurrentKeyFrame->GetMap()->GetAllMapPoints();
                if(mapPoints.size() > 1000 && !initPlane && (mpCurrentKeyFrame->GetMap()->GetAllKeyFrames().size() > 10))
                {
                    for(MapPoint* pMP : mapPoints)
                    {
                        cv::Point3d p3d = cv::Point3d(
                        static_cast<double>(pMP->GetWorldPos()[0]),
                        static_cast<double>(pMP->GetWorldPos()[1]),
                        static_cast<double>(pMP->GetWorldPos()[2])
                        );
                        if (!ransac.is_finished())
                        {
                            std::thread ransac_thread(&RANSAC::solve, &ransac, p3d);
                            if (ransac_thread.joinable())
                                ransac_thread.join();
                        }
                        
                    }
                    initPlane = true;
                    planeVec.push_back(ransac.plane_P.x);
                    planeVec.push_back(ransac.plane_P.y);
                    planeVec.push_back(ransac.plane_P.z);
                    planeVec.push_back(ransac.plane_Q.unit_quaternion().x());
                    planeVec.push_back(ransac.plane_Q.unit_quaternion().y());
                    planeVec.push_back(ransac.plane_Q.unit_quaternion().z());
                    planeVec.push_back(ransac.plane_Q.unit_quaternion().w());
                    std::cout<<"init plane success"<<std::endl;
                }
                //cout<<"localmapping begin"<<endl;
                // 当前地图中关键帧数目大于2个
                if(mpAtlas->KeyFramesInMap()>2)
                {
                    // Step 6.1 处于IMU模式并且当前关键帧所在的地图已经完成IMU初始化
                    if(mbInertial && mpCurrentKeyFrame->GetMap()->isImuInitialized())
                    {
                        // 计算上一关键帧到当前关键帧相机光心的距离 + 上上关键帧到上一关键帧相机光心的距离
                        float dist = (mpCurrentKeyFrame->mPrevKF->GetCameraCenter() - mpCurrentKeyFrame->GetCameraCenter()).norm() +
                                (mpCurrentKeyFrame->mPrevKF->mPrevKF->GetCameraCenter() - mpCurrentKeyFrame->mPrevKF->GetCameraCenter()).norm();
                        // 如果距离大于5厘米，记录当前KF和上一KF时间戳的差，累加到mTinit
                        if(dist>0.05)
                            mTinit += mpCurrentKeyFrame->mTimeStamp - mpCurrentKeyFrame->mPrevKF->mTimeStamp;
                        // 当前关键帧所在的地图尚未完成IMU BA2（IMU第三阶段初始化）
                        if(!mpCurrentKeyFrame->GetMap()->GetIniertialBA2())
                        {
                            //如果累计时间差小于10s 并且 距离小于2厘米，认为运动幅度太小，不足以初始化IMU，将mbBadImu设置为true
                            // if((mTinit<10.f) && (dist<0.02))
                            // {
                            //     cout << "Not enough motion for initializing. Reseting..." << endl;
                            //     unique_lock<mutex> lock(mMutexReset);
                            //     mbResetRequestedActiveMap = true;
                            //     mpMapToReset = mpCurrentKeyFrame->GetMap();
                            //     mbBadImu = true;  // 在跟踪线程里会重置当前活跃地图
                            // }
                        }
                        // 判断成功跟踪匹配的点数是否足够多
                        // 条件---------1.1、跟踪成功的内点数目大于75-----1.2、并且是单目--或--2.1、跟踪成功的内点数目大于100-----2.2、并且不是单目
                        bool bLarge = ((mpTracker->GetMatchesInliers()>75)&&mbMonocular)||((mpTracker->GetMatchesInliers()>100)&&!mbMonocular);
                        // 局部地图+IMU一起优化，优化关键帧位姿、地图点、IMU参数
                        Optimizer::LocalInertialBA(mpCurrentKeyFrame, &mbAbortBA, mpCurrentKeyFrame->GetMap(),num_FixedKF_BA,num_OptKF_BA,num_MPs_BA,num_edges_BA, bLarge, !mpCurrentKeyFrame->GetMap()->GetIniertialBA2());
                        b_doneLBA = true;
                    }
                    else
                    {
                        
                        bool bLarge = ((mpTracker->GetMatchesInliers()>75)&&mbMonocular)||((mpTracker->GetMatchesInliers()>20)&&!mbMonocular);
                        const int globalAlignInitFrames = mpSystem->settings_ ? mpSystem->settings_->globalAlignInitFrames() : 80;
                        const int globalAlignFinishMeasurements = mpSystem->settings_ ? mpSystem->settings_->globalAlignFinishMeasurements() : 85;

                        if(mpTracker->GetGlobalFrameAlignmentState() == Tracking::FIRST_GLOBAL_MEAS_SET && inum>globalAlignInitFrames) //160
                        {
                            int size = mpCurrentKeyFrame->GetMap()->GetKFRelatedToGlobalOrigin()->GetGlobalPositionMeas().size();
                            
                            const GlobalPosition::GlobalPosition *origin = mpCurrentKeyFrame->GetMap()->GetKFRelatedToGlobalOrigin()->GetGlobalPositionMeas()[0];
                            
                            auto framePoses = GetGlobalSensorPoses(mpCurrentKeyFrame->GetMap(),mpCurrentKeyFrame->mGlobalMeasCalib.tbg, origin->timestamp);//获得gps在世界坐标系的坐标
                            Eigen::Matrix4f Twg0 = AlignGlobalFrame(origin, 
                                    mpTracker->GetDataToAlign(),
                                    framePoses);//获得世界坐标系和gps原点坐标系之间的变换
                            std::cout << "poses to align: " << mpTracker->GetDataToAlign().size() <<"framepose size: "<<framePoses.size()<< std::endl;
                            std::cout << "GNSS alignment refinement will finish when inum > "
                                      << globalAlignFinishMeasurements << std::endl;
                            cout<<"Twg0 transtition: "<<Twg0.topRightCorner(3,1).transpose()<<endl;

                            Eigen::Vector3f translation = Twg0.topRightCorner(3, 1);
                            Eigen::Matrix3f rotation = Twg0.block<3, 3>(0, 0);
                            float scale = std::cbrt(rotation.determinant());
                            rotation /= scale;
                            
                            Eigen::Matrix4f T_g0w;//sim3:Twg0的ni
                            T_g0w.block<3, 3>(0, 0) = (1/scale) * rotation.transpose(); // 逆旋转
                            T_g0w.topRightCorner<3, 1>() = (-1/scale) * (rotation.transpose() * translation); // 逆平移
                            T_g0w(3, 3) = 1; // 齐次坐标
                            
                            mpCurrentKeyFrame->GetMap()->Tg0wSim3 = T_g0w;
                            
                            std::cout << "Rotation matrix:\n" << rotation << std::endl;
                            std::cout << "Scale factor: " << scale << std::endl;

                            //Rg0w = Twg0.topLeftCorner(3,3).inverse();
                            Rg0w = rotation.inverse();
                            std::cout << "Translation vector (Twg0): " << translation.transpose() << std::endl;
                            mpCurrentKeyFrame->GetMap()->SetGlobalVIOAlignment(Rg0w);
                            mpCurrentKeyFrame->GetMap()->SetScaleVG(scale);
                            
                            Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame, &mbAbortBA, mpCurrentKeyFrame->GetMap(),num_FixedKF_BA,num_OptKF_BA,num_MPs_BA,num_edges_BA);                            
                            mpTracker->SetGlobalFrameAlignmentState(Tracking::ALIGNING);


                            if(initPlane)
                            {
                                // Sophus::SE3f se3f = mpCurrentKeyFrame->GetPose().inverse();
                                // Eigen::Matrix3f rotation_matrix_f = se3f.so3().matrix();
                                // Eigen::Vector3f translation_vector_f = se3f.translation();
                                // cv::Mat curImage = mpCurrentKeyFrame->imgOri.clone();
                                // Eigen::Matrix3d rotation_matrix_d = rotation_matrix_f.cast<double>();
                                // Eigen::Vector3d translation_vector_d = translation_vector_f.cast<double>();
                                // Sophus::SE3d se3d(Eigen::Quaterniond(rotation_matrix_d), translation_vector_d);
                                // std::pair<cv::Mat, Sophus::SE3d>  trans_frame(curImage, se3d);
                                // Trans.product(trans_frame);
                                // std::cout<<"trans keyframe success"<<std::endl;

                                ComputeAndSendFusionFrame();
                            }
                            inum++;
                        }
                        else if(mpTracker->GetGlobalFrameAlignmentState() == Tracking::ALIGNING)
                        {
                            cout<<"mpTracker->GetGlobalFrameAlignmentState() == Tracking::ALIGNING"
                                << " inum: " << inum
                                << "/" << globalAlignFinishMeasurements << endl;
                            
                            Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame,&mbAbortBA, mpCurrentKeyFrame->GetMap(),num_FixedKF_BA,num_OptKF_BA,num_MPs_BA,num_edges_BA);
                            
                            Optimizer::AlignmentRefinement(mpCurrentKeyFrame, &mbAbortBA, mpCurrentKeyFrame->GetMap(),num_FixedKF_BA,num_OptKF_BA,num_MPs_BA,num_edges_BA, false, bLarge, !mpCurrentKeyFrame->GetMap()->GetIniertialBA2());
                            
                            if(inum > globalAlignFinishMeasurements)
                            {
                                std::cout << "ALIGNED!" << std::endl;
                                mpTracker->SetGlobalFrameAlignmentState(Tracking::ALIGNED);
                                std::cout << "Keep map in visual frame after GNSS alignment." << std::endl;
                            }

                            //Optimizer::LocalBundleAdjustmentWithGlobalMeas(mpCurrentKeyFrame,&mbAbortBA, mpCurrentKeyFrame->GetMap(),num_FixedKF_BA,num_OptKF_BA,num_MPs_BA,num_edges_BA);
                            if(initPlane)
                            {
                                Sophus::SE3f se3f = mpCurrentKeyFrame->GetPose().inverse();
                                Eigen::Matrix3f rotation_matrix_f = se3f.so3().matrix();
                                Eigen::Vector3f translation_vector_f = se3f.translation();
                                cv::Mat curImage = mpCurrentKeyFrame->imgOri.clone();
                                Eigen::Matrix3d rotation_matrix_d = rotation_matrix_f.cast<double>();
                                Eigen::Vector3d translation_vector_d = translation_vector_f.cast<double>();
                                Sophus::SE3d se3d(Eigen::Quaterniond(rotation_matrix_d), translation_vector_d);
                                std::pair<cv::Mat, Sophus::SE3d>  trans_frame(curImage, se3d);
                                // Trans.product(trans_frame);

                                // ComputeAndSendRoofHeight();

                                ComputeAndSendFusionFrame();
                                
                            }
                            inum++;
                        }

                        else if(mpTracker->GetGlobalFrameAlignmentState() == Tracking::ALIGNED)
                        {
                            //Optimizer::AlignmentRefinement(mpCurrentKeyFrame, &mbAbortBA, mpCurrentKeyFrame->GetMap(),num_FixedKF_BA,num_OptKF_BA,num_MPs_BA,num_edges_BA, false, bLarge, !mpCurrentKeyFrame->GetMap()->GetIniertialBA2());
                            const GlobalPosition::GlobalPosition *origin = mpCurrentKeyFrame->GetMap()->GetKFRelatedToGlobalOrigin()->GetGlobalPositionMeas()[0];
                            
                            
                            // if(inum >200 && inum <240)
                            // {
                            //     Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame,&mbAbortBA, mpCurrentKeyFrame->GetMap(),num_FixedKF_BA,num_OptKF_BA,num_MPs_BA,num_edges_BA);
                            // }
                            // else{
                            //     Optimizer::LocalBundleAdjustmentWithGlobalMeas(mpCurrentKeyFrame,&mbAbortBA, mpCurrentKeyFrame->GetMap(),num_FixedKF_BA,num_OptKF_BA,num_MPs_BA,num_edges_BA);
                            // }
                            

                            Optimizer::LocalBundleAdjustmentWithGlobalMeas(mpCurrentKeyFrame,&mbAbortBA, mpCurrentKeyFrame->GetMap(),num_FixedKF_BA,num_OptKF_BA,num_MPs_BA,num_edges_BA);
            

                            //Optimizer::LocalBundleAdjustmentWithGlobalMeas(mpCurrentKeyFrame,&mbAbortBA, mpCurrentKeyFrame->GetMap(),num_FixedKF_BA,num_OptKF_BA,num_MPs_BA,num_edges_BA);
                            cout<<"mpTracker->GetGlobalFrameAlignmentState() == Tracking::ALIGNED"<<endl;
                            
                            //Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame,&mbAbortBA, mpCurrentKeyFrame->GetMap(),num_FixedKF_BA,num_OptKF_BA,num_MPs_BA,num_edges_BA);
                            b_doneLBA = true;


                            //cv::Mat curImage = mpCurrentKeyFrame->mImg.clone();
                            if(initPlane)
                            {
                                Sophus::SE3f se3f = mpCurrentKeyFrame->GetPose().inverse();
                                Eigen::Matrix3f rotation_matrix_f = se3f.so3().matrix();
                                Eigen::Vector3f translation_vector_f = se3f.translation();
                                cv::Mat curImage = mpCurrentKeyFrame->imgOri.clone();
                                Eigen::Matrix3d rotation_matrix_d = rotation_matrix_f.cast<double>();
                                Eigen::Vector3d translation_vector_d = translation_vector_f.cast<double>();
                                Sophus::SE3d se3d(Eigen::Quaterniond(rotation_matrix_d), translation_vector_d);
                                std::pair<cv::Mat, Sophus::SE3d>  trans_frame(curImage, se3d);
                                // Trans.product(trans_frame);

                                // ComputeAndSendRoofHeight();
                                ComputeAndSendFusionFrame();
                            }
                            inum++;
                            
                        }

                        // Step 6.2 不是IMU模式或者当前关键帧所在的地图还未完成IMU初始化
						// 局部地图BA，不包括IMU数据
						// 注意这里的第二个参数是按地址传递的,当这里的 mbAbortBA 状态发生变化时，能够及时执行/停止BA
                        // 局部地图优化，不包括IMU信息。优化关键帧位姿、地图点
                        else{

                            Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame,&mbAbortBA, mpCurrentKeyFrame->GetMap(),num_FixedKF_BA,num_OptKF_BA,num_MPs_BA,num_edges_BA);

                            if(initPlane)
                            {
                                Sophus::SE3f se3f = mpCurrentKeyFrame->GetPose().inverse();
                                Eigen::Matrix3f rotation_matrix_f = se3f.so3().matrix();
                                Eigen::Vector3f translation_vector_f = se3f.translation();
                                cv::Mat curImage = mpCurrentKeyFrame->imgOri.clone();
                                Eigen::Matrix3d rotation_matrix_d = rotation_matrix_f.cast<double>();
                                Eigen::Vector3d translation_vector_d = translation_vector_f.cast<double>();
                                Sophus::SE3d se3d(Eigen::Quaterniond(rotation_matrix_d), translation_vector_d);
                                std::pair<cv::Mat, Sophus::SE3d>  trans_frame(curImage, se3d);
                                // Trans.product(trans_frame);
                                // std::cout<<"trans keyframe success"<<std::endl;

                                // ComputeAndSendRoofHeight();

                                ComputeAndSendFusionFrame();
                            }
                            inum++;
                        }
                            //Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame,&mbAbortBA, mpCurrentKeyFrame->GetMap(),num_FixedKF_BA,num_OptKF_BA,num_MPs_BA,num_edges_BA);
                        
                            
                            
                           
                        
                        b_doneLBA = true;
                    }

                }
#ifdef REGISTER_TIMES
                std::chrono::steady_clock::time_point time_EndLBA = std::chrono::steady_clock::now();

                if(b_doneLBA)
                {
                    timeLBA_ms = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndLBA - time_EndMPCreation).count();
                    vdLBA_ms.push_back(timeLBA_ms);

                    nLBA_exec += 1;
                    if(mbAbortBA)
                    {
                        nLBA_abort += 1;
                    }
                    vnLBA_edges.push_back(num_edges_BA);
                    vnLBA_KFopt.push_back(num_OptKF_BA);
                    vnLBA_KFfixed.push_back(num_FixedKF_BA);
                    vnLBA_MPs.push_back(num_MPs_BA);
                }

#endif

                // Initialize IMU here
                // Step 7 当前关键帧所在地图未完成IMU初始化（第一阶段）
                //cout<<"Step 7 "<<endl;
                if(!mpCurrentKeyFrame->GetMap()->isImuInitialized() && mbInertial)
                {
                    //cout<<"mpCurrentKeyFrame->GetMap()->isImuInitialized()11"<<endl;
                    // 在函数InitializeIMU里设置IMU成功初始化标志 SetImuInitialized
                    // IMU第一次初始化
                    if (mbMonocular)
                        InitializeIMU(1e2, 1e10, true);
                    else
                        InitializeIMU(1e2, 1e5, true);
                     //cout<<"mpCurrentKeyFrame->GetMap()->isImuInitialized()22"<<endl;
                }


                // Check redundant local Keyframes
                // 跟踪中关键帧插入条件比较松，交给LocalMapping线程的关键帧会比较密，这里再删除冗余
                // Step 8 检测并剔除当前帧相邻的关键帧中冗余的关键帧
                // 冗余的判定：该关键帧的90%的地图点可以被其它关键帧观测到



                //cout<<"keyframeculling1"<<endl;
                //KeyFrameCulling();
                //cout<<"keyframeculling2"<<endl;
#ifdef REGISTER_TIMES
                std::chrono::steady_clock::time_point time_EndKFCulling = std::chrono::steady_clock::now();

                timeKFCulling_ms = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndKFCulling - time_EndLBA).count();
                vdKFCulling_ms.push_back(timeKFCulling_ms);
#endif
                // Step 9 如果距离IMU第一阶段初始化成功累计时间差小于100s，进行VIBA
                if ((mTinit<50.0f) && mbInertial)
                {
                    // Step 9.1 根据条件判断是否进行VIBA1（IMU第二次初始化）
                    // 条件：1、当前关键帧所在的地图还未完成IMU初始化---并且--------2、正常跟踪状态----------
                    if(mpCurrentKeyFrame->GetMap()->isImuInitialized() && mpTracker->mState==Tracking::OK) // Enter here everytime local-mapping is called
                    {
                        // 当前关键帧所在的地图还未完成VIBA 1
                        if(!mpCurrentKeyFrame->GetMap()->GetIniertialBA1()){
                            // 如果累计时间差大于5s，开始VIBA1（IMU第二阶段初始化）
                            if (mTinit>5.0f)
                            {
                                cout << "start VIBA 1" << endl;
                                mpCurrentKeyFrame->GetMap()->SetIniertialBA1();
                                if (mbMonocular)
                                    InitializeIMU(1.f, 1e5, true);
                                else
                                    InitializeIMU(1.f, 1e5, true);

                                cout << "end VIBA 1" << endl;
                            }
                        }
                        // Step 9.2 根据条件判断是否进行VIBA2（IMU第三次初始化）
                        // 当前关键帧所在的地图还未完成VIBA 2
                        else if(!mpCurrentKeyFrame->GetMap()->GetIniertialBA2()){
                            if (mTinit>15.0f){
                                cout << "start VIBA 2" << endl;
                                mpCurrentKeyFrame->GetMap()->SetIniertialBA2();
                                if (mbMonocular)
                                    InitializeIMU(0.f, 0.f, true);
                                else
                                    InitializeIMU(0.f, 0.f, true);

                                cout << "end VIBA 2" << endl;
                            }
                        }

                        // scale refinement
                        // Step 9.3 在关键帧小于100时，会在满足一定时间间隔后多次进行尺度、重力方向优化
                        if (((mpAtlas->KeyFramesInMap())<=200) &&
                                ((mTinit>25.0f && mTinit<25.5f)||
                                (mTinit>35.0f && mTinit<35.5f)||
                                (mTinit>45.0f && mTinit<45.5f)||
                                (mTinit>55.0f && mTinit<55.5f)||
                                (mTinit>65.0f && mTinit<65.5f)||
                                (mTinit>75.0f && mTinit<75.5f))){
                            if (mbMonocular)
                                // 使用了所有关键帧，但只优化尺度和重力方向以及速度和偏执（其实就是一切跟惯性相关的量）
                                ScaleRefinement();
                        }
                    }
                    
                }
            }

#ifdef REGISTER_TIMES
            vdLBASync_ms.push_back(timeKFCulling_ms);
            vdKFCullingSync_ms.push_back(timeKFCulling_ms);
#endif
            // Step 10 将当前帧加入到闭环检测队列中
           
            //mpLoopCloser->InsertKeyFrame(mpCurrentKeyFrame);
            //cout<<"localmapping end!"<<endl;
            
#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_EndLocalMap = std::chrono::steady_clock::now();

            double timeLocalMap = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndLocalMap - time_StartProcessKF).count();
            vdLMTotal_ms.push_back(timeLocalMap);
#endif
        }
        else if(Stop() && !mbBadImu) // 当要终止当前线程的时候
        {
            // Safe area to stop
            while(isStopped() && !CheckFinish())
            {
                // 如果还没有结束利索,那么等等它
                usleep(3000);
            }
            // 然后确定终止了就跳出这个线程的主循环
            if(CheckFinish())
                break;
        }
        // 查看是否有复位线程的请求
        ResetIfRequested();
        // Tracking will see that Local Mapping is busy
        // 开始接收关键帧
        SetAcceptKeyFrames(true);
        //cout<<"localmapping end000!"<<endl;
        //cout<<"SetAcceptKeyFrames(true)"<<endl;
        // 如果当前线程已经结束了就跳出主循环
        if(CheckFinish())
            break;

        usleep(2000);
    }

    // 设置线程已经终止
    SetFinish();
}


void LocalMapping::ComputeAndSendFusionFrame()
{
    if (!mpCurrentKeyFrame) {
        std::cerr << "[ERROR] No current KeyFrame!" << std::endl;
        return;
    }

    Sophus::SE3f se3f = mpCurrentKeyFrame->GetPose().inverse();
    Eigen::Matrix3f rotation_matrix_f = se3f.so3().matrix();
    Eigen::Vector3f translation_vector_f = se3f.translation();
    cv::Mat curImage = mpCurrentKeyFrame->imgOri.clone();
    Eigen::Matrix3d rotation_matrix_d = rotation_matrix_f.cast<double>();
    Eigen::Vector3d translation_vector_d = translation_vector_f.cast<double>();
    Sophus::SE3d se3d(Eigen::Quaterniond(rotation_matrix_d), translation_vector_d);

    if(1){
        Trans.product(FusionFrame(curImage, se3d));
        std::cerr << "[Debug] not combine segmentaion now !!!" << std::endl;
        return;
    }

    std::cerr << "[Debug] combine segmentaion now !!!" << std::endl;
    Eigen::Vector3d planePoint(planeVec[0], planeVec[1], planeVec[2]);

    // 2. 构造法向四元数
    Eigen::Quaterniond q(planeVec[6], planeVec[3], planeVec[4], planeVec[5]);
    q.normalize();
    Eigen::Vector3d planeNormal = q * Eigen::Vector3d(0, 0, 1);
    planeNormal.normalize();


    // 1. MapPoint高度统计
    std::vector<MapPoint*> kfMapPoints = mpCurrentKeyFrame->GetMapPointMatches();
    std::vector<double> heights;
    for (MapPoint* pMP : kfMapPoints) {
        if (!pMP || pMP->isBad()) continue;

        Eigen::Vector3d pt = pMP->GetWorldPos().cast<double>();
        double dist = std::abs(planeNormal.dot(pt - planePoint));
        heights.push_back(dist);
        //heights.push_back(static_cast<double>(pMP->GetWorldPos()[2]));
    }

    // if(heights.size() < 30){
    //     Trans.product(FusionFrame(curImage, se3d));
    //     cv::waitKey(1);
    //     std::cerr << "[WARN] Too few total map points (" << heights.size() << ")" << std::endl;
    //     return;
    // }

    

    double meanHeight = 0.0;
    for (const double& h : heights) meanHeight += h;
    meanHeight /= heights.size();

    double variance = 0.0;
    for (const double& h : heights) variance += (h - meanHeight) * (h - meanHeight);
    variance /= heights.size();

    //std::cout << "[DEBUG] Current KF Height Mean: " << meanHeight
              //<< " m, Variance: " << variance << " m^2" << std::endl;

    // // 2. 判断是否需要分割
    // const double heightVarianceThreshold = 5.0;
    // if (variance <= heightVarianceThreshold) {
    //     Trans.product(FusionFrame(curImage, se3d));
    //     std::cout << "[INFO] Height variance acceptable or plane already initialized. Skipping segmentation." << std::endl;
    //     return;
    // }
    
    // const double meanHeightThreshold = 0.07;
    // if (meanHeight <= meanHeightThreshold) {
    //     Trans.product(FusionFrame(curImage, se3d));
    //     cv::waitKey(1);
    //     std::cout << "[INFO] Height variance acceptable or plane already initialized. Skipping segmentation." << std::endl;
    //     return;
    // }

    // 3. 分割
    //std::cout << "[INFO] Height variance too high (" << variance << "), running segmentation..." << std::endl;
    cv::Mat segResult = RunSegmentation(curImage);
    if (segResult.empty() || segResult.size() != curImage.size()) {
        std::cerr << "[ERROR] Segmentation result invalid! Skipping this KF." << std::endl;
        return;
    }

    // 4. 筛选屋顶 MapPoints
    std::vector<double> roofHeights;
    const std::vector<cv::KeyPoint>& kfKeypoints = mpCurrentKeyFrame->mvKeys;
    for (size_t i = 0; i < kfMapPoints.size(); ++i) {
        MapPoint* pMP = kfMapPoints[i];
        if (!pMP || pMP->isBad()) continue;

        const cv::KeyPoint& kp = kfKeypoints[i];
        int u = static_cast<int>(std::round(kp.pt.x));
        int v = static_cast<int>(std::round(kp.pt.y));

        if (u < 0 || v < 0 || u >= segResult.cols || v >= segResult.rows) continue;
        uchar label = segResult.at<uchar>(v, u);
        if (label == 5) {  // 5 = 屋顶

            Eigen::Vector3d pt = pMP->GetWorldPos().cast<double>();
            double dist = std::abs(planeNormal.dot(pt - planePoint));
            roofHeights.push_back(dist);
        }
    }

    cv::Mat segVis(curImage.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    for (int y = 0; y < segResult.rows; ++y) {
        for (int x = 0; x < segResult.cols; ++x) {
            uchar label = segResult.at<uchar>(y, x);
            cv::Vec3b color;
            if (label == 0)       color = cv::Vec3b(0, 255, 0);     // 地面：绿色
            else if (label == 1)  color = cv::Vec3b(128, 128, 128); // 墙体：灰色
            else if (label == 5)  color = cv::Vec3b(0, 0, 255);     // 屋顶：红色
            else                  color = cv::Vec3b(0, 0, 0);       // 其他：黑色
            segVis.at<cv::Vec3b>(y, x) = color;
        }
    }

    // // 3.6 叠加到原图
    // cv::Mat blended;
    // cv::addWeighted(curImage, 0.5, segVis, 0.5, 0, blended);

    // // 3.7 显示
    // cv::imshow("Segmentation Visualization", blended);
    // cv::waitKey(1);

    // if (roofHeights.size() < 10) {
    //     std::cerr << "[WARN] Too few rooftop points detected (" << roofHeights.size() << ")" << std::endl;
    //     Trans.product(FusionFrame(curImage, se3d));
    //     return;
    // }

    double roofMeanHeight = 0.0;
    for (const double& h : roofHeights) roofMeanHeight += h;
    roofMeanHeight /= roofHeights.size();

    //std::cout << "[INFO] Estimated Rooftop Mean Height: " << roofMeanHeight
              //<< " m (from " << roofHeights.size() << " points)" << std::endl;


    // 5. 传给融合线程
    Trans.product(FusionFrame(curImage, se3d, roofMeanHeight, segResult));
}


Eigen::Matrix4f UmeyamaAlignment(std::list<Eigen::Vector3d> trajSrc, std::list<Eigen::Vector3d> trajDst)
{
    const int N = trajSrc.size();
    using Mat = Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>;

    Mat M1(3,N);
    Mat M2(3,N);

    std::list<Eigen::Vector3d>::iterator it1 = trajSrc.begin();
    std::list<Eigen::Vector3d>::iterator it2 = trajDst.begin();
    
    for(int i = 0; it1 != trajSrc.end() && it2 != trajDst.end(); ++it1, ++it2, i++)
    {
        M1.col(i) = *it1;
        M2.col(i) = *it2;
    }
    return Eigen::umeyama(M1,M2,true).cast<float>();
}

Eigen::Matrix4f ZhangYawAlignment(std::list<Eigen::Vector3d> trajSrc, std::list<Eigen::Vector3d> trajDst)
{
    const int N = trajSrc.size();
    using Mat = Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>;

    Mat M1(3,N);
    Mat M2(3,N);

    std::list<Eigen::Vector3d>::iterator it1 = trajSrc.begin();
    std::list<Eigen::Vector3d>::iterator it2 = trajDst.begin();
    
    for(int i = 0; it1 != trajSrc.end() && it2 != trajDst.end(); ++it1, ++it2, i++)
    {
        M1.col(i) = *it1;
        M2.col(i) = *it2;
    }
    Mat srcCentered = M1.colwise() - M1.rowwise().mean();
    Mat dstCentered = M2.colwise() - M2.rowwise().mean();
    Eigen::Matrix3d C = srcCentered * dstCentered.transpose();

    double a = C(0,1) - C(1,0);
    double b = C(0,0) + C(1,1);
    double yaw = M_PI / 2 - std::atan2(b,a);

    Eigen::Matrix4d R;
    R << std::cos(yaw), -std::sin(yaw), 0, 0,
         std::sin(yaw), std::cos(yaw), 0, 0,
         0, 0, 1, 0,
         0, 0, 0, 1;
    return R.cast<float>();

}

Eigen::Matrix4f LocalMapping::AlignGlobalFrame(const GlobalPosition::GlobalPosition* origin, list<const GlobalPosition::GlobalPosition*> gpsData, list<pair<double,Eigen::Vector3f>> framePosesData)
{    
    const double maxGpsInterpolationGap = 1.0;
    const double origin_t = origin->timestamp;
    std::list<Eigen::Vector3d> trajEstimated;
    std::list<Eigen::Vector3d> trajGlobalMeas;
    std::list<double> matchedPoseTimestamps;
    std::list<double> matchedGpsTimestamps;
    std::vector<std::pair<double, Eigen::Vector3d>> gpsLocal;

    for(const GlobalPosition::GlobalPosition* gpsMeas : gpsData)
    {
        if(!gpsMeas || gpsMeas->timestamp <= origin_t)
            continue;

        gpsLocal.emplace_back(gpsMeas->timestamp, gpsMeas->getRelativeFromOrigin(*origin));
    }

    gpsLocal.erase(std::unique(gpsLocal.begin(), gpsLocal.end(),
                               [](const std::pair<double, Eigen::Vector3d>& a,
                                  const std::pair<double, Eigen::Vector3d>& b)
                               {
                                   return a.first == b.first;
                               }),
                   gpsLocal.end());
    
    cout<<"framePoseData size: "<<framePosesData.size()<<endl;
    cout<<"gpsLocal size: "<<gpsLocal.size()<<endl;

    if(gpsLocal.size() < 2)
    {
        std::cerr << "Not enough GNSS samples after origin for interpolation." << std::endl;
        return Eigen::Matrix4f::Identity();
    }

    size_t gpsIdx = 0;
    int iters = 0;
    for(const auto& framePose : framePosesData)
    {
        iters++;
        const double poseTimestamp = framePose.first;
        if(poseTimestamp <= origin_t)
            continue;

        while(gpsIdx + 1 < gpsLocal.size() && gpsLocal[gpsIdx + 1].first < poseTimestamp)
            gpsIdx++;

        if(gpsIdx + 1 >= gpsLocal.size())
            break;

        const double t0 = gpsLocal[gpsIdx].first;
        const double t1 = gpsLocal[gpsIdx + 1].first;
        if(t1 <= t0)
            continue;

        if(poseTimestamp < t0 || poseTimestamp > t1)
            continue;

        if(t1 - t0 > maxGpsInterpolationGap)
            continue;

        const double ratio = (poseTimestamp - t0) / (t1 - t0);
        const Eigen::Vector3d gpsInterp = gpsLocal[gpsIdx].second +
                                          ratio * (gpsLocal[gpsIdx + 1].second - gpsLocal[gpsIdx].second);

        trajEstimated.push_back(framePose.second.cast<double>());
        trajGlobalMeas.push_back(gpsInterp);
        matchedPoseTimestamps.push_back(poseTimestamp);
        matchedGpsTimestamps.push_back(poseTimestamp);

        if(ratio > 0.5 && gpsIdx + 1 < gpsLocal.size())
            gpsIdx++;

    }
    
    // std::cout << "trajEstimated:" << std::endl;
    // for (const auto& pos : trajEstimated) {
    //     std::cout << pos.transpose() << std::endl; // 打印每个位置
    // }

    // 打印 trajGlobalMeas 信息
    // std::cout << "trajGlobalMeas:" << std::endl;
    // for (const auto& meas : trajGlobalMeas) {
    //     std::cout << meas.transpose() << std::endl; // 打印每个测量
    // }
    cout<<"iters: "<<iters<<endl;
    std::cout << "size gps: " << trajGlobalMeas.size() << std::endl;
    std::cout << "size poses: " << trajEstimated.size() << std::endl;
    if(trajGlobalMeas.size() < 3 || trajEstimated.size() < 3)
    {
        std::cerr << "Not enough matched GNSS/visual samples for alignment." << std::endl;
        return Eigen::Matrix4f::Identity();
    }
    Eigen::Matrix4f T = UmeyamaAlignment(trajGlobalMeas, trajEstimated);


    string summaryFilename = "compare3dpoint.txt";
    std::ofstream summaryFile(summaryFilename);
    if (!summaryFile.is_open()) {
        std::cerr << "Error opening file for writing." << std::endl;
    }
    summaryFile << "Global Measurements:" << std::endl;
    for (const auto& point : trajGlobalMeas) {
        summaryFile << point.transpose() << std::endl;
    }
    summaryFile << "Estimated Positions:" << std::endl;
    for (const auto& point : trajEstimated) {
        summaryFile << point.transpose() << std::endl;
    }
    summaryFile << "Transformation Matrix T:" << std::endl;
    summaryFile << T << std::endl;
    summaryFile.close();

    string pairsFilename = "alignment_pairs.csv";
    std::ofstream pairsFile(pairsFilename);
    if (!pairsFile.is_open()) {
        std::cerr << "Error opening alignment_pairs.csv for writing." << std::endl;
    }
    pairsFile << std::fixed << std::setprecision(9);
    pairsFile << "index,pose_ts,gps_ts,gps_x,gps_y,gps_z,est_x,est_y,est_z\n";
    std::list<Eigen::Vector3d>::const_iterator gpsIt = trajGlobalMeas.begin();
    std::list<Eigen::Vector3d>::const_iterator estIt = trajEstimated.begin();
    std::list<double>::const_iterator poseTsIt = matchedPoseTimestamps.begin();
    std::list<double>::const_iterator gpsTsIt = matchedGpsTimestamps.begin();
    int pairIndex = 0;
    while(gpsIt != trajGlobalMeas.end() &&
          estIt != trajEstimated.end() &&
          poseTsIt != matchedPoseTimestamps.end() &&
          gpsTsIt != matchedGpsTimestamps.end())
    {
        pairsFile << pairIndex << ","
                  << *poseTsIt << "," << *gpsTsIt << ","
                  << (*gpsIt)(0) << "," << (*gpsIt)(1) << "," << (*gpsIt)(2) << ","
                  << (*estIt)(0) << "," << (*estIt)(1) << "," << (*estIt)(2) << "\n";
        ++gpsIt;
        ++estIt;
        ++poseTsIt;
        ++gpsTsIt;
        ++pairIndex;
    }
    pairsFile.close();

    string transformFilename = "alignment_transform.txt";
    std::ofstream transformFile(transformFilename);
    if (!transformFile.is_open()) {
        std::cerr << "Error opening alignment_transform.txt for writing." << std::endl;
    }
    transformFile << T << std::endl;
    transformFile.close();

    std::cout << "Alignment data saved to "
              << summaryFilename << ", "
              << pairsFilename << " and "
              << transformFilename << std::endl;


    return T;//ZhangYawAlignment(trajGlobalMeas, trajEstimated);
}

list<pair<double,Eigen::Vector3f>> LocalMapping::GetGlobalSensorPoses(Map* pBiggerMap, Eigen::Vector3f tbg, double originTs)
{
    list<ORB_SLAM3::KeyFrame*>::iterator lRit = mpTracker->mlpReferences.begin();
    list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
    list<bool>::iterator lbL = mpTracker->mlbLost.begin();
    list<pair<double,Eigen::Vector3f>> poses;
    KeyFrame* pKF_origin = mpAtlas->GetCurrentMap()->GetKFRelatedToGlobalOrigin();
    cout<<"orign time: "<<originTs<<" pKF_origin"<<pKF_origin->mTimeStamp<<endl;
    int iter = 0;

    for(auto lit=mpTracker->mlRelativeFramePoses.begin(),
        lend=mpTracker->mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lT++, lbL++)
    {        
        if(*lbL)
        {
            cout<<"GetGlobalSensor Pose *lbL"<<endl;
            continue;
        }
           

        KeyFrame* pKF = *lRit;

        Sophus::SE3f Trw;

        // If the reference keyframe was culled, traverse the spanning tree to get a suitable keyframe.
        if (!pKF)
        {
            cout<<"GetGlobalSensor Pose !PKF"<<endl;
            continue;
        }
            

        while(pKF->isBad())
        {
            Trw = Trw * pKF->mTcp;
            pKF = pKF->GetParent();
        }

        if(!pKF || pKF->GetMap() != pBiggerMap)
        {
            cout<<"!pKF || pKF->GetMap() != pBiggerMap"<<endl;
            continue;
        }

        Trw = Trw * pKF->GetPose();

        if (mpTracker->mSensor == System::IMU_MONOCULAR || mpTracker->mSensor == System::IMU_STEREO || mpTracker->mSensor==System::IMU_RGBD)
        {
            Sophus::SE3f Twb = (pKF->mImuCalib.mTbc * (*lit) * Trw).inverse();
            Eigen::Matrix3f Rwb = Twb.rotationMatrix();
            Eigen::Vector3f twb = Twb.translation();
            Eigen::Vector3f p = Rwb * tbg + twb;
            //std::cout << *lT << " " << originTs << std::endl;
            if(*lT > originTs)
                poses.push_back(make_pair(*lT, p));
        }
        else
        {
            Sophus::SE3f Twc = ((*lit)*Trw).inverse();
            Sophus::SE3f Tow = pKF_origin->GetPose();
            Sophus::SE3f Toc = Tow*Twc; 
            Eigen::Matrix3f Rwc = Toc.rotationMatrix();
            Eigen::Vector3f twc = Toc.translation();
            Eigen::Vector3f p = twc;
            //std::cout << *lT << " " << originTs << std::endl;
            if(*lT > originTs)
                poses.push_back(make_pair(*lT, p));

            // Sophus::SE3f Twc = ((*lit)*Trw).inverse();
            // Eigen::Matrix3f Rwc = Twc.rotationMatrix();
            // Eigen::Vector3f twc = Twc.translation();
            // Eigen::Vector3f p = Rwc * tbg + twc;
            // //std::cout << *lT << " " << originTs << std::endl;
            // if(*lT > originTs)
            //     poses.push_back(make_pair(*lT, p));
        }
        iter++;
    }
    cout<<"GetGlobalSensorPoses iter "<<iter<<"pose size: "<<poses.size()<<endl;
    return poses;//获得gps传感器在世界坐标系中的坐标
}

/**
 * @brief 插入关键帧,由外部（Tracking）线程调用;这里只是插入到列表中,等待线程主函数对其进行处理
 * @param pKF 新的关键帧
 */
void LocalMapping::InsertKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexNewKFs);
    // 将关键帧插入到列表中
    //cout<<"LM:mlNewKeyFrame size "<<mlNewKeyFrames.size()<<endl;
    mlNewKeyFrames.push_back(pKF);
    mbAbortBA=true;
    
}

/**
 * @brief 查看列表中是否有等待被插入的关键帧
 */
bool LocalMapping::CheckNewKeyFrames()
{
   
    unique_lock<mutex> lock(mMutexNewKFs);
    return(!mlNewKeyFrames.empty());
}

/**
 * @brief 处理列表中的关键帧，包括计算BoW、更新观测、描述子、共视图，插入到地图等
 */
void LocalMapping::ProcessNewKeyFrame()
{
    // Step 1：从缓冲队列中取出一帧关键帧
    // 该关键帧队列是Tracking线程向LocalMapping中插入的关键帧组成
    {
        unique_lock<mutex> lock(mMutexNewKFs);
        // 取出列表中最前面的关键帧，作为当前要处理的关键帧
        mpCurrentKeyFrame = mlNewKeyFrames.front();
        // 取出最前面的关键帧后，在原来的列表里删掉该关键帧
        mlNewKeyFrames.pop_front();
        //cout<<"after pop_front mlNewKeyFrames size: "<<mlNewKeyFrames.size()<<endl;
    }

    // Compute Bags of Words structures
    // Step 2：计算该关键帧特征点的Bow信息
    //mpCurrentKeyFrame->ComputeBoW();
    mpCurrentKeyFrame->ComputeBoW3();
    
    // Associate MapPoints to the new keyframe and update normal and descriptor
    // Step 3：当前处理关键帧中有效的地图点，更新normal，描述子等信息
    // TrackLocalMap中和当前帧新匹配上的地图点和当前关键帧进行关联绑定
    const vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    // 对当前处理的这个关键帧中的所有的地图点展开遍历
    for(size_t i=0; i<vpMapPointMatches.size(); i++)
    {
        MapPoint* pMP = vpMapPointMatches[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                if(!pMP->IsInKeyFrame(mpCurrentKeyFrame))
                {
                    // 如果地图点不是来自当前帧的观测，为当前地图点添加观测
                    pMP->AddObservation(mpCurrentKeyFrame, i);
                    // 获得该点的平均观测方向和观测距离范围
                    pMP->UpdateNormalAndDepth();
                    // 更新地图点的最佳描述子
                    pMP->ComputeDistinctiveDescriptors();
                }
                else // this can only happen for new stereo points inserted by the Tracking
                {
                    // 如果当前帧中已经包含了这个地图点,但是这个地图点中却没有包含这个关键帧的信息
                    // 这些地图点可能来自双目或RGBD跟踪过程中新生成的地图点，或者是CreateNewMapPoints 中通过三角化产生
                    // 将上述地图点放入mlpRecentAddedMapPoints，等待后续MapPointCulling函数的检验
                    mlpRecentAddedMapPoints.push_back(pMP);
                }
            }
        }
    }

    // Update links in the Covisibility Graph
    // Step 4：更新关键帧间的连接关系（共视图）
    mpCurrentKeyFrame->UpdateConnections();

    // Insert Keyframe in Map
    // Step 5：将该关键帧插入到地图中
    mpAtlas->AddKeyFrame(mpCurrentKeyFrame);
}

/**
 * @brief 处理新的关键帧，使队列为空，注意这里只是处理了关键帧，并没有生成MP
 */
void LocalMapping::EmptyQueue()
{
    while(CheckNewKeyFrames())
        ProcessNewKeyFrame();
}

/**
 * @brief 检查新增地图点，根据地图点的观测情况剔除质量不好的新增的地图点
 * mlpRecentAddedMapPoints: 存储新增的地图点，这里是要删除其中不靠谱的
 */
void LocalMapping::MapPointCulling()
{
    // Check Recent Added MapPoints
    list<MapPoint*>::iterator lit = mlpRecentAddedMapPoints.begin();
    const unsigned long int nCurrentKFid = mpCurrentKeyFrame->mnId;

    // Step 1：根据相机类型设置不同的观测阈值
    int nThObs;
    if(mbMonocular)
        nThObs = 2;
    else
        nThObs = 3;
    const int cnThObs = nThObs;

    int borrar = mlpRecentAddedMapPoints.size();
    // Step 2：遍历检查的新添加的MapPoints
    while(lit!=mlpRecentAddedMapPoints.end())
    {
        MapPoint* pMP = *lit;

        if(pMP->isBad())
            // Step 2.1：已经是坏点的MapPoints直接从检查链表中删除
            lit = mlpRecentAddedMapPoints.erase(lit);
        else if(pMP->GetFoundRatio()<0.05f)//0.25
        {
            // Step 2.2：跟踪到该MapPoint的Frame数相比预计可观测到该MapPoint的Frame数的比例小于25%，删除
            // (mnFound/mnVisible） < 25%
            // mnFound ：地图点被多少帧（包括普通帧）看到，次数越多越好
            // mnVisible：地图点应该被看到的次数
            // (mnFound/mnVisible）：对于大FOV镜头这个比例会高，对于窄FOV镜头这个比例会低
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=2 && pMP->Observations()<=cnThObs)
        {
            // Step 2.3：从该点建立开始，到现在已经过了不小于2个关键帧
            // 但是观测到该点的关键帧数却不超过cnThObs帧，那么删除该点
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        // Step 2.4：从建立该点开始，已经过了3个关键帧而没有被剔除，则认为是质量高的点
        // 因此没有SetBadFlag()，仅从队列中删除，放弃继续对该MapPoint的检测
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=3)
            lit = mlpRecentAddedMapPoints.erase(lit);
        else
        {
            lit++;
            borrar--;
        }
    }
}

/**
 * @brief 用当前关键帧与相邻关键帧通过三角化产生新的地图点，使得跟踪更稳
 */
void LocalMapping::CreateNewMapPoints()
{
    // Retrieve neighbor keyframes in covisibility graph
    // nn表示搜索最佳共视关键帧的数目
    // 不同传感器下要求不一样,单目的时候需要有更多的具有较好共视关系的关键帧来建立地图
    int nn = 10;
    // For stereo inertial case 这具原注释有问题吧 应该是For Monocular case
    // 0.4版本的这个参数为20
    if(mbMonocular)
        nn=10; //30
    // Step 1：在当前关键帧的共视关键帧中找到共视程度最高的nn帧相邻关键帧
    vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);
    //cout<<"step4.1: "<<mlNewKeyFrames.size()<<endl;
    // imu模式下在附近添加更多的帧进来
    // if (mbInertial)
    // {
    //     KeyFrame* pKF = mpCurrentKeyFrame;
    //     int count=0;
    //     // 在总数不够且上一关键帧存在，且添加的帧没有超过总数时
    //     while((vpNeighKFs.size()<=nn)&&(pKF->mPrevKF)&&(count++<nn))
    //     {
    //         vector<KeyFrame*>::iterator it = std::find(vpNeighKFs.begin(), vpNeighKFs.end(), pKF->mPrevKF);
    //         if(it==vpNeighKFs.end())
    //             vpNeighKFs.push_back(pKF->mPrevKF);
    //         pKF = pKF->mPrevKF;
    //     }
    // }

    float th = 0.6f;
    // 特征点匹配配置 最小距离 < 0.6*次小距离，比较苛刻了。不检查旋转
    ORBmatcher matcher(th,false);
    //SPmatcher matcher_sp(th);
    // 取出当前帧从世界坐标系到相机坐标系的变换矩阵
    Sophus::SE3<float> sophTcw1 = mpCurrentKeyFrame->GetPose();
    Eigen::Matrix<float,3,4> eigTcw1 = sophTcw1.matrix3x4();
    Eigen::Matrix<float,3,3> Rcw1 = eigTcw1.block<3,3>(0,0);
    Eigen::Matrix<float,3,3> Rwc1 = Rcw1.transpose();
    Eigen::Vector3f tcw1 = sophTcw1.translation();
    // 得到当前关键帧（左目）光心在世界坐标系中的坐标、内参
    Eigen::Vector3f Ow1 = mpCurrentKeyFrame->GetCameraCenter();

    const float &fx1 = mpCurrentKeyFrame->fx;
    const float &fy1 = mpCurrentKeyFrame->fy;
    const float &cx1 = mpCurrentKeyFrame->cx;
    const float &cy1 = mpCurrentKeyFrame->cy;
    const float &invfx1 = mpCurrentKeyFrame->invfx;
    const float &invfy1 = mpCurrentKeyFrame->invfy;
    //cout<<"step4.2: "<<mlNewKeyFrames.size()<<endl;
    // 用于后面的点深度的验证;这里的1.5是经验值
    const float ratioFactor = 1.5f*mpCurrentKeyFrame->mfScaleFactor;

    // 以下是统计点数用的，没啥作用
    int countStereo = 0;
    int countStereoGoodProj = 0;
    int countMonoGoodProj = 0;
    int countStereoAttempt = 0;
    int totalStereoPts = 0;
    // Search matches with epipolar restriction and triangulate

    // Step 2：遍历相邻关键帧vpNeighKFs
    for(size_t i=0; i<vpNeighKFs.size(); i++)
    {
        
        // 下面的过程会比较耗费时间,因此如果有新的关键帧需要处理的话,就先去处理新的关键帧吧
        if(i>0 && CheckNewKeyFrames())
            return;
        //cout<<"step4.3: "<<mlNewKeyFrames.size()<<endl;
        //cout<<vpNeighKFs.size()<<endl;
        KeyFrame* pKF2 = vpNeighKFs[i];

        GeometricCamera* pCamera1 = mpCurrentKeyFrame->mpCamera, *pCamera2 = pKF2->mpCamera;

        // Check first that baseline is not too short
        // 邻接的关键帧光心在世界坐标系中的坐标
        Eigen::Vector3f Ow2 = pKF2->GetCameraCenter();
        // 基线向量，两个关键帧间的相机位移
        Eigen::Vector3f vBaseline = Ow2-Ow1;
        // 基线长度
        const float baseline = vBaseline.norm();

        // Step 3：判断相机运动的基线是不是足够长
        if(!mbMonocular)
        {
            // 如果是双目相机，关键帧间距小于本身的基线时不生成3D点
            // 因为太短的基线下能够恢复的地图点不稳定
            if(baseline<pKF2->mb)
                continue;
        }
        else
        {
            // 单目相机情况
            // 邻接关键帧的场景深度中值
            const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
            // baseline与景深的比例
            const float ratioBaselineDepth = baseline/medianDepthKF2;
            // 如果特别远(比例特别小)，基线太短恢复3D点不准，那么跳过当前邻接的关键帧，不生成3D点
            if(ratioBaselineDepth<0.01){
                //cout<<"ssss"<<endl;
                continue;
            }

        }
        //cout<<"step4.4: "<<mlNewKeyFrames.size()<<endl;
        // Search matches that fullfil epipolar constraint
        // Step 4：通过BoW对两关键帧的未匹配的特征点快速匹配，用极线约束抑制离群点，生成新的匹配点对
        vector<pair<size_t,size_t> > vMatchedIndices;
        // 当惯性模式下，并且经过三次初始化，且为刚丢失状态
        bool bCoarse = mbInertial && mpTracker->mState==Tracking::RECENTLY_LOST && mpCurrentKeyFrame->GetMap()->GetIniertialBA2();

        // cv::Mat F12 = ComputeF12(mpCurrentKeyFrame, pKF2);

        // 通过极线约束的方式找到匹配点（且该点还没有成为MP，注意非单目已经生成的MP这里直接跳过不做匹配，所以最后并不会覆盖掉特征点对应的MP）
        //matcher.SearchForTriangulation(mpCurrentKeyFrame,pKF2,vMatchedIndices,false,bCoarse);
        auto start = std::chrono::high_resolution_clock::now();
        //cout<<"step4。4。0: "<<mlNewKeyFrames.size()<<endl;
        int size = mspmatcher.SearchForTriangulation(mpCurrentKeyFrame,pKF2,vMatchedIndices,false,bCoarse);
        
        //cout<<"step4。4。1: "<<mlNewKeyFrames.size()<<endl;
        if(mlNewKeyFrames.size() == 1){
           cout<<"mlNewKeyFrames.size(): "<<mlNewKeyFrames.front()->mnFrameId<<endl;
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        //cout<<"triangulation size: "<<size<<endl;
        //std::cout << "代码执行时间为: " << duration.count() << " 秒" << std::endl;
        // 取出与mpCurrentKeyFrame共视关键帧的内外参信息
        Sophus::SE3<float> sophTcw2 = pKF2->GetPose();
        Eigen::Matrix<float,3,4> eigTcw2 = sophTcw2.matrix3x4();
        Eigen::Matrix<float,3,3> Rcw2 = eigTcw2.block<3,3>(0,0);
        Eigen::Matrix<float,3,3> Rwc2 = Rcw2.transpose();
        Eigen::Vector3f tcw2 = sophTcw2.translation();

        const float &fx2 = pKF2->fx;
        const float &fy2 = pKF2->fy;
        const float &cx2 = pKF2->cx;
        const float &cy2 = pKF2->cy;
        const float &invfx2 = pKF2->invfx;
        const float &invfy2 = pKF2->invfy;

        // Triangulate each match
        // Step 5：对每对匹配通过三角化生成3D点,和 Triangulate函数差不多
        const int nmatches = vMatchedIndices.size();
        for(int ikp=0; ikp<nmatches; ikp++)
        {
            // 5.0
            // 当前匹配对在当前关键帧中的索引
            const int &idx1 = vMatchedIndices[ikp].first;
            // 当前匹配对在邻接关键帧中的索引
            const int &idx2 = vMatchedIndices[ikp].second;

            
            // 5.1
            // 当前匹配在当前关键帧中的特征点
            const cv::KeyPoint &kp1 = (mpCurrentKeyFrame -> NLeft == -1) ? mpCurrentKeyFrame->mvKeysUn[idx1]
                                                                         : (idx1 < mpCurrentKeyFrame -> NLeft) ? mpCurrentKeyFrame -> mvKeys[idx1]
                                                                                                               : mpCurrentKeyFrame -> mvKeysRight[idx1 - mpCurrentKeyFrame -> NLeft];
            // mvuRight中存放着极限校准后双目特征点在右目对应的像素横坐标，如果不是基线校准的双目或者没有找到匹配点，其值将为-1（或者rgbd）
            const float kp1_ur=mpCurrentKeyFrame->mvuRight[idx1];
            bool bStereo1 = (!mpCurrentKeyFrame->mpCamera2 && kp1_ur>=0);
            // 查看点idx1是否为右目的点
            const bool bRight1 = (mpCurrentKeyFrame -> NLeft == -1 || idx1 < mpCurrentKeyFrame -> NLeft) ? false
                                                                                                         : true;
            
            
            // 5.2
            // 当前匹配在邻接关键帧中的特征点
            const cv::KeyPoint &kp2 = (pKF2 -> NLeft == -1) ? pKF2->mvKeysUn[idx2]
                                                            : (idx2 < pKF2 -> NLeft) ? pKF2 -> mvKeys[idx2]
                                                                                     : pKF2 -> mvKeysRight[idx2 - pKF2 -> NLeft];
            // mvuRight中存放着双目的深度值，如果不是双目，其值将为-1
            // mvuRight中存放着极限校准后双目特征点在右目对应的像素横坐标，如果不是基线校准的双目或者没有找到匹配点，其值将为-1（或者rgbd）
            const float kp2_ur = pKF2->mvuRight[idx2];
            bool bStereo2 = (!pKF2->mpCamera2 && kp2_ur>=0);
            // 查看点idx2是否为右目的点
            const bool bRight2 = (pKF2 -> NLeft == -1 || idx2 < pKF2 -> NLeft) ? false
                                                                               : true;

            // 5.3 当目前为左右目时，确定两个点所在相机之间的位姿关系
            if(mpCurrentKeyFrame->mpCamera2 && pKF2->mpCamera2){
                if(bRight1 && bRight2){
                    sophTcw1 = mpCurrentKeyFrame->GetRightPose();
                    Ow1 = mpCurrentKeyFrame->GetRightCameraCenter();

                    sophTcw2 = pKF2->GetRightPose();
                    Ow2 = pKF2->GetRightCameraCenter();

                    pCamera1 = mpCurrentKeyFrame->mpCamera2;
                    pCamera2 = pKF2->mpCamera2;
                }
                else if(bRight1 && !bRight2){
                    sophTcw1 = mpCurrentKeyFrame->GetRightPose();
                    Ow1 = mpCurrentKeyFrame->GetRightCameraCenter();

                    sophTcw2 = pKF2->GetPose();
                    Ow2 = pKF2->GetCameraCenter();

                    pCamera1 = mpCurrentKeyFrame->mpCamera2;
                    pCamera2 = pKF2->mpCamera;
                }
                else if(!bRight1 && bRight2){
                    sophTcw1 = mpCurrentKeyFrame->GetPose();
                    Ow1 = mpCurrentKeyFrame->GetCameraCenter();

                    sophTcw2 = pKF2->GetRightPose();
                    Ow2 = pKF2->GetRightCameraCenter();

                    pCamera1 = mpCurrentKeyFrame->mpCamera;
                    pCamera2 = pKF2->mpCamera2;
                }
                else{
                    sophTcw1 = mpCurrentKeyFrame->GetPose();
                    Ow1 = mpCurrentKeyFrame->GetCameraCenter();

                    sophTcw2 = pKF2->GetPose();
                    Ow2 = pKF2->GetCameraCenter();

                    pCamera1 = mpCurrentKeyFrame->mpCamera;
                    pCamera2 = pKF2->mpCamera;
                }
                eigTcw1 = sophTcw1.matrix3x4();
                Rcw1 = eigTcw1.block<3,3>(0,0);
                Rwc1 = Rcw1.transpose();
                tcw1 = sophTcw1.translation();

                eigTcw2 = sophTcw2.matrix3x4();
                Rcw2 = eigTcw2.block<3,3>(0,0);
                Rwc2 = Rcw2.transpose();
                tcw2 = sophTcw2.translation();
            }

            // Check parallax between rays
            // Step 5.4：利用匹配点反投影得到视差角
            // 特征点反投影,其实得到的是在各自相机坐标系下的一个非归一化的方向向量,和这个点的反投影射线重合
            Eigen::Vector3f xn1 = pCamera1->unprojectEig(kp1.pt);
            Eigen::Vector3f xn2 = pCamera2->unprojectEig(kp2.pt);
            // 由相机坐标系转到世界坐标系(得到的是那条反投影射线的一个同向向量在世界坐标系下的表示,还是只能够表示方向)，得到视差角余弦值
            Eigen::Vector3f ray1 = Rwc1 * xn1;
            Eigen::Vector3f ray2 = Rwc2 * xn2;
            // 这个就是求向量之间角度公式
            const float cosParallaxRays = ray1.dot(ray2)/(ray1.norm() * ray2.norm());

            // 加1是为了让cosParallaxStereo随便初始化为一个很大的值
            float cosParallaxStereo = cosParallaxRays+1;
            float cosParallaxStereo1 = cosParallaxStereo;
            float cosParallaxStereo2 = cosParallaxStereo;

            // Step 5.5：对于双目，利用双目得到视差角；单目相机没有特殊操作
            if(bStereo1)
                // 传感器是双目相机,并且当前的关键帧的这个点有对应的深度
                // 假设是平行的双目相机，计算出两个相机观察这个点的时候的视差角;
                // ? 感觉直接使用向量夹角的方式计算会准确一些啊（双目的时候），那么为什么不直接使用那个呢？
                // 回答：因为双目深度值、基线是更可靠的，比特征匹配再三角化出来的稳
                cosParallaxStereo1 = cos(2*atan2(mpCurrentKeyFrame->mb/2,mpCurrentKeyFrame->mvDepth[idx1]));
            else if(bStereo2)
                //传感器是双目相机,并且邻接的关键帧的这个点有对应的深度，和上面一样操作
                cosParallaxStereo2 = cos(2*atan2(pKF2->mb/2,pKF2->mvDepth[idx2]));

            // 统计用的
            if (bStereo1 || bStereo2) totalStereoPts++;
            
            // 得到双目观测的视差角
            cosParallaxStereo = min(cosParallaxStereo1,cosParallaxStereo2);

            // Step 5.6：三角化恢复3D点
            Eigen::Vector3f x3D;

            bool goodProj = false;
            bool bPointStereo = false;
            // cosParallaxRays>0 && (bStereo1 || bStereo2 || cosParallaxRays<0.9998)表明视线角正常
            // cosParallaxRays<cosParallaxStereo表明前后帧视线角比双目视线角大，所以用前后帧三角化而来，反之使用双目的，如果没有双目则跳过
            // 视差角度小时用三角法恢复3D点，视差角大时（离相机近）用双目恢复3D点（双目以及深度有效）
            if(cosParallaxRays<cosParallaxStereo && cosParallaxRays>0 && (bStereo1 || bStereo2 ||
                                                                          (cosParallaxRays<0.9996 && mbInertial) || (cosParallaxRays<0.9998 && !mbInertial)))
            {
                // 三角化，包装成了函数
                goodProj = GeometricTools::Triangulate(xn1, xn2, eigTcw1, eigTcw2, x3D);
                if(!goodProj)
                    continue;
            }
            else if(bStereo1 && cosParallaxStereo1<cosParallaxStereo2)
            {
                countStereoAttempt++;
                bPointStereo = true;
                // 如果是双目，用视差角更大的那个双目信息来恢复，直接用已知3D点反投影了
                goodProj = mpCurrentKeyFrame->UnprojectStereo(idx1, x3D);
            }
            else if(bStereo2 && cosParallaxStereo2<cosParallaxStereo1)
            {
                countStereoAttempt++;
                bPointStereo = true;
                // 如果是双目，用视差角更大的那个双目信息来恢复，直接用已知3D点反投影了
                goodProj = pKF2->UnprojectStereo(idx2, x3D);
            }
            else
            {
                continue; //No stereo and very low parallax
            }

            // 成功三角化
            if(goodProj && bPointStereo)
                countStereoGoodProj++;
           
            if(!goodProj)
                continue;

            //Check triangulation in front of cameras
            // Step 5.7：检测生成的3D点是否在相机前方,不在的话就放弃这个点
            float z1 = Rcw1.row(2).dot(x3D) + tcw1(2);
            if(z1<=0)
                continue;

            float z2 = Rcw2.row(2).dot(x3D) + tcw2(2);
            if(z2<=0)
                continue;

            //Check reprojection error in first keyframe
            // Step 5.7：计算3D点在当前关键帧下的重投影误差
            const float &sigmaSquare1 = mpCurrentKeyFrame->mvLevelSigma2[kp1.octave];
            const float x1 = Rcw1.row(0).dot(x3D)+tcw1(0);
            const float y1 = Rcw1.row(1).dot(x3D)+tcw1(1);
            const float invz1 = 1.0/z1;

            if(!bStereo1)
            {
                // 单目情况下
                cv::Point2f uv1 = pCamera1->project(cv::Point3f(x1,y1,z1));
                float errX1 = uv1.x - kp1.pt.x;
                float errY1 = uv1.y - kp1.pt.y;

                // 假设测量有一个像素的偏差，2自由度卡方检验阈值是5.991
                if((errX1*errX1+errY1*errY1)>5.991*sigmaSquare1)
                    continue;

            }
            else
            {
                // 双目情况
                float u1 = fx1*x1*invz1+cx1;
                // 根据视差公式计算假想的右目坐标
                float u1_r = u1 - mpCurrentKeyFrame->mbf*invz1;
                float v1 = fy1*y1*invz1+cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                float errX1_r = u1_r - kp1_ur;
                // 自由度为3，卡方检验阈值是7.8
                if((errX1*errX1+errY1*errY1+errX1_r*errX1_r)>7.8*sigmaSquare1)
                    continue;
            }

            //Check reprojection error in second keyframe
            // 计算3D点在另一个关键帧下的重投影误差，操作同上
            const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave];
            const float x2 = Rcw2.row(0).dot(x3D)+tcw2(0);
            const float y2 = Rcw2.row(1).dot(x3D)+tcw2(1);
            const float invz2 = 1.0/z2;
            if(!bStereo2)
            {
                cv::Point2f uv2 = pCamera2->project(cv::Point3f(x2,y2,z2));
                float errX2 = uv2.x - kp2.pt.x;
                float errY2 = uv2.y - kp2.pt.y;
                if((errX2*errX2+errY2*errY2)>5.991*sigmaSquare2)
                    continue;
            }
            else
            {
                float u2 = fx2*x2*invz2+cx2;
                float u2_r = u2 - mpCurrentKeyFrame->mbf*invz2;
                float v2 = fy2*y2*invz2+cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                float errX2_r = u2_r - kp2_ur;
                if((errX2*errX2+errY2*errY2+errX2_r*errX2_r)>7.8*sigmaSquare2)
                    continue;
            }

            //Check scale consistency
            // Step 5.8：检查尺度连续性

            // 世界坐标系下，3D点与相机间的向量，方向由相机指向3D点
            Eigen::Vector3f normal1 = x3D - Ow1;
            float dist1 = normal1.norm();

            Eigen::Vector3f normal2 = x3D - Ow2;
            float dist2 = normal2.norm();

            if(dist1==0 || dist2==0)
                continue;

            if(mbFarPoints && (dist1>=mThFarPoints||dist2>=mThFarPoints)) // MODIFICATION
                continue;
            // ratioDist是不考虑金字塔尺度下的距离比例
            const float ratioDist = dist2/dist1;
            // 金字塔尺度因子的比例
            const float ratioOctave = mpCurrentKeyFrame->mvScaleFactors[kp1.octave]/pKF2->mvScaleFactors[kp2.octave];

            // 距离的比例和图像金字塔的比例不应该差太多，否则就跳过
            if(ratioDist*ratioFactor<ratioOctave || ratioDist>ratioOctave*ratioFactor)
                continue;

            // Triangulation is succesfull
            // Step 6：三角化生成3D点成功，构造成MapPoint
            MapPoint* pMP = new MapPoint(x3D, mpCurrentKeyFrame, mpAtlas->GetCurrentMap());

            //使用gps注释*****
            //cv::Point3d p3d= cv::Point3d(static_cast<double>(x3D.x()), static_cast<double>(x3D.y()), static_cast<double>(x3D.z()));
            // if (!ransac.is_finished())
            // {
            //     if(mpTracker->GetGlobalFrameAlignmentState() == Tracking::ALIGNED)
            //         ransac.begin = true;
            //     ransac.solve(p3d);
            // }
            //**
            
            if (bPointStereo)
                countStereo++;
            countMonoGoodProj++;
            // Step 6.1：为该MapPoint添加属性：
            // a.观测到该MapPoint的关键帧
            pMP->AddObservation(mpCurrentKeyFrame,idx1);
            pMP->AddObservation(pKF2,idx2);

            mpCurrentKeyFrame->AddMapPoint(pMP,idx1);
            pKF2->AddMapPoint(pMP,idx2);

            // b.该MapPoint的描述子
            pMP->ComputeDistinctiveDescriptors();

            // c.该MapPoint的平均观测方向和深度范围
            pMP->UpdateNormalAndDepth();

            mpAtlas->AddMapPoint(pMP);
            // Step 7：将新产生的点放入检测队列
            // 这些MapPoints都会经过MapPointCulling函数的检验
            mlpRecentAddedMapPoints.push_back(pMP);
        }
         //cout<<"step4.5: "<<mlNewKeyFrames.size()<<endl;
    }
    //cout<<"sanjiaohua: "<< countMonoGoodProj<<endl;
}

// Eigen::Matrix<double, 6, 6> LocalMapping::Initinformation(const vector<Eigen::Vector3d>& pointSet,double fx, double fy){
//     vector<Eigen::Vector3d> selectedPoints;
//     Eigen::MatrixXd informationMatrix = Eigen::MatrixXd::Zero(6, 6);
//     for(int i = 0 ; i < 6 ; ++i){
//         double maxValue = -numeric_limits<double>::infinity();
//         int maxIndex = -1;
//         for(int j = 0 ; j < pointSet.size(); ++j){
//             double sum = 0.0;
//             Sophus::SE3d pose;
//             Eigen::Matrix<double, 2, 6> J = computeJacobian(pointSet[j],fx,fy, pose);
//             for(int row = 0; row < J.rows(); ++row)
//             {
//                 sum += abs(J(row, i));
//             }
//             if(sum >= maxValue && find(selectedPoints.begin(), selectedPoints.end(), pointSet[j]) == selectedPoints.end())
//             {
//                 maxValue = sum;
//                 maxIndex = j;
//             }
//         }
//         Sophus::SE3d pose;
//         Eigen::VectorXd selectedPoint = pointSet[maxIndex];
//         selectedPoints.push_back(selectedPoint);
//         Eigen::Matrix<double, 2, 6> J = computeJacobian(selectedPoint, fx, fy, pose);
//         double sigma_squared = 1;
//         Eigen::MatrixXd informationOfSelectedPoint = J.transpose()*sigma_squared*J;
//         informationMatrix += informationOfSelectedPoint;
        
//     }
//     return informationMatrix;
// }

// Eigen::MatrixXd computeJacobian(const Eigen::Vector3d& p, double fx, double fy, Sophus::SE3d &pose){
//     double X = p(0);
//     double Y = p(1);
//     double Z = p(2);

//     Eigen::Vector3d pc = pose*p;
//     double inv_z = 1.0 / pc[2];
//     double inv_z2 = inv_z * inv_z;
//     Eigen::Matrix<double, 2, 6> J;
//     J << -fx * inv_z, 0, fx*pc[0]*inv_z2, fx*pc[0]*pc[1]*inv_z2,-fx-fx*pc[0]*pc[0]*inv_z2, fx*pc[1]*inv_z,0,
//     -fy*inv_z, fy*pc[1]*inv_z, fy+fy*pc[1]*inv_z2, -fy*pc[0]*pc[1]*inv_z2, -fy*pc[0]*inv_z;
//     return J;
// }

// double ComputeMuInfo()
// {

// }
// MapPoint* SelectMostInformativePoint(vector<MapPoint>& mapPoints,
//                                     const vector<MapPoint>& selectedPoints,
//                                     const Eigen::Vector3d& border) {

//     double maxMuInfo = -std::numeric_limits<double>::infinity();
//     MapPoint* bestPoint;
//     for (auto& mapPoint : mapPoints) {
//         Eigen::Vector3f Pos = mapPoint.GetWorldPos();//需要添加将该地图点投影到当前帧上的代码
//         double muInfo = ComputeMuInfo();
//         if(muInfo > maxMuInfo){
//             maxMuInfo = muInfo;
//             bestPoint = &mapPoint;
//         }
//     }
//     // 返回地图点应该也需要改进，全部变为地址
//     return bestPoint;
// }

// vector<Eigen::Vector3d> LocalMapping::InfoPointSelection(const vector<MapPoint>& mapPoints, int M, double fx, double fy)
// {
//     vector<MapPoint> selectedPoints;
//     Eigen::MatrixXd initinformatrix =  Initinformation(mapPoints, fx,fy);
//     Eigen::Vector3d border(100, 100, 0);
//     Eigen::Vector3d bestPoint;

//     while(selectedPoints.size() < M){
//         MapPoint mostInformativePoint = SelectMostInformativePoint(mapPoints, selectedPoints, border);
//     }
//     double maxFunctionValue = -std::numeric_limits<double>::infinity();
//     for(const auto& mapPoint : mapPoints){
//         double functionValue = ComputeFunctionValue();
//         if(functionValue > maxFunctionValue){
//             maxFunctionValue = functionValue;
//             bestPoint = mapPoint;
//         }
//     }
//     return bestPoint;
// } 

/**
 * @brief 检查并融合当前关键帧与相邻帧（两级相邻）重复的MapPoints
 */
void LocalMapping::SearchInNeighbors()
{
    // Retrieve neighbor keyframes
    // Step 1：获得当前关键帧在共视图中权重排名前nn的邻接关键帧
    // 开始之前先定义几个概念
    // 当前关键帧的邻接关键帧，称为一级相邻关键帧，也就是邻居
    // 与一级相邻关键帧相邻的关键帧，称为二级相邻关键帧，也就是邻居的邻居

    // 单目情况要30个邻接关键帧,0.4版本是20个，又加了许多
    int nn = 10;
    if(mbMonocular)
        nn=30;
    
    // 和当前关键帧相邻的关键帧，也就是一级相邻关键帧
    const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);
    // Step 2：存储一级相邻关键帧及其二级相邻关键帧
    vector<KeyFrame*> vpTargetKFs;
    // 开始对所有候选的一级关键帧展开遍历：
    for(vector<KeyFrame*>::const_iterator vit=vpNeighKFs.begin(), vend=vpNeighKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;
        // 没有和当前帧进行过融合的操作
        if(pKFi->isBad() || pKFi->mnFuseTargetForKF == mpCurrentKeyFrame->mnId)
            continue;
        // 加入一级相邻关键帧
        vpTargetKFs.push_back(pKFi);
        // 标记已经加入
        pKFi->mnFuseTargetForKF = mpCurrentKeyFrame->mnId;
    }

    // Add some covisible of covisible
    // Extend to some second neighbors if abort is not requested
    // 以一级相邻关键帧的共视关系最好的5个相邻关键帧 作为二级相邻关键帧
    for(int i=0, imax=vpTargetKFs.size(); i<imax; i++)
    {
        const vector<KeyFrame*> vpSecondNeighKFs = vpTargetKFs[i]->GetBestCovisibilityKeyFrames(20);
        // 遍历得到的二级相邻关键帧
        for(vector<KeyFrame*>::const_iterator vit2=vpSecondNeighKFs.begin(), vend2=vpSecondNeighKFs.end(); vit2!=vend2; vit2++)
        {
            KeyFrame* pKFi2 = *vit2;
            if(pKFi2->isBad() || pKFi2->mnFuseTargetForKF==mpCurrentKeyFrame->mnId || pKFi2->mnId==mpCurrentKeyFrame->mnId)
                continue;
            // 存入二级相邻关键帧
            vpTargetKFs.push_back(pKFi2);
            pKFi2->mnFuseTargetForKF=mpCurrentKeyFrame->mnId;
        }
        if (mbAbortBA)
            break;
    }

    // Extend to temporal neighbors
    // imu模式下加了一些prevKF
    if(mbInertial)
    {
        KeyFrame* pKFi = mpCurrentKeyFrame->mPrevKF;
        while(vpTargetKFs.size()<20 && pKFi)
        {
            if(pKFi->isBad() || pKFi->mnFuseTargetForKF==mpCurrentKeyFrame->mnId)
            {
                pKFi = pKFi->mPrevKF;
                continue;
            }
            vpTargetKFs.push_back(pKFi);
            pKFi->mnFuseTargetForKF=mpCurrentKeyFrame->mnId;
            pKFi = pKFi->mPrevKF;
        }
    }

    // Search matches by projection from current KF in target KFs
    // 使用默认参数, 最优和次优比例0.6,匹配时检查特征点的旋转
    //ORBmatcher matcher;
    //SPmatcher matcher_sp(0.6);
    // Step 3：将当前帧的地图点分别与一级二级相邻关键帧地图点进行融合 -- 正向
    vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for(vector<KeyFrame*>::iterator vit=vpTargetKFs.begin(), vend=vpTargetKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;

        // 将地图点投影到关键帧中进行匹配和融合；融合策略如下
        // 1.如果地图点能匹配关键帧的特征点，并且该点有对应的地图点，那么选择观测数目多的替换两个地图点
        // 2.如果地图点能匹配关键帧的特征点，并且该点没有对应的地图点，那么为该点添加该投影地图点
        // 注意这个时候对地图点融合的操作是立即生效的
        //matcher.Fuse(pKFi,vpMapPointMatches);
        mspmatcher.Fuse(pKFi, vpMapPointMatches);
        if(pKFi->NLeft != -1) mspmatcher.Fuse(pKFi,vpMapPointMatches,true);
    }


    if (mbAbortBA)
        return;

    // Search matches by projection from target KFs in current KF
    // Step 4：将一级二级相邻关键帧地图点分别与当前关键帧地图点进行融合 -- 反向
    // 用于进行存储要融合的一级邻接和二级邻接关键帧所有MapPoints的集合
    vector<MapPoint*> vpFuseCandidates;
    vpFuseCandidates.reserve(vpTargetKFs.size()*vpMapPointMatches.size());

    //  Step 4.1：遍历每一个一级邻接和二级邻接关键帧，收集他们的地图点存储到 vpFuseCandidates
    for(vector<KeyFrame*>::iterator vitKF=vpTargetKFs.begin(), vendKF=vpTargetKFs.end(); vitKF!=vendKF; vitKF++)
    {
        KeyFrame* pKFi = *vitKF;

        vector<MapPoint*> vpMapPointsKFi = pKFi->GetMapPointMatches();

        // 遍历当前一级邻接和二级邻接关键帧中所有的MapPoints,找出需要进行融合的并且加入到集合中
        for(vector<MapPoint*>::iterator vitMP=vpMapPointsKFi.begin(), vendMP=vpMapPointsKFi.end(); vitMP!=vendMP; vitMP++)
        {
            MapPoint* pMP = *vitMP;
            if(!pMP)
                continue;
            
            // 如果地图点是坏点，或者已经加进集合vpFuseCandidates，跳过
            if(pMP->isBad() || pMP->mnFuseCandidateForKF == mpCurrentKeyFrame->mnId)
                continue;
            
            // 加入集合，并标记已经加入
            pMP->mnFuseCandidateForKF = mpCurrentKeyFrame->mnId;
            vpFuseCandidates.push_back(pMP);
        }
    }

    // Step 4.2：进行地图点投影融合,和正向融合操作是完全相同的
    // 不同的是正向操作是"每个关键帧和当前关键帧的地图点进行融合",而这里的是"当前关键帧和所有邻接关键帧的地图点进行融合"
    //matcher.Fuse(mpCurrentKeyFrame,vpFuseCandidates);
    mspmatcher.Fuse(mpCurrentKeyFrame, vpFuseCandidates);
    if(mpCurrentKeyFrame->NLeft != -1) mspmatcher.Fuse(mpCurrentKeyFrame,vpFuseCandidates,true);


    // Update points
    // Step 5：更新当前帧地图点的描述子、深度、观测主方向等属性
    vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for(size_t i=0, iend=vpMapPointMatches.size(); i<iend; i++)
    {
        MapPoint* pMP=vpMapPointMatches[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                // 在所有找到pMP的关键帧中，获得最佳的描述子
                pMP->ComputeDistinctiveDescriptors();

                // 更新平均观测方向和观测距离
                pMP->UpdateNormalAndDepth();
            }
        }
    }

    // Update connections in covisibility graph
    // Step 6：更新当前帧的MapPoints后更新与其它帧的连接关系
    // 更新covisibility图
    mpCurrentKeyFrame->UpdateConnections();
}

/**
 * @brief 外部线程调用,请求停止当前线程的工作; 其实是回环检测线程调用,来避免在进行全局优化的过程中局部建图线程添加新的关键帧
 */
void LocalMapping::RequestStop()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopRequested = true;
    unique_lock<mutex> lock2(mMutexNewKFs);
    mbAbortBA = true;
}

/**
 * @brief 检查是否要把当前的局部建图线程停止工作,运行的时候要检查是否有终止请求,如果有就执行. 由run函数调用
 */
bool LocalMapping::Stop()
{
    unique_lock<mutex> lock(mMutexStop);
    // 如果当前线程还没有准备停止,但是已经有终止请求了,那么就准备停止当前线程
    if(mbStopRequested && !mbNotStop)
    {
        mbStopped = true;
        cout << "Local Mapping STOP" << endl;
        return true;
    }

    return false;
}

/**
 * @brief 检查mbStopped是否为true，为true表示可以并终止localmapping 线程
 */
bool LocalMapping::isStopped()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopped;
}

/**
 * @brief 求外部线程调用，为true，表示外部线程请求停止 local mapping
 */
bool LocalMapping::stopRequested()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopRequested;
}

/**
 * @brief 释放当前还在缓冲区中的关键帧指针
 */
void LocalMapping::Release()
{
    unique_lock<mutex> lock(mMutexStop);
    unique_lock<mutex> lock2(mMutexFinish);
    if(mbFinished)
        return;
    mbStopped = false;
    mbStopRequested = false;
    for(list<KeyFrame*>::iterator lit = mlNewKeyFrames.begin(), lend=mlNewKeyFrames.end(); lit!=lend; lit++)
        delete *lit;
    mlNewKeyFrames.clear();

    cout << "Local Mapping RELEASE" << endl;
}

/**
 * @brief 查看是否接收关键帧，也就是当前线程是否在处理数据，当然tracking线程也不会全看这个值，他会根据队列阻塞情况
 */
bool LocalMapping::AcceptKeyFrames()
{
    unique_lock<mutex> lock(mMutexAccept);
    return mbAcceptKeyFrames;
}

/**
 * @brief 设置"允许接受关键帧"的状态标志
 */
void LocalMapping::SetAcceptKeyFrames(bool flag)
{
    unique_lock<mutex> lock(mMutexAccept);
    mbAcceptKeyFrames=flag;
}

/**
 * @brief 如果不让它暂停，即使发出了暂停信号也不暂停
 */
bool LocalMapping::SetNotStop(bool flag)
{
    unique_lock<mutex> lock(mMutexStop);

    if(flag && mbStopped)
        return false;

    mbNotStop = flag;

    return true;
}

/**
 * @brief 放弃这次操作，虽然叫BA但并不是只断优化
 */
void LocalMapping::InterruptBA()
{
    mbAbortBA = true;
}

/**
 * @brief 检测当前关键帧在共视图中的关键帧，根据地图点在共视图中的冗余程度剔除该共视关键帧
 * 冗余关键帧的判定：90%以上的地图点能被其他关键帧（至少3个）观测到
 */
void LocalMapping::KeyFrameCulling()
{
    // Check redundant keyframes (only local keyframes)
    // A keyframe is considered redundant if the 90% of the MapPoints it sees, are seen
    // in at least other 3 keyframes (in the same or finer scale)
    // We only consider close stereo points

    // 该函数里变量层层深入，这里列一下：
    // mpCurrentKeyFrame：当前关键帧，本程序就是判断它是否需要删除
    // pKF： mpCurrentKeyFrame的某一个共视关键帧
    // vpMapPoints：pKF对应的所有地图点
    // pMP：vpMapPoints中的某个地图点
    // observations：所有能观测到pMP的关键帧
    // pKFi：observations中的某个关键帧
    // scaleLeveli：pKFi的金字塔尺度
    // scaleLevel：pKF的金字塔尺度
    const int Nd = 21;
    // 更新共视关系
    mpCurrentKeyFrame->UpdateBestCovisibles();
    // 1. 根据Covisibility Graph提取当前帧的共视关键帧
    vector<KeyFrame*> vpLocalKeyFrames = mpCurrentKeyFrame->GetVectorCovisibleKeyFrames();

    float redundant_th;
    // 非IMU模式时
    if(!mbInertial)
        redundant_th = 0.95;
    // 单目+imu 时
    else if (mbMonocular)
        redundant_th = 0.95;
    // 双目+imu时
    else
        redundant_th = 0.5;

    const bool bInitImu = mpAtlas->isImuInitialized();
    int count=0;

    // Compoute last KF from optimizable window:
    unsigned int last_ID;
    if (mbInertial)
    {
        int count = 0;
        KeyFrame* aux_KF = mpCurrentKeyFrame;
        // 找到第前21个关键帧的关键帧id
        while(count<Nd && aux_KF->mPrevKF)
        {
            aux_KF = aux_KF->mPrevKF;
            count++;
        }
        last_ID = aux_KF->mnId;
    }


    // 对所有的共视关键帧进行遍历
    for(vector<KeyFrame*>::iterator vit=vpLocalKeyFrames.begin(), vend=vpLocalKeyFrames.end(); vit!=vend; vit++)
    {
        count++;
        KeyFrame* pKF = *vit;

        // 如果是地图里第1个关键帧或者是标记为坏帧，则跳过
        if((pKF->mnId==pKF->GetMap()->GetInitKFid()) || pKF->isBad())
            continue;
        // Step 2：提取每个共视关键帧的地图点
        const vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();

        // 记录某个点被观测次数，后面并未使用
        int nObs = 3;
        const int thObs=nObs;  // 观测次数阈值，默认为3
        // 记录冗余观测点的数目
        int nRedundantObservations=0;
        int nMPs=0;

        // Step 3：遍历该共视关键帧的所有地图点，判断是否90%以上的地图点能被其它至少3个关键帧（同样或者更低层级）观测到
        for(size_t i=0, iend=vpMapPoints.size(); i<iend; i++)
        {
            MapPoint* pMP = vpMapPoints[i];
            if(pMP)
            {
                if(!pMP->isBad())
                {
                    if(!mbMonocular)
                    {
                        // 对于双目，仅考虑近处（不超过基线的40倍 ）的地图点
                        if(pKF->mvDepth[i]>pKF->mThDepth || pKF->mvDepth[i]<0)
                            continue;
                    }

                    nMPs++;
                    // pMP->Observations() 是观测到该地图点的相机总数目（单目1，双目2）
                    if(pMP->Observations()>thObs)
                    {
                        const int &scaleLevel = (pKF -> NLeft == -1) ? pKF->mvKeysUn[i].octave
                                                                     : (i < pKF -> NLeft) ? pKF -> mvKeys[i].octave
                                                                                          : pKF -> mvKeysRight[i].octave;
                        const map<KeyFrame*, tuple<int,int>> observations = pMP->GetObservations();
                        int nObs=0;
                        // 遍历观测到该地图点的关键帧
                        for(map<KeyFrame*, tuple<int,int>>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
                        {
                            KeyFrame* pKFi = mit->first;
                            if(pKFi==pKF)
                                continue;
                            tuple<int,int> indexes = mit->second;
                            int leftIndex = get<0>(indexes), rightIndex = get<1>(indexes);
                            int scaleLeveli = -1;
                            if(pKFi -> NLeft == -1)
                                scaleLeveli = pKFi->mvKeysUn[leftIndex].octave;
                            else {
                                if (leftIndex != -1) {
                                    scaleLeveli = pKFi->mvKeys[leftIndex].octave;
                                }
                                if (rightIndex != -1) {
                                    int rightLevel = pKFi->mvKeysRight[rightIndex - pKFi->NLeft].octave;
                                    scaleLeveli = (scaleLeveli == -1 || scaleLeveli > rightLevel) ? rightLevel
                                                                                                  : scaleLeveli;
                                }
                            }
                            // 尺度约束：为什么pKF 尺度+1 要大于等于 pKFi 尺度？
							// 回答：因为同样或更低金字塔层级的地图点更准确
                            if(scaleLeveli<=scaleLevel+1)
                            {
                                nObs++;
                                // 已经找到3个满足条件的关键帧，就停止不找了
                                if(nObs>thObs)
                                    break;
                            }
                        }
                        // 地图点至少被3个关键帧观测到，就记录为冗余点，更新冗余点计数数目
                        if(nObs>thObs)
                        {
                            nRedundantObservations++;
                        }
                    }
                }
            }
        }

        // Step 4：该关键帧90%以上的有效地图点被判断为冗余的，则删除该关键帧
        if(nRedundantObservations>redundant_th*nMPs)
        {
            // imu模式下需要更改前后关键帧的连续性，且预积分要叠加起来
            if (mbInertial)
            {
                // 关键帧少于Nd个，跳过不删
                if (mpAtlas->KeyFramesInMap()<=Nd)
                    continue;

                // 关键帧与当前关键帧id差一个，跳过不删
                if(pKF->mnId>(mpCurrentKeyFrame->mnId-2))
                    continue;

                // 关键帧具有前后关键帧
                if(pKF->mPrevKF && pKF->mNextKF)
                {
                    const float t = pKF->mNextKF->mTimeStamp-pKF->mPrevKF->mTimeStamp;
                    // 下面两个括号里的内容一模一样
                    // imu初始化了，且距当前帧的ID超过21，且前后两个关键帧时间间隔小于3s
                    // 或者时间间隔小于0.5s
                    if((bInitImu && (pKF->mnId<last_ID) && t<3.) || (t<0.5))
                    {
                        pKF->mNextKF->mpImuPreintegrated->MergePrevious(pKF->mpImuPreintegrated);
                        pKF->mNextKF->mPrevKF = pKF->mPrevKF;
                        pKF->mPrevKF->mNextKF = pKF->mNextKF;
                        pKF->mNextKF = NULL;
                        pKF->mPrevKF = NULL;
                        pKF->SetBadFlag();
                    }
                    // 没经过imu初始化的第三阶段，且关键帧与其前一个关键帧的距离小于0.02m，且前后两个关键帧时间间隔小于3s
                    else if(!mpCurrentKeyFrame->GetMap()->GetIniertialBA2() && ((pKF->GetImuPosition()-pKF->mPrevKF->GetImuPosition()).norm()<0.02) && (t<3))
                    {
                        pKF->mNextKF->mpImuPreintegrated->MergePrevious(pKF->mpImuPreintegrated);
                        pKF->mNextKF->mPrevKF = pKF->mPrevKF;
                        pKF->mPrevKF->mNextKF = pKF->mNextKF;
                        pKF->mNextKF = NULL;
                        pKF->mPrevKF = NULL;
                        pKF->SetBadFlag();
                    }
                }
            }
            // 非imu就没那么多事儿了，直接干掉
            else
            {
                pKF->SetBadFlag();
            }
        }
        // 遍历共视关键帧个数超过一定，就不弄了
        if((count > 20 && mbAbortBA) || count>100)
        {
            break;
        }
    }
}

/**
 * @brief 请求reset
 */
void LocalMapping::RequestReset()
{
    {
        unique_lock<mutex> lock(mMutexReset);
        cout << "LM: Map reset recieved" << endl;
        mbResetRequested = true;
    }
    cout << "LM: Map reset, waiting..." << endl;

    // 一直等到局部建图线程响应之后才可以退出
    while(1)
    {
        {
            unique_lock<mutex> lock2(mMutexReset);
            if(!mbResetRequested)
                break;
        }
        usleep(3000);
    }
    cout << "LM: Map reset, Done!!!" << endl;
}

/**
 * @brief 接收重置当前地图的信号
 */
void LocalMapping::RequestResetActiveMap(Map* pMap)
{
    {
        unique_lock<mutex> lock(mMutexReset);
        cout << "LM: Active map reset recieved" << endl;
        mbResetRequestedActiveMap = true;
        mpMapToReset = pMap;
    }
    cout << "LM: Active map reset, waiting..." << endl;

    while(1)
    {
        {
            unique_lock<mutex> lock2(mMutexReset);
            if(!mbResetRequestedActiveMap)
                break;
        }
        usleep(3000);
    }
    cout << "LM: Active map reset, Done!!!" << endl;
}

/**
 * @brief 检查是否有复位线程的请求
 */
void LocalMapping::ResetIfRequested()
{
    bool executed_reset = false;
    {
        unique_lock<mutex> lock(mMutexReset);
        // 执行复位操作:清空关键帧缓冲区,清空待cull的地图点缓冲
        if(mbResetRequested)
        {
            executed_reset = true;

            cout << "LM: Reseting Atlas in Local Mapping..." << endl;
            mlNewKeyFrames.clear();
            mlpRecentAddedMapPoints.clear();
            // 恢复为false表示复位过程完成
            mbResetRequested = false;
            mbResetRequestedActiveMap = false;

            // Inertial parameters
            mTinit = 0.f;
            mbNotBA2 = true;
            mbNotBA1 = true;
            mbBadImu=false;

            mIdxInit=0;

            cout << "LM: End reseting Local Mapping..." << endl;
        }

        if(mbResetRequestedActiveMap) {
            executed_reset = true;
            cout << "LM: Reseting current map in Local Mapping..." << endl;
            mlNewKeyFrames.clear();
            mlpRecentAddedMapPoints.clear();

            // Inertial parameters
            mTinit = 0.f;
            mbNotBA2 = true;
            mbNotBA1 = true;
            mbBadImu=false;

            mbResetRequested = false;
            mbResetRequestedActiveMap = false;
            cout << "LM: End reseting Local Mapping..." << endl;
        }
    }
    if(executed_reset)
        cout << "LM: Reset free the mutex" << endl;

}

/**
 * @brief 请求终止当前线程
 */
void LocalMapping::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

/**
 * @brief 查看完成信号，跳出while循环
 */
bool LocalMapping::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

/**
 * @brief 设置当前线程已经真正地结束了
 */
void LocalMapping::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;    
    unique_lock<mutex> lock2(mMutexStop);
    mbStopped = true;
}

/**
 * @brief 当前线程的run函数是否已经终止
 */
bool LocalMapping::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

// cv::Mat LocalMapping::SkewSymmetricMatrix(const cv::Mat &v)
// {
//     return (cv::Mat_<float>(3,3) <<             0, -v.at<float>(2), v.at<float>(1),
//             v.at<float>(2),               0,-v.at<float>(0),
//             -v.at<float>(1),  v.at<float>(0),              0);
// }

// cv::Mat LocalMapping::ComputeF12(KeyFrame *&pKF1, KeyFrame *&pKF2)
// {
//     Eigen::Matrix3f R1w_E= pKF1->GetRotation();
//     cv::Mat R1w(3, 3, CV_32F);
//     cv::eigen2cv(R1w_E, R1w);

//     cv::Mat t1w = (cv::Mat_<float>(3, 1) << pKF1->GetTranslation()(0), pKF1->GetTranslation()(1), pKF1->GetTranslation()(2));

//     Eigen::Matrix3f R2w_E= pKF1->GetRotation();
//     cv::Mat R2w(3, 3, CV_32F);
//     cv::eigen2cv(R2w_E, R2w);

//     cv::Mat t2w = (cv::Mat_<float>(3, 1) << pKF2->GetTranslation()(0), pKF2->GetTranslation()(1), pKF2->GetTranslation()(2));

//     // cv::Mat R1w = pKF1->GetRotation();
//     // cv::Mat t1w = pKF1->GetTranslation();
//     // cv::Mat R2w = pKF2->GetRotation();
//     // cv::Mat t2w = pKF2->GetTranslation();

//     cv::Mat R12 = R1w*R2w.t();
//     cv::Mat t12 = -R1w*R2w.t()*t2w+t1w;
//     cv::Mat t12x = SkewSymmetricMatrix(t12);

//     Eigen::Matrix3f mK_E = pKF1->mK_;
//     cv::Mat K1(3, 3, CV_32F);
//     cv::eigen2cv(mK_E, K1);

//     cv::Mat K2(3, 3, CV_32F);
//     cv::eigen2cv(pKF2->mK_, K2);
//     // const cv::Mat &K1 = pKF1->mK_;
//     // const cv::Mat &K2 = pKF2->mK_;


//     return K1.t().inv()*t12x*R12*K2.inv();
// }

/** 
 * @brief imu初始化
 * @param priorG 陀螺仪偏置的信息矩阵系数，主动设置时一般bInit为true，也就是只优化最后一帧的偏置，这个数会作为计算信息矩阵时使用
 * @param priorA 加速度计偏置的信息矩阵系数
 * @param bFIBA 是否做BA优化，目前都为true
 */
void LocalMapping::InitializeIMU(float priorG, float priorA, bool bFIBA)
{
    cout<<"InitializeIMU"<<endl;
    // 1. 将所有关键帧放入列表及向量里，且查看是否满足初始化条件
    if (mbResetRequested)
    {
        cout<<"mbResetRequested"<<endl;
        return;
    }
        

    float minTime;
    int nMinKF;
    // 从时间及帧数上限制初始化，不满足下面的不进行初始化
    if (mbMonocular)
    {
        minTime = 1.5; //2.0
        nMinKF = 10;
    }
    else
    {
        minTime = 1.0;
        nMinKF = 10;
    }

    // 当前地图大于10帧才进行初始化
    if(mpAtlas->KeyFramesInMap()<nMinKF){
        cout<<"mpAltas keyframe number: "<< mpAtlas->KeyFramesInMap()<<" < "<<nMinKF<<endl;
        return;

    }
    cout<<"KeyFramesInMap()>nMinKF"<<endl;
    // Retrieve all keyframe in temporal order
    // 按照顺序存放目前地图里的关键帧，顺序按照前后顺序来，包括当前关键帧
    list<KeyFrame*> lpKF;
    KeyFrame* pKF = mpCurrentKeyFrame;
    while(pKF->mPrevKF)
    {
        lpKF.push_front(pKF);
        pKF = pKF->mPrevKF;
    }
    lpKF.push_front(pKF);
    // 同样内容再构建一个和lpKF一样的容器vpKF
    vector<KeyFrame*> vpKF(lpKF.begin(),lpKF.end());
    if(vpKF.size()<nMinKF)
        return;

    mFirstTs=vpKF.front()->mTimeStamp;
    if(mpCurrentKeyFrame->mTimeStamp-mFirstTs<minTime){
        cout<<"mpCurrentKeyFrame->mTimeStamp: "<< mpCurrentKeyFrame->mTimeStamp-mFirstTs<<" < "<<minTime<<endl;
        return;
    }
        
        

    // 正在做IMU的初始化，在tracking里面使用，如果为true，暂不添加关键帧
    bInitializing = true;

    // 先处理新关键帧，防止堆积且保证数据量充足
    while(CheckNewKeyFrames())
    {
        ProcessNewKeyFrame();
        vpKF.push_back(mpCurrentKeyFrame);
        lpKF.push_back(mpCurrentKeyFrame);
    }

    // 2. 正式IMU初始化
    const int N = vpKF.size();
    IMU::Bias b(0,0,0,0,0,0);

    // Compute and KF velocities mRwg estimation
    // 在IMU连一次初始化都没有做的情况下
    if (!mpCurrentKeyFrame->GetMap()->isImuInitialized())
    {
        Eigen::Matrix3f Rwg;
        Eigen::Vector3f dirG;
        dirG.setZero();

        int have_imu_num = 0;
        for(vector<KeyFrame*>::iterator itKF = vpKF.begin(); itKF!=vpKF.end(); itKF++)
        {
            if (!(*itKF)->mpImuPreintegrated)
                continue;
            if (!(*itKF)->mPrevKF)
                continue;

            have_imu_num++;
            // 初始化时关于速度的预积分定义Ri.t()*(s*Vj - s*Vi - Rwg*g*tij)
            dirG -= (*itKF)->mPrevKF->GetImuRotation() * (*itKF)->mpImuPreintegrated->GetUpdatedDeltaVelocity();
            // 求取实际的速度，位移/时间
            Eigen::Vector3f _vel = ((*itKF)->GetImuPosition() - (*itKF)->mPrevKF->GetImuPosition())/(*itKF)->mpImuPreintegrated->dT;
            (*itKF)->SetVelocity(_vel);
            (*itKF)->mPrevKF->SetVelocity(_vel);
        }

        if (have_imu_num < 6)
        {
            cout << "imu初始化失败, 由于带有imu预积分信息的关键帧数量太少" << endl;
            bInitializing=false;
            mbBadImu = true;
            return;
        }

        // dirG = sV1 - sVn + n*Rwg*g*t
        // 归一化，约等于重力在世界坐标系下的方向
        dirG = dirG/dirG.norm();
        // 原本的重力方向
        Eigen::Vector3f gI(0.0f, 0.0f, -1.0f);
        // 求“重力在重力坐标系下的方向”与的“重力在世界坐标系（纯视觉）下的方向”叉乘
        Eigen::Vector3f v = gI.cross(dirG);
        // 求叉乘模长
        const float nv = v.norm();
        // 求转角大小
        const float cosg = gI.dot(dirG);
        const float ang = acos(cosg);
        // v/nv 表示垂直于两个向量的轴  ang 表示转的角度，组成角轴
        Eigen::Vector3f vzg = v*ang/nv;
        // 获得重力坐标系到世界坐标系的旋转矩阵的初值
        Rwg = Sophus::SO3f::exp(vzg).matrix();
        mRwg = Rwg.cast<double>();
        mTinit = mpCurrentKeyFrame->mTimeStamp-mFirstTs;
    }
    else
    {
        mRwg = Eigen::Matrix3d::Identity();
        mbg = mpCurrentKeyFrame->GetGyroBias().cast<double>();
        mba = mpCurrentKeyFrame->GetAccBias().cast<double>();
    }

    mScale=1.0;

    // 暂时没发现在别的地方出现过
    mInitTime = mpTracker->mLastFrame.mTimeStamp-vpKF.front()->mTimeStamp;

    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    // 3. 计算残差及偏置差，优化尺度重力方向及速度偏置，偏置先验为0，双目时不优化尺度
    Optimizer::InertialOptimization(mpAtlas->GetCurrentMap(), mRwg, mScale, mbg, mba, mbMonocular, infoInertial, false, false, priorG, priorA);
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    // 尺度太小的话初始化认为失败
    if (mScale<5e-1)//1e-1
    {
        cout << "scale too small" << mScale<<endl;
        bInitializing=false;
        return;
    }
    cout << "imu initializel success!" << mScale<<endl;
    // 到此时为止，前面做的东西没有改变map

    // Before this line we are not changing the map
    {
        unique_lock<mutex> lock(mpAtlas->GetCurrentMap()->mMutexMapUpdate);
        // 尺度变化超过设定值，或者非单目时（无论带不带imu，但这个函数只在带imu时才执行，所以这个可以理解为双目imu）
        // if ((fabs(mScale - 1.f) > 0.00001) || !mbMonocular) {
        if (fabs(mScale - 1.f) > 0.00001) {
            // 4.1 恢复重力方向与尺度信息
            Sophus::SE3f Twg(mRwg.cast<float>().transpose(), Eigen::Vector3f::Zero());
            if(mpTracker->GetGlobalFrameAlignmentState() == Tracking::ALIGNED)
            {
                Eigen::Matrix3f Rg0w_ = mpCurrentKeyFrame->GetMap()->GetGlobalVIOAlignment();
                std::cout << "Set correction" << std::endl;
                mpCurrentKeyFrame->GetMap()->SetGlobalVIOAlignment(Rg0w_ * Twg.rotationMatrix().transpose());
            }
            mpAtlas->GetCurrentMap()->ApplyScaledRotation(Twg, mScale, true);
            // 4.2 更新普通帧的位姿，主要是当前帧与上一帧
            mpTracker->UpdateFrameIMU(mScale, vpKF[0]->GetImuBias(), mpCurrentKeyFrame);
        }

        // Check if initialization OK
        // 即使初始化成功后面还会执行这个函数重新初始化
        // 在之前没有初始化成功情况下（此时刚刚初始化成功）对每一帧都标记，后面的kf全部都在tracking里面标记为true
        // 也就是初始化之前的那些关键帧即使有imu信息也不算
        if (!mpAtlas->isImuInitialized())
            for (int i = 0; i < N; i++) {
                KeyFrame *pKF2 = vpKF[i];
                pKF2->bImu = true;
            }
    }

    // TODO 这步更新是否有必要做待研究，0.4版本是放在FullInertialBA下面做的
    // 这个版本FullInertialBA不直接更新位姿及三位点了
    mpTracker->UpdateFrameIMU(1.0,vpKF[0]->GetImuBias(),mpCurrentKeyFrame);

    // 设置经过初始化了
    if (!mpAtlas->isImuInitialized())
    {
        // ! 重要！标记初始化成功
        mpAtlas->SetImuInitialized();
        mpTracker->t0IMU = mpTracker->mCurrentFrame.mTimeStamp;
        mpCurrentKeyFrame->bImu = true;
    }

    std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();
    // 代码里都为true
    if (bFIBA)
    {
        // 5. 承接上一步纯imu优化，按照之前的结果更新了尺度信息及适应重力方向，所以要结合地图进行一次视觉加imu的全局优化，这次带了MP等信息
        // ! 1.0版本里面不直接赋值了，而是将所有优化后的信息保存到变量里面
        if (priorA!=0.f)
            Optimizer::FullInertialBA(mpAtlas->GetCurrentMap(), 100, false, mpCurrentKeyFrame->mnId, NULL, true, priorG, priorA);
        else
            Optimizer::FullInertialBA(mpAtlas->GetCurrentMap(), 100, false, mpCurrentKeyFrame->mnId, NULL, false);
    }

    std::chrono::steady_clock::time_point t5 = std::chrono::steady_clock::now();

    Verbose::PrintMess("Global Bundle Adjustment finished\nUpdating map ...", Verbose::VERBOSITY_NORMAL);

    // Get Map Mutex
    unique_lock<mutex> lock(mpAtlas->GetCurrentMap()->mMutexMapUpdate);

    unsigned long GBAid = mpCurrentKeyFrame->mnId;

    // Process keyframes in the queue
    // 6. 处理一下新来的关键帧，这些关键帧没有参与优化，但是这部分bInitializing为true，只在第2次跟第3次初始化会有新的关键帧进来
    // 这部分关键帧也需要被更新
    while(CheckNewKeyFrames())
    {
        ProcessNewKeyFrame();
        vpKF.push_back(mpCurrentKeyFrame);
        lpKF.push_back(mpCurrentKeyFrame);
    }

    // Correct keyframes starting at map first keyframe
    // 7. 更新位姿与三维点
    // 获取地图中初始关键帧，第一帧肯定经过修正的
    list<KeyFrame*> lpKFtoCheck(mpAtlas->GetCurrentMap()->mvpKeyFrameOrigins.begin(),mpAtlas->GetCurrentMap()->mvpKeyFrameOrigins.end());

    // 初始就一个关键帧，顺藤摸瓜找到父子相连的所有关键帧
    // 类似于树的广度优先搜索，其实也就是根据父子关系遍历所有的关键帧，有的参与了FullInertialBA有的没参与
    while(!lpKFtoCheck.empty())
    {
        // 7.1 获得这个关键帧的子关键帧
        KeyFrame* pKF = lpKFtoCheck.front();
        const set<KeyFrame*> sChilds = pKF->GetChilds();
        Sophus::SE3f Twc = pKF->GetPoseInverse();  // 获得关键帧的优化前的位姿
        // 7.2 遍历这个关键帧所有的子关键帧
        for(set<KeyFrame*>::const_iterator sit=sChilds.begin();sit!=sChilds.end();sit++)
        {
            // 确认是否能用
            KeyFrame* pChild = *sit;
            if(!pChild || pChild->isBad())
                continue;

            // 这个判定为true表示pChild没有参与前面的优化，因此要根据已经优化过的更新，结果全部暂存至变量
            if(pChild->mnBAGlobalForKF!=GBAid)
            {
                // pChild->GetPose()也是优化前的位姿，Twc也是优化前的位姿
                // 7.3 因此他们的相对位姿是比较准的，可以用于更新pChild的位姿
                Sophus::SE3f Tchildc = pChild->GetPose() * Twc;
                // 使用相对位姿，根据pKF优化后的位姿更新pChild位姿，最后结果都暂时放于mTcwGBA
                pChild->mTcwGBA = Tchildc * pKF->mTcwGBA;

                // 7.4 使用相同手段更新速度
                Sophus::SO3f Rcor = pChild->mTcwGBA.so3().inverse() * pChild->GetPose().so3();
                if(pChild->isVelocitySet()){
                    pChild->mVwbGBA = Rcor * pChild->GetVelocity();
                }
                else {
                    Verbose::PrintMess("Child velocity empty!! ", Verbose::VERBOSITY_NORMAL);
                }

                pChild->mBiasGBA = pChild->GetImuBias();
                pChild->mnBAGlobalForKF = GBAid;

            }
            // 加入到list中，再去寻找pChild的子关键帧
            lpKFtoCheck.push_back(pChild);
        }

        // 7.5 此时pKF的利用价值就没了，但是里面的数值还都是优化前的，优化后的全部放于类似mTcwGBA这样的变量中
        // 所以要更新到正式的状态里，另外mTcwBefGBA要记录更新前的位姿，用于同样的手段更新三维点用
        pKF->mTcwBefGBA = pKF->GetPose();
        pKF->SetPose(pKF->mTcwGBA);

        // 速度偏置同样更新
        if(pKF->bImu)
        {
            pKF->mVwbBefGBA = pKF->GetVelocity();
            pKF->SetVelocity(pKF->mVwbGBA);
            pKF->SetNewBias(pKF->mBiasGBA);
        } else {
            cout << "KF " << pKF->mnId << " not set to inertial!! \n";
        }

        // pop
        lpKFtoCheck.pop_front();
    }

    // Correct MapPoints
    // 8. 更新三维点，三维点在优化后同样没有正式的更新，而是找了个中间变量保存了优化后的数值
    const vector<MapPoint*> vpMPs = mpAtlas->GetCurrentMap()->GetAllMapPoints();

    for(size_t i=0; i<vpMPs.size(); i++)
    {
        MapPoint* pMP = vpMPs[i];

        if(pMP->isBad())
            continue;

        // 8.1 如果这个点参与了全局优化，那么直接使用优化后的值来赋值
        if(pMP->mnBAGlobalForKF==GBAid)
        {
            // If optimized by Global BA, just update
            pMP->SetWorldPos(pMP->mPosGBA);
        }
        // 如果没有参与，与关键帧的更新方式类似
        else
        {
            // Update according to the correction of its reference keyframe
            KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();

            if(pRefKF->mnBAGlobalForKF!=GBAid)
                continue;

            // Map to non-corrected camera
            // 8.2 根据优化前的世界坐标系下三维点的坐标以及优化前的关键帧位姿计算这个点在关键帧下的坐标
            Eigen::Vector3f Xc = pRefKF->mTcwBefGBA * pMP->GetWorldPos();

            // Backproject using corrected camera
            // 8.3 根据优化后的位姿转到世界坐标系下作为这个点优化后的三维坐标
            pMP->SetWorldPos(pRefKF->GetPoseInverse() * Xc);
        }
    }

    Verbose::PrintMess("Map updated!", Verbose::VERBOSITY_NORMAL);

    mnKFs=vpKF.size();
    mIdxInit++;

    // 9. 再有新的来就不要了~不然陷入无限套娃了
    for(list<KeyFrame*>::iterator lit = mlNewKeyFrames.begin(), lend=mlNewKeyFrames.end(); lit!=lend; lit++)
    {
        (*lit)->SetBadFlag();
        delete *lit;
    }
    mlNewKeyFrames.clear();

    mpTracker->mState=Tracking::OK;
    bInitializing = false;

    mpCurrentKeyFrame->GetMap()->IncreaseChangeIndex();

    return;
}

/**
 * @brief 通过BA优化进行尺度更新，关键帧小于100，使用了所有关键帧的信息，但只优化尺度和重力方向。每10s在这里的时间段内时多次进行尺度更新
 */
void LocalMapping::ScaleRefinement()
{
    // Minimum number of keyframes to compute a solution
    // Minimum time (seconds) between first and last keyframe to compute a solution. Make the difference between monocular and stereo
    // unique_lock<mutex> lock0(mMutexImuInit);
    if (mbResetRequested)
        return;

    // Retrieve all keyframes in temporal order
    // 1. 检索所有的关键帧（当前地图）
    list<KeyFrame*> lpKF;
    KeyFrame* pKF = mpCurrentKeyFrame;
    while(pKF->mPrevKF)
    {
        lpKF.push_front(pKF);
        pKF = pKF->mPrevKF;
    }
    lpKF.push_front(pKF);
    vector<KeyFrame*> vpKF(lpKF.begin(),lpKF.end());

    // 加入新添加的帧
    while(CheckNewKeyFrames())
    {
        ProcessNewKeyFrame();
        vpKF.push_back(mpCurrentKeyFrame);
        lpKF.push_back(mpCurrentKeyFrame);
    }

    const int N = vpKF.size();

    // 2. 更新旋转与尺度
    // 待优化变量的初值
    mRwg = Eigen::Matrix3d::Identity();
    mScale=1.0;

    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    // 优化重力方向与尺度
    Optimizer::InertialOptimization(mpAtlas->GetCurrentMap(), mRwg, mScale);
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    if (mScale<1e-1) // 1e-1
    {
        cout << "scale too small" << endl;
        bInitializing=false;
        return;
    }

    Sophus::SO3d so3wg(mRwg);


    // Before this line we are not changing the map
    // 3. 开始更新地图
    unique_lock<mutex> lock(mpAtlas->GetCurrentMap()->mMutexMapUpdate);
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    // 3.1 如果尺度更新较多，或是在双目imu情况下更新地图
    // 0.4版本这个值为0.00001
    if ((fabs(mScale-1.f)>0.002)||!mbMonocular)
    {
        Sophus::SE3f Tgw(mRwg.cast<float>().transpose(),Eigen::Vector3f::Zero());
        mpAtlas->GetCurrentMap()->ApplyScaledRotation(Tgw,mScale,true);
        mpTracker->UpdateFrameIMU(mScale,mpCurrentKeyFrame->GetImuBias(),mpCurrentKeyFrame);
    }
    std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();

    // 3.2 优化的这段时间新进来的kf全部清空不要
    for(list<KeyFrame*>::iterator lit = mlNewKeyFrames.begin(), lend=mlNewKeyFrames.end(); lit!=lend; lit++)
    {
        (*lit)->SetBadFlag();
        delete *lit;
    }
    mlNewKeyFrames.clear();

    double t_inertial_only = std::chrono::duration_cast<std::chrono::duration<double> >(t1 - t0).count();

    // To perform pose-inertial opt w.r.t. last keyframe
    mpCurrentKeyFrame->GetMap()->IncreaseChangeIndex();

    return;
}

/**
 * @brief 返回是否正在做IMU的初始化，在tracking里面使用，如果为true，暂不添加关键帧
 */
bool LocalMapping::IsInitializing()
{
    return bInitializing;
}

/**
 * @brief 获取当前关键帧的时间戳，System::GetTimeFromIMUInit()中调用
 */
double LocalMapping::GetCurrKFTime()
{

    if (mpCurrentKeyFrame)
    {
        return mpCurrentKeyFrame->mTimeStamp;
    }
    else
        return 0.0;
}

/**
 * @brief 获取当前关键帧
 */
KeyFrame* LocalMapping::GetCurrKF()
{
    return mpCurrentKeyFrame;
}

} //namespace ORB_SLAM
