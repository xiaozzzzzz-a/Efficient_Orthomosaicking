#include <iostream>
#include <fstream>
#include <sophus/so3.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include "Map2DFusion.h"
#include "Map2D.h"
#include <unistd.h>
// #include <yaml-cpp/yaml.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "DataTrans.h"
#include "MultiBandMap2DCPU.h"
#include "LocalMapping.h"

using namespace std;
using namespace ORB_SLAM3;
namespace Map2DFusion
{
    TrajectoryLengthCalculator::TrajectoryLengthCalculator() : length(-1)
    {

    }

    TrajectoryLengthCalculator::~TrajectoryLengthCalculator()
    {
        cout << "TrajectoryLength:" << length << endl;
    }

    void TrajectoryLengthCalculator::feed(cv::Point3d position)
    {
        if (length < 0)
        {
            length = 0;
            lastPosition = position;
        }
        else
        {
            length += cv::norm(position - lastPosition);
            lastPosition = position;
        }
    }

    TestSystem::TestSystem()
    {
        //datapath = "/media/xiao/data2/image_stitching/Rover-slam-fusion/Examples/Monocular/map_fusion.yaml";
        //config = YAML::LoadFile(datapath+"/config.yaml");

        config = cv::FileStorage(datapath.c_str(), cv::FileStorage::READ);
    }

    TestSystem::~TestSystem()
    {
        //stop();
        cout<<"_testSystem xigou"<<endl;
        //map->RenderTime2File();
        cout<<"_testSystem xigou"<<endl;
        map->save("resultss.png");
        // if(map.get())
        //     map->save("resultss.png");
    
        map = shared_ptr<Map2D>();
    }

    bool TestSystem::obtainFrame(std::pair<cv::Mat,Sophus::SE3d>& frame)
    {
        string line;
        if(!getline(*in, line)) return false;
        stringstream ifs(line);
        string imgfile;
        ifs>>imgfile;
        imgfile = datapath+"/rgb/"+imgfile+".jpg";
        //pi::timer.enter("obtainFrame");
        frame.first=cv::imread(imgfile);
        if(frame.first.empty()) return false;

        double x,y,z;
        double rx,ry,rz,rw;
        ifs>>x >> y >> z >> rx >> ry >> rz >> rw;
        Eigen::Quaterniond rotation(rw, rx, ry, rz);
        Eigen::Vector3d translation(x, y, z);
        frame.second.setQuaternion(rotation);
        frame.second.translation() = translation;

        if(gps_origin)
        {
            if(!lengthCalculator.get()) lengthCalculator=std::shared_ptr<TrajectoryLengthCalculator>(
                        new TrajectoryLengthCalculator());
        }

        return true;
    }

    bool TestSystem::obtainSlamFrame(FusionFrame& frame)
    {
        bool success;

        if (!firstFrameReceived) {
            // 第一次必须等到
            Trans.consumption(frame); 
            success = true;
            firstFrameReceived = true;
        } 
        else {
            // 之后用超时版本
            success = Trans.consumption(frame, std::chrono::milliseconds(100000));
        }

        if (!success) {
            std::cerr << "[WARN] obtainSlamFrame timed out waiting 50 seconds." << std::endl;
            return false;
        }

//  Eigen::Matrix3d R = Eigen::AngleAxisd(M_PI, Eigen::Vector3d(0,0,1)).toRotationMatrix();
// Sophus::SE3d zRot180(R, Eigen::Vector3d::Zero());
// frame.pose = frame.pose * zRot180;

// Eigen::Matrix3d Rz180 = Eigen::AngleAxisd(M_PI, Eigen::Vector3d(0,0,1)).toRotationMatrix();

// // 原始旋转和平移
// Eigen::Matrix3d R_orig = frame.pose.rotationMatrix();
// Eigen::Vector3d t_orig = frame.pose.translation();

// // 位置取Z轴对称
// Eigen::Vector3d t_new(-t_orig.x(), -t_orig.y(), t_orig.z());

// // 方向也要绕Z轴旋转180°
// Eigen::Matrix3d R_new = R_orig * Rz180;

// // 构造新的SE3
// frame.pose = Sophus::SE3d(R_new, t_new);

        std::cout << "[INFO] Trans consumed FusionFrame successfully!" << std::endl;

        // 轨迹长度计算
        if (gps_origin) {
            if (!lengthCalculator.get())
                lengthCalculator = std::make_shared<TrajectoryLengthCalculator>();

            Eigen::Vector3d t = frame.pose.translation();
            cv::Point3d point(t.x(), t.y(), t.z());
            lengthCalculator->feed(point);
        }

        return true;
    }


    // bool TestSystem::obtainSlamFrame(std::pair<cv::Mat, Sophus::SE3d> &frame)
    // {
    //     Trans.consumption(frame);
    //     cout<<"Trans consumption frame success!"<<endl;
    //     if(gps_origin)
    //     {
    //         if(!lengthCalculator.get())
    //             lengthCalculator = shared_ptr<TrajectoryLengthCalculator>(new TrajectoryLengthCalculator());
    //             Eigen::Vector3d t = frame.second.translation();
    //             cv::Point3d point(t.x(), t.y(), t.z());
    //             lengthCalculator->feed(point);
    //     }
    //     return true;
    // }

    // int TestSystem::testMap2D()
    // {
    //     cout<<"Act = TestMap2D\n";
    //     datapath = "/media/xiao/data1/datasets_machine/phantom3-npu/";
    // }

    int TestSystem::testMap2D()
    {
        cout<<"Act=TestMap2D\n";

        //datapath=svar.GetString("Map2D.DataPath","");
        datapath = "/media/xiao/data1/datasets_machine/phantom3-npu/";//修改
        if(!datapath.size())
        {
            cerr<<"Map2D.DataPath is not seted!\n";
            return -1;
        }
        
        if(config["Plane"].empty())
        {
            cerr<<"Plane is not defined!\n";
            return -2;
        }
        std::string trajectory = "/media/xiao/data2/image_stitching/Rover-slam-fusion/CameraTrajectory_map.txt";
        if(!in.get())
            in = std::shared_ptr<ifstream>(new ifstream((trajectory).c_str()));

        if(!in->is_open())
        {
            cerr<<"Can't open file "<<(datapath+"/trajectory.txt")<<endl;
            return -3;
        }
        deque<std::pair<cv::Mat,Sophus::SE3d>> frames;
        int iEnd ;
        config["PrepareFrameNum"] >> iEnd;
        for(int i = 0, iend=iEnd;  i<iend; i++)
        //for(int i = 0, iend=10;  i<iend; i++)
        {
            std::pair<cv::Mat, Sophus::SE3d> frame;
            if(!obtainFrame(frame)) break;
            frames.push_back(frame);
        }
        cout<<"Loaded "<<frames.size()<<" frames.\n";

        if(!frames.size()) return -4;
        int type;
        config["Map2D.Type"]>>type;
        map = Map2D::create(type);

        if(!map.get())
        {
            cerr<<"No map2d created!\n";
            return -5;
        }

        std::vector<double> vecP;
        config["Camera.Paraments"] >>vecP;

        //提取config文件中的se(3)
        // std::vector<double> plane;
        // config["Plane"] >> plane;

        // //std::vector<double> plane = config["Plane"].as<std::vector<double>>();
        // Eigen::Quaterniond quat(plane[3], plane[4], plane[5], plane[6]);
        // Eigen::Vector3d trans(plane[0], plane[1], plane[2]);
        // Sophus::SE3d se3(quat, trans);
        
        // //std::vector<double> vecP = config["Camera.Paraments"].as<std::vector<double>>();//此处config为数据集的配置文件
        // map->prepare(se3, PinHoleParameters(vecP[0],vecP[1],vecP[2],vecP[3],vecP[4],vecP[5]),
        //             frames);
    //****************************
        std::vector<double> plane;
        Trans_PlaneVec.consumption(plane);
        Eigen::Quaterniond quat(plane[3], plane[4], plane[5], plane[6]);
        Eigen::Vector3d trans(plane[0], plane[1], plane[2]);
        Sophus::SE3d se3(quat, trans);
        map->prepare(se3, PinHoleParameters(vecP[0],vecP[1],vecP[2],vecP[3],vecP[4],vecP[5]),
                    frames);

        //******************
        cout<<"plane param: ";
        for(int i = 0 ; i < plane.size(); i++)
        {
            cout<<plane[i]<<", ";
        }
        cout<<endl;
        // Sophus::SE3d plane;
        // Trans_Plane.consumption(plane);

        // std::cout<<"plane so3: "<<plane.unit_quaternion().coeffs().transpose()<<std::endl;
        // std::cout<<"plane transtition: "<<plane.translation().transpose()<<std::endl;
        // map->prepare(plane, PinHoleParameters(vecP[0],vecP[1],vecP[2],vecP[3],vecP[4],vecP[5]),
        //             frames);

        cout<<"********************get plane param success!"<<endl;    
        int autofeedframes;
        config["AutoFeedFrames"] >> autofeedframes;
        if(autofeedframes == 1)
        {

            if(!glfwInit())
            {
                std::cerr << "Failed to initialize GLFW" << std::endl;
            }

            GLFWwindow* window = glfwCreateWindow(2200, 1600, "2D Map Stitching", NULL, NULL);

            if(!window) 
            {
                std::cerr << "Failed to create GLFW window" << std::endl;
                glfwTerminate();
            }

            glfwMakeContextCurrent(window);
            glewExperimental = GL_TRUE;
            if (glewInit() != GLEW_OK) 
            {
                std::cerr << "Failed to initialize GLEW" << std::endl;
            }
            glEnable(GL_TEXTURE_2D);
            while (!glfwWindowShouldClose(window)) 
            {
                if(map->queueSize()<2)
                {
                    std::pair<cv::Mat, Sophus::SE3d> frame;
                    if(!obtainFrame(frame)) break;
                    map->feed(frame.first, frame.second);
                }
                map->draw2D();
                glBindTexture(GL_TEXTURE_2D, 0);
                glfwSwapBuffers(window);
                glfwPollEvents();
            }
        }
    }

    void TestSystem::saveCurrentFramebuffer(const std::string& filename, int width, int height)
    {
        std::vector<unsigned char> pixels(3 * width * height);

        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadBuffer(GL_FRONT);
        glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());

        cv::Mat img(height, width, CV_8UC3, pixels.data());
        cv::Mat imgFlipped;
        cv::flip(img, imgFlipped, 0);

        // ⭐ Gamma correction from linear RGB to sRGB
        imgFlipped.convertTo(imgFlipped, CV_32FC3, 1.0/255.0);
        cv::pow(imgFlipped, 1.0/2.2, imgFlipped);
        imgFlipped = imgFlipped * 255.0;
        imgFlipped.convertTo(imgFlipped, CV_8UC3);

        cv::imwrite(filename, imgFlipped);
        std::cout << "[INFO] Saved framebuffer with gamma correction to " << filename << std::endl;
    }


    int TestSystem::Map2DWithSLAM()
    {
        cout<< "Map2D.Act = Map2DWithSLAM\n";
        //deque<std::pair<cv::Mat, Sophus::SE3d>> frames;

        deque<FusionFrame> frames;
        int iEnd ;
        config["PrepareFrameNum"] >> iEnd;
        for (int i = 0 , iend = iEnd; i < iend; i++)
        {
            //std::pair<cv::Mat, Sophus::SE3d> frame;
            FusionFrame frame;
            if (!obtainSlamFrame(frame)) break;
            frames.push_back(frame);
        }

        cout<< "Loaded " <<frames.size() << " frames.\n ";
        if(!frames.size()) return -4;
        int type;
        config["Map2D.Type"]>>type;
        map = Map2D::create(type);
        if(!map.get())
        {
            cerr <<"No map2d created!\n";
            return -5;
        }

        std::vector<double> vecP;
        config["Camera.Paraments"] >>vecP;
        if (vecP.size()!= 6)
        {
            cerr <<"Invalid camera parameters!\n";
            return -5;
        }
        Sophus::SE3d plane;
        Trans_Plane.consumption(plane);
        cout<<"Trans_plane consumption success!"<<endl;

        deque<std::pair<cv::Mat, Sophus::SE3d>> frames_ori;

        for (const auto& f : frames) {

            frames_ori.emplace_back(f.image, f.pose);
        }

        map->prepare(plane, PinHoleParameters(vecP[0], vecP[1], vecP[2], vecP[3], vecP[4], vecP[5]), frames_ori);
        int autofeedframes;
        config["AutoFeedFrames"] >> autofeedframes;
        cout<<"11111"<<endl;
        if(autofeedframes)
        {
            int i =0;
            if(!glfwInit())
            {
                std::cerr << "Failed to initialize GLFW" << std::endl;
            }

            GLFWwindow* window = glfwCreateWindow(2200, 1600, "2D Map Stitching", NULL, NULL);
           

            if(!window) 
            {
                std::cerr << "Failed to create GLFW window" << std::endl;
                glfwTerminate();
            }

            glfwMakeContextCurrent(window);
            // 将 map 指针传递给 GLFW
            glfwSetWindowUserPointer(window, dynamic_cast<MultiBandMap2DCPU*>(map.get()));

            // 设置滚轮缩放回调
            glfwSetScrollCallback(window, MultiBandMap2DCPU::scroll_callback);
            glfwSetKeyCallback(window, MultiBandMap2DCPU::key_callback);  
            glewExperimental = GL_TRUE;
            if (glewInit() != GLEW_OK) 
            {
                std::cerr << "Failed to initialize GLEW" << std::endl;
            }
            glEnable(GL_TEXTURE_2D);
            int num_save = 0;
            while (!glfwWindowShouldClose(window)) 
            {
                cout<<"map->queueSize(): "<<map->queueSize()<<endl;
                
                FusionFrame frame;
                if(!obtainSlamFrame(frame)) 
                {
                    glfwSetWindowShouldClose(window, true);
                    cout<<"set window should close"<<endl;
                    break;
                }

                //  if (i > 0 && (i % 50 == 0)) {
                //     std::ostringstream fname;
                //     fname << "result_" << std::setw(5) << std::setfill('0') << i << ".png";
                //     cout << "[INFO] Saving intermediate result: " << fname.str() << endl;
                //     map->save(fname.str());
                //     num_save++;
                // }

                map->feed(frame);
                //map->feed(frame.image, frame.pose); 
                map->draw2D();
                
                glBindTexture(GL_TEXTURE_2D, 0);
                glfwSwapBuffers(window);
                glfwPollEvents();
                i++;
                
            }
            int w, h;
            glfwGetFramebufferSize(window, &w, &h);
            saveCurrentFramebuffer("FinalRenderOutput.png", w, h);
            glfwDestroyWindow(window);
            glfwTerminate();

            cout<<"save succes!"<<endl;
            map->save("resultss.png");
            cout<<"save succes!"<<endl;
        }
    }

    void TestSystem::run()
    {
        string act;
        config["Act"] >> act;
        cout<<act<<endl;
        if(act == "TestMap2D" || act == "Default") testMap2D();
        else if(act == "Map2DWithSLAM") Map2DWithSLAM();
        else std::cout <<"No act" << act << "!\n";
    }

    
}