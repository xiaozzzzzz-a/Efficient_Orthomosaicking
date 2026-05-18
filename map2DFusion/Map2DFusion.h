#ifndef MAP2DFUSION_H
#define MAP2DFUSION_H

#include "Map2D.h"
#include "opencv2/opencv.hpp"
#include "sophus/se3.hpp"
#include "LocalMapping.h"
using namespace ORB_SLAM3;

namespace Map2DFusion
{
    class TrajectoryLengthCalculator
    {
    public:
        TrajectoryLengthCalculator();

        ~TrajectoryLengthCalculator();

        void feed(cv::Point3d position);

    private:
        double length;
        cv::Point3d lastPosition;
    };


    class TestSystem 
    {
    public:
        TestSystem();

        ~TestSystem();

       

        int TestMap2DItem();

        bool obtainFrame(std::pair<cv::Mat, Sophus::SE3d> &frame);

        //bool obtainSlamFrame(std::pair<cv::Mat, Sophus::SE3d> &frame);

        bool obtainSlamFrame(FusionFrame& frame);

        int testMap2D();

        int Map2DWithSLAM();

        virtual void run();
    
        void saveCurrentFramebuffer(const std::string& filename, int width, int height);
    std::shared_ptr<std::ifstream>      in;
    std::shared_ptr<Map2D>       map;
    std::shared_ptr<TrajectoryLengthCalculator> lengthCalculator;
    bool myStopFLag = false;
    bool myRunningFlag = true;
    bool gps_origin = true;
    std::string datapath = "/media/xiao/data2/image_stitching/Rover-slam-gnss-exif/Examples/Monocular/map_fusion.yaml";
    cv::FileStorage config;
    static TestSystem* instance;
    int newwidth, newheight;
    bool firstFrameReceived = false;
    };

    int _main_map2dfusion(int argc, char **argv);
}
#endif // MAP2DFUSION_H