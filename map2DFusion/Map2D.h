#ifndef MAP2D_H
#define MAP2D_H
#include <deque>
#include <opencv2/features2d/features2d.hpp>
#include <sophus/se3.hpp>
#include <shared_mutex>
#define  ELE_PIXELS 256

#include "LocalMapping.h"
using namespace ORB_SLAM3;

struct PinHoleParameters
{
    PinHoleParameters(){}
    PinHoleParameters(int _w,int _h,double _fx,double _fy,double _cx,double _cy)
        :w(_w),h(_h),fx(_fx),fy(_fy),cx(_cx),cy(_cy){}
    double w,h,fx,fy,cx,cy; 
};

struct Map2DPrepare
{
    uint queueSize(){return _frames.size();};

    bool prepare(const Sophus::SE3d& plane,const PinHoleParameters& camera,
                 const std::deque<std::pair<cv::Mat, Sophus::SE3d> >& frames);

    cv::Point2d Project(const cv::Point3d& pt)
    {
        double zinv = 1./pt.z;
        return cv::Point2d(_camera.fx*pt.x*zinv + _camera.cx,
                            _camera.fy*pt.y*zinv+_camera.cy);
    }

    cv::Point3d UnProject(const cv::Point2d& pt)
    {
        return cv::Point3d((pt.x-_camera.cx)*_fxinv,
                           (pt.y-_camera.cy)*_fyinv,1.);
    }
    std::deque<std::pair<cv::Mat,Sophus::SE3d> > getFrames()
    {
        std::shared_lock<std::shared_mutex> lock(mutexFrames);
        return _frames;
    }
    PinHoleParameters _camera;
    double _fxinv,_fyinv;
    Sophus::SE3d _plane;
    std::deque<std::pair<cv::Mat,Sophus::SE3d>> _frames;
    std::shared_mutex mutexFrames;
};

class Map2D
{
public:
    enum Map2DType{NoType=0,TypeCPU=1,TypeGPU=2,TypeMultiBandCPU=3,TypeRender=4};

    static std::shared_ptr<Map2D> create(int type=TypeCPU,bool thread=false);

    virtual ~Map2D(){}

    virtual bool prepare(const Sophus::SE3d& plane,const PinHoleParameters& camera,
            const std::deque<std::pair<cv::Mat,Sophus::SE3d> >& frames){return false;}

    virtual bool prepare(const Sophus::SE3d& plane,const PinHoleParameters& camera,
            const FusionFrame& frames){return false;}
    
    virtual bool feed(cv::Mat img,const Sophus::SE3d& pose){return false;}

    virtual bool feed(FusionFrame& frame){return false;}

    virtual void draw(){}

    virtual void draw2D(){}
    
    virtual void RenderTime2File(){}

    virtual bool save(const std::string& filename){return false;}
    virtual bool save_1(const std::string& filename){return false;}

    virtual uint queueSize(){return 0;}
};

#endif // MAP2D_H