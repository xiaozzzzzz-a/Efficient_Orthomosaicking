#ifndef MAP2DCPU_H
#define MAP2DCPU_H
#include "Map2D.h"
#include <thread>
#include <shared_mutex>
#include <boost/make_shared.hpp>
#define  ELE_PIXELS 256
#include "LocalMapping.h"
using namespace ORB_SLAM3;

class Map2DCPU :public Map2D
{
    typedef Map2DPrepare Map2DCPUPrepare;
    struct Map2DCPUEle
    {
        Map2DCPUEle():texName(0),Ischanged(false){}
        ~Map2DCPUEle();
        cv::Mat img;
        uint    texName;
        bool    Ischanged;
        std::shared_mutex mutex_data;
    };

    struct Map2DCPUData
    {
        Map2DCPUData():_w(0),_h(0){}
        Map2DCPUData(double eleSize_,double lengthPixel_,cv::Point3d max_,cv::Point3d min_,
                     int w_,int h_,const std::vector<std::shared_ptr<Map2DCPUEle> >& d_)
            :_eleSize(eleSize_),_eleSizeInv(1./eleSize_),
              _lengthPixel(lengthPixel_),_lengthPixelInv(1./lengthPixel_),
              _min(min_),_max(max_),_w(w_),_h(h_),_data(d_){}

        bool prepare(std::shared_ptr<Map2DCPUPrepare> prepared);

        double eleSize()const{return _eleSize;}
        double lengthPixel()const{return _lengthPixel;}
        double eleSizeInv()const{return _eleSizeInv;}
        double lengthPixelInv()const{return _lengthPixelInv;}
        const cv::Point3d& min()const{return _min;}
        const cv::Point3d& max()const{return _max;}
        const int w()const{return _w;}
        const int h()const{return _h;}

        std::vector<std::shared_ptr<Map2DCPUEle> > data()
        {
            std::shared_lock<std::shared_mutex> readLock(mutex_data);
            return _data;
        }

        std::shared_ptr<Map2DCPUEle> ele(uint idx)
        {
            std::shared_lock<std::shared_mutex> lock(mutex_data);
            if(idx >_data.size()) return std::shared_ptr<Map2DCPUEle>();
            else if(!_data[idx].get())
            {
                _data[idx]=std::shared_ptr<Map2DCPUEle>(new Map2DCPUEle());
            }
            return _data[idx];
        }
    private:
        double      _eleSize,_lengthPixel,_eleSizeInv,_lengthPixelInv;
        cv::Point3d _max,_min;
        int         _w,_h;
        std::vector<std::shared_ptr<Map2DCPUEle> >  _data;
        std::shared_mutex mutex_data;
    };

public:

    Map2DCPU(bool thread=true);

    //pcl::PointCloud<pcl::PointXYZRGB>::Ptr imageToPointCloud(const cv::Mat& image, int x0, int y0);
    //pcl::visualization::CloudViewer viewer;

    virtual ~Map2DCPU(){_valid=false;}

    virtual bool prepare(const Sophus::SE3d& plane,const PinHoleParameters& camera,
                    const std::deque<std::pair<cv::Mat,Sophus::SE3d> >& frames);
    
    virtual bool feed(cv::Mat img,const Sophus::SE3d& pose);//world coordinate

    virtual void draw();

    virtual void draw2D();
    virtual bool save(const std::string& filename);


    virtual uint queueSize(){
        if(prepared.get()) return prepared->queueSize();
        else               return 0;
    }

    virtual void run();

private:

    bool getFrame(std::pair<cv::Mat,Sophus::SE3d>& frame);
    bool renderFrame(const std::pair<cv::Mat,Sophus::SE3d>& frame);

    bool renderFrame(const FusionFrame& frame);
    bool spreadMap(double xmin,double ymin,double xmax,double ymax);

    std::shared_ptr<Map2DCPUPrepare> prepared;
    std::shared_ptr<Map2DCPUData> data;
    std::shared_mutex mutex;

    int i =0;
    bool _valid,_thread,_changed;
    std::thread run_thread;
    //std::thread _thread;
    cv::Mat weightImage;
    int alpha;
};

#endif // MAP2DCPU_H