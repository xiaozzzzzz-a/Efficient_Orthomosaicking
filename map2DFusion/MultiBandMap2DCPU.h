#ifndef MultiBandMap2DCPU_H
#define MultiBandMap2DCPU_H
#include "Map2D.h"
#include "TilePublisher.h"
#include "LocalMapping.h"
#include "thread"
#include <sophus/se3.hpp>
#include <memory>
#include <shared_mutex>
#include <boost/make_shared.hpp>
#include <GLFW/glfw3.h>

using namespace ORB_SLAM3;
using namespace std;
class MultiBandMap2DCPU:public Map2D{
    typedef Map2DPrepare MultiBandMap2DCPUPrepare;

    struct MultiBandMap2DCPUEle
    {
        MultiBandMap2DCPUEle():texName(0),Ischanged(false){}
        ~MultiBandMap2DCPUEle();

        static bool normalizeUsingWeightMap(const cv::Mat& weight, cv::Mat& src);
        static bool mulWeightMap(const cv::Mat& weight, cv::Mat& src);

        cv::Mat blend(const std::vector<std::shared_ptr<MultiBandMap2DCPUEle> >& neighbors
                      =std::vector<std::shared_ptr<MultiBandMap2DCPUEle> >());
        bool updateTexture(const std::vector<std::shared_ptr<MultiBandMap2DCPUEle> >& neighbors
                =std::vector<std::shared_ptr<MultiBandMap2DCPUEle> >());

        std::vector<cv::Mat> pyr_laplace;
        std::vector<cv::Mat> weights;
        uint    texName;
        bool    Ischanged;
        std::shared_mutex mutexData;
    };

    struct MultiBandMap2DCPUData
    {
        MultiBandMap2DCPUData():_w(0),_h(0){}
        MultiBandMap2DCPUData(double eleSize_,double lengthPixel_,cv::Point3d max_,cv::Point3d min_,
                     int w_,int h_,const std::vector<std::shared_ptr<MultiBandMap2DCPUEle> >& d_);
        bool   prepare(std::shared_ptr<MultiBandMap2DCPUPrepare> prepared);// only done Once!
        double eleSize()const{return _eleSize;}
        double lengthPixel()const{return _lengthPixel;}
        double eleSizeInv()const{return _eleSizeInv;}
        double lengthPixelInv()const{return _lengthPixelInv;}
        const cv::Point3d& gpsOrigin()const{return _gpsOrigin;}
        const cv::Point3d& min()const{return _min;}
        const cv::Point3d& max()const{return _max;}
        const int w()const{return _w;}
        const int h()const{return _h;}

        std::vector<std::shared_ptr<MultiBandMap2DCPUEle> > data(){std::shared_lock<std::shared_mutex> lock(mutexData);return _data;}

        std::shared_ptr<MultiBandMap2DCPUEle> ele(uint idx)
        {
            std::shared_lock<std::shared_mutex> lock(mutexData);
            if(idx>_data.size()) return std::shared_ptr<MultiBandMap2DCPUEle>();
            else if(!_data[idx].get())
            {
                _data[idx]=std::shared_ptr<MultiBandMap2DCPUEle>(new MultiBandMap2DCPUEle());
            }
            return _data[idx];
        }

    private:
        //IMPORTANT: everything should never changed after prepared!
        double      _eleSize,_lengthPixel,_eleSizeInv,_lengthPixelInv;
        cv::Point3d _gpsOrigin;
        cv::Point3d _max,_min;
        int         _w,_h;
        std::vector<std::shared_ptr<MultiBandMap2DCPUEle> >  _data;
        std::shared_mutex mutexData;
    };
public:

    MultiBandMap2DCPU(bool thread=true);

    virtual ~MultiBandMap2DCPU(){_valid=false;}

    virtual bool prepare(const Sophus::SE3d& plane,const PinHoleParameters& camera,
                    const std::deque<std::pair<cv::Mat,Sophus::SE3d> >& frames);

    virtual bool feed(cv::Mat img,const Sophus::SE3d& pose);//world coordinate

    virtual bool feed( FusionFrame& frame);

    //virtual void draw();

    virtual void draw2D();

    virtual bool save(const std::string& filename);

    virtual bool save_1(const std::string& filename);

    virtual uint queueSize(){
        if(prepared.get()) return prepared->queueSize();
        else               return 0;
    }
    virtual void run();
    virtual void RenderTime2File();

    std::thread run_thread;
    
    vector<double> vdrender_ms;

    static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
    static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
    
    double zoom_scale = 1.0;
    double pan_offset_x = 0.0;
    double pan_offset_y = 0.0;

    bool g_request_save = false;
    void saveCurrentFramebuffer(const std::string& filename, int width, int height);


    

private:
    void ensureTilePublisherStarted();
    void publishTileIfNeeded(
        int tileX,
        int tileY,
        const std::shared_ptr<MultiBandMap2DCPUEle>& ele,
        const std::vector<std::shared_ptr<MultiBandMap2DCPUEle> >& neighbors,
        const std::shared_ptr<MultiBandMap2DCPUData>& d);

    bool getFrame(std::pair<cv::Mat,Sophus::SE3d>& frame);
    
    bool renderFrame(const std::pair<cv::Mat,Sophus::SE3d>& frame);
    bool renderFrame(const FusionFrame& frame);
    bool spreadMap(double xmin,double ymin,double xmax,double ymax);

    double cached_max_dis = -1.0; // -1 表示未初始化

    //source
    shared_ptr<MultiBandMap2DCPUPrepare>             prepared;
    shared_ptr<MultiBandMap2DCPUData>                data;
    shared_mutex                       mutex;

    bool                              _valid,_thread,_changed;
    cv::Mat                           weightImage;
    int                               alpha,_bandNum,_highQualityShow;
    std::unique_ptr<TilePublisher>    tilePublisher;
    bool                              tilePublisherStarted = false;

    
    
};

#endif
