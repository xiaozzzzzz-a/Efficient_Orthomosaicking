#include "Map2D.h"
#include <iostream>
#include "Map2DCPU.h"
#include "MultiBandMap2DCPU.h"
using namespace std;

bool Map2DPrepare::prepare(const Sophus::SE3d& plane,const PinHoleParameters& camera,
                                        const std::deque<std::pair<cv::Mat,Sophus::SE3d> >& frames)
{
    if(frames.size() == 0||camera.w<=0||camera.h<=0||camera.fx==0||camera.fy==0)
    {
        cerr<<"Map2D::prepare:Not valid prepare!\n";
        return false;
    }
    _camera=camera;_fxinv=1./camera.fx;_fyinv=1./camera.fy;
    _plane =plane;
    _frames=frames;
    for(std::deque<std::pair<cv::Mat,Sophus::SE3d> >::iterator it=_frames.begin();it!=_frames.end();it++)
    {
        Sophus::SE3d& pose=it->second;
        pose=plane.inverse()*pose;//plane coordinate
    }
    return true;
}

std::shared_ptr<Map2D> Map2D::create(int type, bool thread )
{
    if(type == NoType) return shared_ptr<Map2D>();
    else if(type == TypeCPU) return shared_ptr<Map2D>(new Map2DCPU(thread));
    else if(type == TypeMultiBandCPU) return shared_ptr<MultiBandMap2DCPU>(new MultiBandMap2DCPU(thread));

}