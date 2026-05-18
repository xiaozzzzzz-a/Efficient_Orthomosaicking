/******************************************************************************

  This file is part of Map2DFusion.

  Copyright 2016 (c)  Yong Zhao <zd5945@126.com> http://www.zhaoyong.adv-ci.com

  ----------------------------------------------------------------------------

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.

*******************************************************************************/
#include "MultiBandMap2DCPU.h"

#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/stitching/detail/blenders.hpp>
#include <unistd.h>
#include <chrono> 
#include <mutex>
#include "type.h"
#include <iomanip>
#include <fstream>
#include <Eigen/Core>
#include <thread>
#include <GL/glut.h> 
#include <GLFW/glfw3.h>
#include <cstdlib>
#include "LocalMapping.h"
using namespace ORB_SLAM3;

using namespace std;

/**

  __________max
  |    |    |
  |____|____|
  |    |    |
  |____|____|
 min
 */

MultiBandMap2DCPU::MultiBandMap2DCPUEle::~MultiBandMap2DCPUEle()
{
    //if(texName) pi::gl::Signal_Handle::instance().delete_texture(texName);
}

bool MultiBandMap2DCPU::MultiBandMap2DCPUEle::normalizeUsingWeightMap(const cv::Mat& weight, cv::Mat& src)
{
    if(!(src.type()==CV_32FC3&&weight.type()==CV_32FC1)) return false;
    cv::Point3f* srcP=(cv::Point3f*)src.data;
    float*    weightP=(float*)weight.data;
    for(float* Pend=weightP+weight.cols*weight.rows;weightP!=Pend;weightP++,srcP++)
        *srcP=(*srcP)/(*weightP+1e-5);
    return true;
}

bool MultiBandMap2DCPU::MultiBandMap2DCPUEle::mulWeightMap(const cv::Mat& weight, cv::Mat& src)
{
    if(!(src.type()==CV_32FC3&&weight.type()==CV_32FC1)) return false;
    cv::Point3f* srcP=(cv::Point3f*)src.data;
    float*    weightP=(float*)weight.data;
    for(float* Pend=weightP+weight.cols*weight.rows;weightP!=Pend;weightP++,srcP++)
        *srcP=(*srcP)*(*weightP);
    return true;
}

void convertMatToUMat(const std::vector<cv::Mat>&src, std::vector<cv::UMat> &dst){
    dst.clear();
    for (const auto& mat : src) {
        cv::UMat umat;
        mat.copyTo(umat);  // 将 Mat 转换为 UMat
        dst.push_back(umat);
    }
}

void convertUMatToMat(const std::vector<cv::UMat>& src, std::vector<cv::Mat>& dst) {
    dst.clear();  // 清空目标 vector

    // 遍历源 vector 中的每一个 UMat 对象
    for (const auto& umat : src) {
        cv::Mat mat;
        umat.copyTo(mat);  // 将 UMat 转换为 Mat
        dst.push_back(mat);  // 将转换后的 Mat 对象添加到目标 vector 中
    }
}

cv::Mat MultiBandMap2DCPU::MultiBandMap2DCPUEle::blend(const std::vector<shared_ptr<MultiBandMap2DCPUEle> >& neighbors)
{
    if(!pyr_laplace.size()) return cv::Mat();
    if(neighbors.size()==9)
    {
        //blend with neighbors, this obtains better visualization
        int flag=0;
        for(int i=0;i<neighbors.size();i++)
        {
            flag<<=1;
            if(neighbors[i].get()&&neighbors[i]->pyr_laplace.size())
                flag|=1;
        }
        switch (flag) {
        case 0X01FF:
        {
            vector<cv::Mat> pyr_laplaceClone(pyr_laplace.size());
            for(int i=0;i<pyr_laplace.size();i++)
            {
                int borderSize=1<<(pyr_laplace.size()-i-1);
                int srcrows=pyr_laplace[i].rows;
                int dstrows=srcrows+(borderSize<<1);
                pyr_laplaceClone[i]=cv::Mat(dstrows,dstrows,pyr_laplace[i].type());

                for(int y=0;y<3;y++)
                    for(int x=0;x<3;x++)
                {
                    const shared_ptr<MultiBandMap2DCPUEle>& ele=neighbors[3*y+x];
                    std::shared_lock<std::shared_mutex> lock(ele->mutexData);
                    if(ele->pyr_laplace[i].empty())
                        continue;
                    cv::Rect      src,dst;
                    src.width =dst.width =(x==1)?srcrows:borderSize;
                    src.height=dst.height=(y==1)?srcrows:borderSize;
                    src.x=(x==0)?(srcrows-borderSize):0;
                    src.y=(y==0)?(srcrows-borderSize):0;
                    dst.x=(x==0)?0:((x==1)?borderSize:(dstrows-borderSize));
                    dst.y=(y==0)?0:((y==1)?borderSize:(dstrows-borderSize));
                    ele->pyr_laplace[i](src).copyTo(pyr_laplaceClone[i](dst));
                }
            }
            std::vector<cv::UMat> pyr_laplaceUMat;
            convertMatToUMat(pyr_laplaceClone, pyr_laplaceUMat);
            cv::detail::restoreImageFromLaplacePyr(pyr_laplaceUMat);
            convertUMatToMat(pyr_laplaceUMat, pyr_laplaceClone);

            {
                cv::Mat result;
                int borderSize=1<<(pyr_laplace.size()-1);
                pyr_laplaceClone[0](cv::Rect(borderSize,borderSize,ELE_PIXELS,ELE_PIXELS)).copyTo(result);
                return  result.setTo(cv::Scalar::all(0),weights[0]==0);
            }
        }
            break;
        default:
            break;
        }
    }

    {
        //blend by self
        vector<cv::Mat> pyr_laplaceClone(pyr_laplace.size());
        for(int i=0;i<pyr_laplace.size();i++)
        {
            pyr_laplaceClone[i]=pyr_laplace[i].clone();
        }

        std::vector<cv::UMat> pyr_laplaceUMat;
        convertMatToUMat(pyr_laplaceClone, pyr_laplaceUMat);
        cv::detail::restoreImageFromLaplacePyr(pyr_laplaceUMat);
        convertUMatToMat(pyr_laplaceUMat, pyr_laplaceClone);

        return  pyr_laplaceClone[0].setTo(cv::Scalar::all(0),weights[0]==0);
    }
}

// this is a bad idea, just for test
bool MultiBandMap2DCPU::MultiBandMap2DCPUEle::updateTexture(const std::vector<shared_ptr<MultiBandMap2DCPUEle> >& neighbors)
{
    cv::Mat tmp=blend(neighbors);
    uint type=0;
    if(tmp.empty()) return false;
    else if(tmp.type()==CV_16SC3)
    {
        tmp.convertTo(tmp,CV_8UC3);
        type=GL_UNSIGNED_BYTE;
    }
    else if(tmp.type()==CV_32FC3)
        type=GL_FLOAT;
    if(!type) return false;

    {
        if(texName==0)// create texture
        {
            glGenTextures(1, &texName);
            glBindTexture(GL_TEXTURE_2D,texName);
            glTexImage2D(GL_TEXTURE_2D, 0,
                         GL_RGB,tmp.cols,tmp.rows, 0,
                         GL_BGR, type,tmp.data);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,GL_NEAREST);
        }
        else
        {
            glBindTexture(GL_TEXTURE_2D,texName);
            glTexImage2D(GL_TEXTURE_2D, 0,
                         GL_RGB,tmp.cols,tmp.rows, 0,
                         GL_BGR, type,tmp.data);
        }
    }

    // SvarWithType<cv::Mat>::instance()["LastTexMat"]=tmp;
    // SvarWithType<cv::Mat>::instance()["LastTexMatWeight"]=weights[0].clone();

    Ischanged=false;
    return true;
}

MultiBandMap2DCPU::MultiBandMap2DCPUData::MultiBandMap2DCPUData(double eleSize_,double lengthPixel_,cv::Point3d max_,cv::Point3d min_,
             int w_,int h_,const std::vector<shared_ptr<MultiBandMap2DCPUEle> >& d_)
    :_eleSize(eleSize_),_eleSizeInv(1./eleSize_),
      _lengthPixel(lengthPixel_),_lengthPixelInv(1./lengthPixel_),
      _min(min_),_max(max_),_w(w_),_h(h_),_data(d_)
{
    //_gpsOrigin=svar.get_var("GPS.Origin",_gpsOrigin);
}

bool MultiBandMap2DCPU::MultiBandMap2DCPUData::prepare(shared_ptr<MultiBandMap2DCPUPrepare> prepared)
{
    if(_w||_h) return false;//already prepared
    {
        _max=cv::Point3d(-1e10,-1e10,-1e10);
        _min=-_max;
        for(std::deque<std::pair<cv::Mat,Sophus::SE3d> >::iterator it=prepared->_frames.begin();
            it!=prepared->_frames.end();it++)
        {
            Sophus::SE3d& pose=it->second;
            Eigen::Vector3d translation = pose.translation();
            cv::Point3d cv_translation(translation.x(), translation.y(), translation.z());
            cv::Point3d& t=cv_translation;
            _max.x=t.x>_max.x?t.x:_max.x;
            _max.y=t.y>_max.y?t.y:_max.y;
            _max.z=t.z>_max.z?t.z:_max.z;
            _min.x=t.x<_min.x?t.x:_min.x;
            _min.y=t.y<_min.y?t.y:_min.y;
            _min.z=t.z<_min.z?t.z:_min.z;
        }
        if(_min.z*_max.z<=0) return false;
        cout<<"Box:Min:"<<_min<<",Max:"<<_max<<endl;
    }
    //estimate w,h and bonding box
    {
        double maxh;
        if(_max.z>0) maxh=_max.z;
        else maxh=-_min.z;
        cv::Point3d line=prepared->UnProject(cv::Point2d(prepared->_camera.w,prepared->_camera.h))
                -prepared->UnProject(cv::Point2d(0,0));
        double radius=0.5*maxh*sqrt((line.x*line.x+line.y*line.y));


        //_lengthPixel=svar.GetDouble("Map2D.Resolution",0);
        _lengthPixel = 0;
        if(!_lengthPixel)
        {
            cout<<"Auto resolution from max height "<<maxh<<"m.\n";
            _lengthPixel=2*radius/sqrt(prepared->_camera.w*prepared->_camera.w
                                       +prepared->_camera.h*prepared->_camera.h);

            //_lengthPixel/=svar.GetDouble("Map2D.Scale",1);
            _lengthPixel /= 1;
        }
        cout<<"Map2D.Resolution="<<_lengthPixel<<endl;
        _lengthPixelInv=1./_lengthPixel;
        _min=_min-cv::Point3d(radius,radius,0);
        _max=_max+cv::Point3d(radius,radius,0);
        cv::Point3d center=0.5*(_min+_max);
        _min=2*_min-center;_max=2*_max-center;
        _eleSize=ELE_PIXELS*_lengthPixel;
        _eleSizeInv=1./_eleSize;
        {
            _w=ceil((_max.x-_min.x)/_eleSize);
            _h=ceil((_max.y-_min.y)/_eleSize);
            _max.x=_min.x+_eleSize*_w;
            _max.y=_min.y+_eleSize*_h;
            _data.resize(_w*_h);
        }
    }
    //_gpsOrigin=svar.get_var("GPS.Origin",_gpsOrigin);
    return true;
}

MultiBandMap2DCPU::MultiBandMap2DCPU(bool thread)
    :alpha(0),
     _valid(false),_thread(thread),
     _bandNum(5),
     _highQualityShow(1)
{
    _bandNum=min(_bandNum, static_cast<int>(ceil(log(ELE_PIXELS) / log(2.0))));

    TilePublisher::Options options;
    if (const char* host = std::getenv("MAP2D_TILE_HOST")) {
        options.host = host;
    }
    if (const char* port = std::getenv("MAP2D_TILE_PORT")) {
        options.port = std::atoi(port);
    }
    if (const char* taskId = std::getenv("MAP2D_TILE_TASK")) {
        options.taskId = taskId;
    }
    tilePublisher.reset(new TilePublisher(options));
}

void MultiBandMap2DCPU::ensureTilePublisherStarted()
{
    if (!tilePublisher || tilePublisherStarted) {
        return;
    }
    tilePublisher->start();
    tilePublisherStarted = true;
}

void MultiBandMap2DCPU::publishTileIfNeeded(
    int tileX,
    int tileY,
    const std::shared_ptr<MultiBandMap2DCPUEle>& ele,
    const std::vector<std::shared_ptr<MultiBandMap2DCPUEle> >& neighbors,
    const std::shared_ptr<MultiBandMap2DCPUData>& d)
{
    if (!tilePublisher || !ele) {
        return;
    }

    cv::Mat tileImage = ele->blend(neighbors);
    if (tileImage.empty()) {
        return;
    }
    if (tileImage.type() == CV_16SC3) {
        tileImage.convertTo(tileImage, CV_8UC3);
    } else if (tileImage.type() == CV_32FC3) {
        tileImage.convertTo(tileImage, CV_8UC3, 255.0);
    } else if (tileImage.type() != CV_8UC3 && tileImage.type() != CV_8UC4) {
        return;
    }

    TilePublisher::TileJob job;
    job.level = tilePublisher->options().level;
    job.x = tileX;
    job.y = tileY;
    job.worldMinX = d->min().x + tileX * d->eleSize();
    job.worldMinY = d->min().y + tileY * d->eleSize();
    job.metersPerPixel = d->lengthPixel();
    job.image = tileImage;
    tilePublisher->enqueue(std::move(job));
}

bool MultiBandMap2DCPU::prepare(const Sophus::SE3d& plane,const PinHoleParameters& camera,
                const std::deque<std::pair<cv::Mat,Sophus::SE3d> >& frames)
{
    //insert frames
    shared_ptr<MultiBandMap2DCPUPrepare> p(new MultiBandMap2DCPUPrepare);
    shared_ptr<MultiBandMap2DCPUData>    d(new MultiBandMap2DCPUData);

    if(p->prepare(plane,camera,frames))
        if(d->prepare(p))
        {
            std::unique_lock<std::shared_mutex> lock(mutex);
            prepared=p;
            data=d;
            weightImage.release();
            cout<<"create new thread MultiBandMap2DCPU::run"<<endl;
            run_thread = std::thread(&MultiBandMap2DCPU::run, this);

            _valid=true;
            return true;
        }
    return false;
}


bool MultiBandMap2DCPU::feed(FusionFrame& frame)
{
    if(!_valid) return false;
    shared_ptr<MultiBandMap2DCPUPrepare> p;
    shared_ptr<MultiBandMap2DCPUData>    d;
    {
        std::shared_lock<std::shared_mutex> lock(mutex);
        p=prepared;d=data;
    }
    
    
    std::pair<cv::Mat,Sophus::SE3d> frame1(frame.image,p->_plane.inverse()*frame.pose);

    frame.pose = p->_plane.inverse()*frame.pose;
    if(_thread)
    {
        std::unique_lock lock(p->mutexFrames);
        p->_frames.push_back(frame1);
        if(p->_frames.size()>20) p->_frames.pop_front();
        return true;
    }
    else
    {
        return renderFrame(frame);
    }

}


bool MultiBandMap2DCPU::feed(cv::Mat img,const Sophus::SE3d& pose)
{
    if(!_valid) return false;
    shared_ptr<MultiBandMap2DCPUPrepare> p;
    shared_ptr<MultiBandMap2DCPUData>    d;
    {
        std::shared_lock<std::shared_mutex> lock(mutex);
        p=prepared;d=data;
    }
    std::pair<cv::Mat,Sophus::SE3d> frame(img,p->_plane.inverse()*pose);
    if(_thread)
    {
        std::unique_lock lock(p->mutexFrames);
        p->_frames.push_back(frame);
        if(p->_frames.size()>20) p->_frames.pop_front();
        cout<<"feedd1dd11111"<<endl;
        return true;
    }
    else
    {
        return renderFrame(frame);
    }

}


void saveSE3ToFile(const std::string& filename, const Sophus::SE3d& pose) {
    std::ofstream file(filename, std::ios::app);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Cannot open file for writing: " << filename << std::endl;
        return;
    }

    // 提取平移
    Eigen::Vector3d t = pose.translation();

    // 提取四元数
    Eigen::Quaterniond q = pose.unit_quaternion();

    
    file << t.x() << " " << t.y() << " " << t.z() << " ";
    file << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;

    file.close();
    std::cout << "[INFO] SE3 pose saved to: " << filename << std::endl;
}
void writeCameraPoseToFile(const std::string& filename, const Eigen::Vector3d& cam_pos) {
    std::ofstream ofs(filename, std::ios::app);  // 以追加模式写入
    if (!ofs.is_open()) {
        std::cerr << "[ERROR] Could not open file for writing: " << filename << std::endl;
        return;
    }
    ofs << cam_pos.x() << " " << cam_pos.y() << " " << cam_pos.z() << std::endl;
    ofs.close();
}

void applySeamCutBlending(
    cv::Point3f* srcL, float* srcW,
    cv::Point3f* dstL, float* dstW,
    int width, int height, int skip)
{
    // 1. 构建差值代价图
    cv::Mat costMap(height, width, CV_32F);
    for(int y=0; y<height; y++) {
        for(int x=0; x<width; x++) {
            costMap.at<float>(y,x) = std::abs(srcW[y*width + x] - dstW[y*width + x]);
        }
    }

    // 2. 动态规划寻找纵向最短路径
    cv::Mat dp = costMap.clone();
    for(int y=1; y<height; y++) {
        for(int x=0; x<width; x++) {
            float prev = dp.at<float>(y-1,x);
            if(x>0) prev = std::min(prev, dp.at<float>(y-1,x-1));
            if(x<width-1) prev = std::min(prev, dp.at<float>(y-1,x+1));
            dp.at<float>(y,x) += prev;
        }
    }

    // 3. 回溯 seam
    std::vector<int> seamPath(height);
    seamPath[height-1] = 0;
    float minCost = dp.at<float>(height-1,0);
    for(int x=1; x<width; x++) {
        if(dp.at<float>(height-1,x) < minCost) {
            minCost = dp.at<float>(height-1,x);
            seamPath[height-1] = x;
        }
    }
    for(int y=height-2; y>=0; y--) {
        int prevX = seamPath[y+1];
        int bestX = prevX;
        float bestVal = dp.at<float>(y,prevX);
        if(prevX>0 && dp.at<float>(y,prevX-1)<bestVal) {
            bestX = prevX-1; bestVal = dp.at<float>(y,prevX-1);
        }
        if(prevX<width-1 && dp.at<float>(y,prevX+1)<bestVal) {
            bestX = prevX+1; bestVal = dp.at<float>(y,prevX+1);
        }
        seamPath[y] = bestX;
    }

    // 4. 根据 seam 生成 mask
    cv::Mat mask(height, width, CV_32F, cv::Scalar(0));
    for(int y=0; y<height; y++) {
        for(int x=0; x<width; x++) {
            if(x < seamPath[y]) mask.at<float>(y,x) = 1.0;
            else mask.at<float>(y,x) = 0.0;
        }
    }

    // 5. 最后应用融合
    for(int y=0; y<height; y++, srcL+=skip, srcW+=skip, dstL+=skip, dstW+=skip) {
        for(int x=0; x<width; x++, srcL++, dstL++, srcW++, dstW++) {
            float m = mask.at<float>(y,x);
            *dstL = (*srcL) * m + (*dstL) * (1.0f - m);
            *dstW = std::max(*dstW, *srcW);
        }
    }
}

void applySeamCutBlending_INT16(
    cv::Point3_<short>* srcL, float* srcW,
    cv::Point3_<short>* dstL, float* dstW,
    int width, int height, int skip)
{
    printf("[DEBUG] Enter applySeamCutBlending_INT16\n");
    printf("[DEBUG] width = %d, height = %d, skip = %d\n", width, height, skip);

    // 1. 构建 costMap：颜色差
    cv::Mat costMap(height, width, CV_32F);
    for(int y=0; y<height; y++) {
        for(int x=0; x<width; x++) {
            float diffB = static_cast<float>(srcL[y*width + x].x) - static_cast<float>(dstL[y*width + x].x);
            float diffG = static_cast<float>(srcL[y*width + x].y) - static_cast<float>(dstL[y*width + x].y);
            float diffR = static_cast<float>(srcL[y*width + x].z) - static_cast<float>(dstL[y*width + x].z);

            costMap.at<float>(y,x) = std::abs(diffB) + std::abs(diffG) + std::abs(diffR);
        }
    }

    double minCost, maxCost;
    cv::minMaxLoc(costMap, &minCost, &maxCost);
    printf("[DEBUG] costMap min=%.6f max=%.6f\n", minCost, maxCost);

    // 归一化 costMap 避免层间爆炸
    if (maxCost > 0.0f) {
        costMap /= (maxCost + 1e-5f);
    }

    // 2. 动态规划
    cv::Mat dp = costMap.clone();
    for(int y=1; y<height; y++) {
        for(int x=0; x<width; x++) {
            float prev = dp.at<float>(y-1,x);
            if(x>0) prev = std::min(prev, dp.at<float>(y-1,x-1));
            if(x<width-1) prev = std::min(prev, dp.at<float>(y-1,x+1));
            dp.at<float>(y,x) += prev;
        }
    }
    double dpMin, dpMax;
    cv::minMaxLoc(dp, &dpMin, &dpMax);
    printf("[DEBUG] dp min=%.6f max=%.6f\n", dpMin, dpMax);

    // 3. 回溯 seam
    std::vector<int> seamPath(height);
    seamPath[height-1] = 0;
    float minVal = dp.at<float>(height-1,0);
    for(int x=1; x<width; x++) {
        if(dp.at<float>(height-1,x) < minVal) {
            minVal = dp.at<float>(height-1,x);
            seamPath[height-1] = x;
        }
    }
    for(int y=height-2; y>=0; y--) {
        int prevX = seamPath[y+1];
        int bestX = prevX;
        float bestVal = dp.at<float>(y,prevX);
        if(prevX>0 && dp.at<float>(y,prevX-1)<bestVal) {
            bestX = prevX-1;
            bestVal = dp.at<float>(y,prevX-1);
        }
        if(prevX<width-1 && dp.at<float>(y,prevX+1)<bestVal) {
            bestX = prevX+1;
            bestVal = dp.at<float>(y,prevX+1);
        }
        seamPath[y] = bestX;
    }

    printf("[DEBUG] seamPath sample: ");
    for(int i=0; i<std::min(10, height); i++) {
        printf("%d ", seamPath[i]);
    }
    printf("...\n");

    // 4. 生成软 mask
    cv::Mat mask(height, width, CV_32F, cv::Scalar(0));
    for(int y=0; y<height; y++) {
        for(int x=0; x<width; x++) {
            // Soft edge instead of hard 0/1
            if(x < seamPath[y]) mask.at<float>(y,x) = 0.9f;
            else mask.at<float>(y,x) = 0.1f;
        }
    }
    printf("[DEBUG] Building mask\n");

    // 5. 融合
    printf("[DEBUG] Blending\n");
    for(int y=0; y<height; y++, srcL+=skip, srcW+=skip, dstL+=skip, dstW+=skip) {
        for(int x=0; x<width; x++, srcL++, dstL++, srcW++, dstW++) {
            float m = mask.at<float>(y,x);
            if (x < 3 && y < 3) {
                printf("[DEBUG] Blend y=%d x=%d m=%.6f\n", y, x, m);
            }

            float srcB = static_cast<float>(srcL->x);
            float srcG = static_cast<float>(srcL->y);
            float srcR = static_cast<float>(srcL->z);

            float dstB = static_cast<float>(dstL->x);
            float dstG = static_cast<float>(dstL->y);
            float dstR = static_cast<float>(dstL->z);

            dstL->x = static_cast<short>(srcB * m + dstB * (1.0f - m) + 0.5f);
            dstL->y = static_cast<short>(srcG * m + dstG * (1.0f - m) + 0.5f);
            dstL->z = static_cast<short>(srcR * m + dstR * (1.0f - m) + 0.5f);

            *dstW = std::max(*dstW, *srcW);
        }
    }

    printf("[DEBUG] applySeamCutBlending_INT16 done\n");
}


bool MultiBandMap2DCPU::renderFrame(const std::pair<cv::Mat, Sophus::SE3d>& frame)
{

    std::chrono::steady_clock::time_point time_renderbegin = std::chrono::steady_clock::now();
    shared_ptr<MultiBandMap2DCPUPrepare> p;
    shared_ptr<MultiBandMap2DCPUData>    d;
    {
        std::shared_lock<shared_mutex> lock(mutex);
        p=prepared;d=data;
    }
    if(frame.first.cols!=p->_camera.w||frame.first.rows!=p->_camera.h||frame.first.type()!=CV_8UC3)
    {
        cerr<<"MultiBandMap2DCPU::renderFrame: frame.first.cols!=p->_camera.w||frame.first.rows!=p->_camera.h||frame.first.type()!=CV_8UC3\n";

        std::cerr << "Map2DCPU::renderFrame: 输入图像参数不匹配！\n";
    
    std::cerr << "frame.first.cols = " << frame.first.cols 
              << ", expected = " << p->_camera.w << std::endl;
    
    std::cerr << "frame.first.rows = " << frame.first.rows 
              << ", expected = " << p->_camera.h << std::endl;

    std::cerr << "frame.first.type = " << frame.first.type() 
              << ", expected = " << CV_8UC3 
              << " (即 CV_8UC3 = " << CV_8UC3 << ")" << std::endl;
        return false;
    }
    // 1. pose->pts
    std::vector<cv::Point2d> imgPts;
    {
        imgPts.reserve(4);
        imgPts.push_back(cv::Point2d(0,0));
        imgPts.push_back(cv::Point2d(p->_camera.w,0));
        imgPts.push_back(cv::Point2d(0,p->_camera.h));
        imgPts.push_back(cv::Point2d(p->_camera.w,p->_camera.h));
    }
    vector<cv::Point2d> pts;
    pts.reserve(imgPts.size());
    cv::Point3d downLook(0,0,-1);
    if(frame.second.translation().z()<0) downLook=cv::Point3d(0,0,1);
    for(int i=0;i<imgPts.size();i++)
    {
        //pi::Point3d axis=frame.second.get_rotation()*p->UnProject(imgPts[i]);
        cv::Point3d point3d = p->UnProject(imgPts[i]);//将像素坐标转换为归一化相机坐标，变为相机光心到该像素的射线。

        cout<<point3d<<endl;
        Eigen::Vector3d axis_e=frame.second.so3() *Eigen::Vector3d(point3d.x, point3d.y, point3d.z);//将该方向转换至世界坐标系上
    
        
        cv::Point3d axis(axis_e.x(), axis_e.y(), axis_e.z());

        double dot_product = axis.dot(downLook);
        std::cout << "[DEBUG] Dot product with downLook = " << dot_product << std::endl;

        if(axis.dot(downLook)<0.4)//downLook = (0,0,-1) 或 (0,0,1)，取决于相机高度的正负方向。如果小于 0.4，表示与地面法向接近 90°，投影会发散或不稳定。
        {
            return false;
        }
        axis_e=frame.second.translation()
                -axis_e*(frame.second.translation().z()/axis_e.z());

        Eigen::Vector3d cam_pos = frame.second.translation();
        std::cout << "[INFO] Camera Height (Z): " << cam_pos.z() << " m" << std::endl;

        cv::Point3d axis1(axis_e.x(), axis_e.y(), axis_e.z());  
        pts.push_back(cv::Point2d(axis1.x,axis1.y));
    }
    // 2. dest location?
    double xmin=pts[0].x;
    double xmax=xmin;
    double ymin=pts[0].y;
    double ymax=ymin;
   
    for(int i=1;i<pts.size();i++)
    {
        if(pts[i].x<xmin) xmin=pts[i].x;
        if(pts[i].y<ymin) ymin=pts[i].y;
        if(pts[i].x>xmax) xmax=pts[i].x;
        if(pts[i].y>ymax) ymax=pts[i].y;
    }

    //std::cout<<"xmin11: "<<xmin<<" "<<xmax<<" "<<ymin<<" "<<ymax<<endl;
    if(xmin<d->min().x||xmax>d->max().x||ymin<d->min().y||ymax>d->max().y)
    {
        if(p!=prepared)//what if prepare called?
        {
            return false;
        }
        if(!spreadMap(xmin,ymin,xmax,ymax))
        {
            return false;
        }
        else
        {
            std::shared_lock<shared_mutex> lock(mutex);
            if(p!=prepared)//what if prepare called?
            {
                return false;
            }
            d=data;//new data
        }
    }
    int xminInt=floor((xmin-d->min().x)*d->eleSizeInv());
    int yminInt=floor((ymin-d->min().y)*d->eleSizeInv());
    int xmaxInt= ceil((xmax-d->min().x)*d->eleSizeInv());
    int ymaxInt= ceil((ymax-d->min().y)*d->eleSizeInv());
    if(xminInt<0||yminInt<0||xmaxInt>d->w()||ymaxInt>d->h()||xminInt>=xmaxInt||yminInt>=ymaxInt)
    {
        cerr<<"MultiBandMap2DCPU::renderFrame:should never happen!\n";
        return false;
    }
    {
        xmin=d->min().x+d->eleSize()*xminInt;
        ymin=d->min().y+d->eleSize()*yminInt;
        xmax=d->min().x+d->eleSize()*xmaxInt;
        ymax=d->min().y+d->eleSize()*ymaxInt;
    }
    // 3.prepare weight and warp images
    cv::Mat weight_src;
    if(1)
    {

        auto t_start = std::chrono::high_resolution_clock::now();
        std::shared_lock<shared_mutex> lock(mutex);


        int w=frame.first.cols;
        int h=frame.first.rows;

        weightImage.create(h,w,CV_32FC1);
        float* p = (float*)weightImage.data;
        double fx = prepared->_camera.fx;
        double fy = prepared->_camera.fy;
        double cx = prepared->_camera.cx;
        double cy = prepared->_camera.cy;

        // 图像中心方向向量
        Eigen::Vector3d center_dir = Eigen::Vector3d((0.0 - cx) / fx, (0.0 - cy) / fy, 1.0).normalized();
        std::cout << "[Debug] center_dir (unit vector): " << center_dir.transpose() << std::endl;

        // 地图分辨率（米/像素）
        double r = data->lengthPixel();
        std::cout << "[Debug] map resolution r: " << r << " m/pixel" << std::endl;

        // 相机光轴方向（相对于世界坐标系）
        Eigen::Vector3d cam_z = frame.second.so3() * Eigen::Vector3d(0, 0, 1);
        std::cout << "[Debug] camera direction (cam_z): " << cam_z.transpose() << std::endl;

        // 地面法线方向
        Eigen::Vector3d normal_plane = prepared->_plane.so3() * Eigen::Vector3d(0, 0, 1);
        std::cout << "[Debug] ground normal (from plane): " << normal_plane.transpose() << std::endl;

        // 相机高度

        Eigen::Vector3d t_cam = frame.second.translation();
        Eigen::Vector3d t_plane = prepared->_plane.translation();

        // 垂直距离 = 法线 · (相机 - 地面)
        double Alt = std::abs(normal_plane.dot(t_cam - t_plane));
        //std::cout << "[Debug] Correct Alt (camera to plane): " << Alt << " meters" << std::endl;

        // 相机与地面法线夹角（弧度）
        double cos_theta = cam_z.dot(normal_plane);
        cos_theta = std::clamp(cos_theta, -1.0, 1.0);
        double theta = std::acos(cos_theta);
        std::cout << "[Debug] cos(theta): " << cos_theta << ", theta (rad): " << theta
                << ", theta (deg): " << theta * 180.0 / M_PI << std::endl;

        // 图像对角线一半 Dm（单位：像素）
        double Dm = 0.5 * std::sqrt(w * w + h * h);
        std::cout << "[Debug] Dm (half-diagonal in pixels): " << Dm << std::endl;

        // 计算全局正交性权重项
        double global_term = 1.0 - (Alt * std::tan(theta)) / (r * Dm);
        global_term = std::clamp(global_term, 0.0, 1.0);
        std::cout << "[Debug] global_term (orthogonality weight): " << global_term << std::endl;

        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        std::cout << "[Timing] Orthogonality weight calculation time: " << elapsed_ms << " ms" << std::endl;

        double gamma = 2.0;
        

        if (cached_max_dis < 0) {
            double max_dis = 0.0;
            for (int v = 0; v < h; ++v) {
                for (int u = 0; u < w; ++u) {
                    double dx = (u - cx) / fx;
                    double dy = (v - cy) / fy;
                    double d = std::sqrt(dx * dx + dy * dy);
                    if (d > max_dis) max_dis = d;
                }
            }
            cached_max_dis = max_dis;
        }

        
        for (int v = 0; v < h; ++v) {
            for (int u = 0; u < w; ++u) {
                double dx = (u - cx) / fx;
                double dy = (v - cy) / fy;
                double d = std::sqrt(dx * dx + dy * dy);
                float wgt = std::pow(1.0 - d / cached_max_dis, gamma);
                *p = std::max((wgt)*(float)global_term, 1e-5f);
                p++;
            }
        }
        weight_src=weightImage.clone();



    }
    else
    {
        std::shared_lock<shared_mutex> lock(mutex);
        weight_src=weightImage.clone();
    }

    std::vector<cv::Point2f>  imgPtsCV;
    {
        imgPtsCV.reserve(imgPts.size());
        for(int i=0;i<imgPts.size();i++)
            imgPtsCV.push_back(cv::Point2f(imgPts[i].x,imgPts[i].y));
    }
    std::vector<cv::Point2f> destPoints;
    destPoints.reserve(imgPtsCV.size());
    for(int i=0;i<imgPtsCV.size();i++)
    {
        destPoints.push_back(cv::Point2f((pts[i].x-xmin)*d->lengthPixelInv(),
                             (pts[i].y-ymin)*d->lengthPixelInv()));
    }

    cv::Mat transmtx = cv::getPerspectiveTransform(imgPtsCV, destPoints);

    cv::Mat img_src;
    //if(svar.GetInt("MultiBandMap2DCPU.ForceFloat",0))
    if(0)
        frame.first.convertTo(img_src,CV_32FC3,1./255.);
    else
        frame.first.convertTo(img_src,CV_16SC3);

    cv::Mat weight_warped((ymaxInt-yminInt)*ELE_PIXELS,(xmaxInt-xminInt)*ELE_PIXELS,CV_32FC1);
    cv::Mat image_warped((ymaxInt-yminInt)*ELE_PIXELS,(xmaxInt-xminInt)*ELE_PIXELS,img_src.type());
    cv::warpPerspective(img_src, image_warped, transmtx, image_warped.size(),cv::INTER_LINEAR,cv::BORDER_REFLECT);
    //cv::warpPerspective(img_src, image_warped, transmtx, image_warped.size(),cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));

    cv::warpPerspective(weight_src, weight_warped, transmtx, weight_warped.size(),cv::INTER_NEAREST);

    if(0)//showed wraped!
    {
        cv::imshow("image_warped",image_warped);
        cv::imshow("weight_warped",weight_warped);
        if(0)//Save  ImageWarped!
        {
            cout<<"Saving warped image.\n";
            cv::imwrite("image_warped.png",image_warped);
            cv::imwrite("weight_warped.png",weight_warped);
        }
        cv::waitKey(0);
    }

    // 4. blender dst to eles
    std::vector<cv::Mat> pyr_laplace;
    std::vector<cv::UMat> pyr_laplaceUMat;
    
    cv::detail::createLaplacePyr(image_warped, _bandNum, pyr_laplaceUMat);
    convertUMatToMat(pyr_laplaceUMat, pyr_laplace);

    std::vector<cv::Mat> pyr_weights(_bandNum+1);
    pyr_weights[0]=weight_warped;
    for (int i = 0; i < _bandNum; ++i)
        cv::pyrDown(pyr_weights[i], pyr_weights[i + 1]);

   
    std::vector<shared_ptr<MultiBandMap2DCPUEle> > dataCopy=d->data();
    for(int x=xminInt;x<xmaxInt;x++)
        for(int y=yminInt;y<ymaxInt;y++)
        {
            shared_ptr<MultiBandMap2DCPUEle> ele=dataCopy[y*d->w()+x];
            if(!ele.get())
            {
                ele=d->ele(y*d->w()+x);
            }
            {
                std::shared_lock<shared_mutex> lock(ele->mutexData);
                if(!ele->pyr_laplace.size())
                {
                    ele->pyr_laplace.resize(_bandNum+1);
                    ele->weights.resize(_bandNum+1);
                }

                int width=ELE_PIXELS,height=ELE_PIXELS;

                for (int i = 0; i <= _bandNum; ++i)
                {
                    if(ele->pyr_laplace[i].empty())
                    {
                        //fresh
                        cv::Rect rect(width*(x-xminInt),height*(y-yminInt),width,height);
                        pyr_laplace[i](rect).copyTo(ele->pyr_laplace[i]);
                        pyr_weights[i](rect).copyTo(ele->weights[i]);
                    }
                    else
                    {
                        if(pyr_laplace[i].type()==CV_32FC3)
                        {
                            int org =(x-xminInt)*width+(y-yminInt)*height*pyr_laplace[i].cols;
                            int skip=pyr_laplace[i].cols-ele->pyr_laplace[i].cols;
//注意！！！！！！！！
                            cv::Point3f *srcL=((cv::Point3f*)pyr_laplace[i].data)+org;
                            float       *srcW=((float*)pyr_weights[i].data)+org;

                            cv::Point3f *dstL=(cv::Point3f*)ele->pyr_laplace[i].data;
                            float       *dstW=(float*)ele->weights[i].data;

                            for(int eleY=0;eleY<height;eleY++,srcL+=skip,srcW+=skip)
                                for(int eleX=0;eleX<width;eleX++,srcL++,dstL++,srcW++,dstW++)
                                {
                                    if((*srcW)>=(*dstW))
                                    {
                                        *dstL=(*srcL);
                                        *dstW=*srcW;
                                    }
                                }
                        }
                        else if(pyr_laplace[i].type()==CV_16SC3)
                        {
                            int org =(x-xminInt)*width+(y-yminInt)*height*pyr_laplace[i].cols;
                            int skip=pyr_laplace[i].cols-ele->pyr_laplace[i].cols;

                            cv::Point3_<short> *srcL=((cv::Point3_<short>*)pyr_laplace[i].data)+org;
                            float       *srcW=((float*)pyr_weights[i].data)+org;

                            cv::Point3_<short> *dstL=(cv::Point3_<short>*)ele->pyr_laplace[i].data;
                            float       *dstW=(float*)ele->weights[i].data;

                            for(int eleY=0;eleY<height;eleY++,srcL+=skip,srcW+=skip)
                                for(int eleX=0;eleX<width;eleX++,srcL++,dstL++,srcW++,dstW++)
                                {
                                    if((*srcW)>=(*dstW))
                                    {
                                        *dstL=(*srcL);
                                        *dstW=*srcW;
                                    }
                                }

                            
                        }
                    }
                    width/=2;height/=2;
                }
                ele->Ischanged=true;
            }
        }
    //pi::timer.leave("MultiBandMap2DCPU::Apply");

    return true;
}


bool MultiBandMap2DCPU::renderFrame(const FusionFrame& frame)
{

    std::chrono::steady_clock::time_point time_renderbegin = std::chrono::steady_clock::now();
    shared_ptr<MultiBandMap2DCPUPrepare> p;
    shared_ptr<MultiBandMap2DCPUData>    d;
    {
        std::shared_lock<shared_mutex> lock(mutex);
        p=prepared;d=data;
    }
    if(frame.image.cols!=p->_camera.w||frame.image.rows!=p->_camera.h||frame.image.type()!=CV_8UC3)
    {
        cerr<<"MultiBandMap2DCPU::renderFrame: frame.first.cols!=p->_camera.w||frame.first.rows!=p->_camera.h||frame.first.type()!=CV_8UC3\n";

        std::cerr << "Map2DCPU::renderFrame: 输入图像参数不匹配！\n";
    
    std::cerr << "frame.first.cols = " << frame.image.cols 
              << ", expected = " << p->_camera.w << std::endl;
    
    std::cerr << "frame.first.rows = " << frame.image.rows 
              << ", expected = " << p->_camera.h << std::endl;

    std::cerr << "frame.first.type = " << frame.image.type() 
              << ", expected = " << CV_8UC3 
              << " (即 CV_8UC3 = " << CV_8UC3 << ")" << std::endl;
        return false;
    }
    // 1. pose->pts
    std::vector<cv::Point2d> imgPts;
    {
        imgPts.reserve(4);
        imgPts.push_back(cv::Point2d(0,0));
        imgPts.push_back(cv::Point2d(p->_camera.w,0));
        imgPts.push_back(cv::Point2d(0,p->_camera.h));
        imgPts.push_back(cv::Point2d(p->_camera.w,p->_camera.h));
    }
    vector<cv::Point2d> pts;
    pts.reserve(imgPts.size());
    cv::Point3d downLook(0,0,-1);
    if(frame.pose.translation().z()<0) downLook=cv::Point3d(0,0,1);
    for(int i=0;i<imgPts.size();i++)
    {
        //pi::Point3d axis=frame.second.get_rotation()*p->UnProject(imgPts[i]);
        cv::Point3d point3d = p->UnProject(imgPts[i]);//将像素坐标转换为归一化相机坐标，变为相机光心到该像素的射线。
        Eigen::Vector3d axis_e=frame.pose.so3() *Eigen::Vector3d(point3d.x, point3d.y, point3d.z);//将该方向转换至世界坐标系上
        cv::Point3d axis(axis_e.x(), axis_e.y(), axis_e.z());
        //cout<<"axis: ."<<axis<<endl;
        double dot_product = axis.dot(downLook);
        //std::cout << "[DEBUG] Dot product with downLook = " << dot_product << std::endl;

        if(axis.dot(downLook)<0.4)//downLook = (0,0,-1) 或 (0,0,1)，取决于相机高度的正负方向。如果小于 0.4，表示与地面法向接近 90°，投影会发散或不稳定。
        {
            return false;
        }
        axis_e=frame.pose.translation()
                -axis_e*(frame.pose.translation().z()/axis_e.z());

        Eigen::Vector3d cam_pos = frame.pose.translation();
        //std::cout << "[INFO] Camera Height (Z): " << cam_pos.z() << " m" << std::endl;

        cv::Point3d axis1(axis_e.x(), axis_e.y(), axis_e.z());  
        pts.push_back(cv::Point2d(axis1.x,axis1.y));
    }
    // 2. dest location?
    double xmin=pts[0].x;
    double xmax=xmin;
    double ymin=pts[0].y;
    double ymax=ymin;
   
    for(int i=1;i<pts.size();i++)
    {
        if(pts[i].x<xmin) xmin=pts[i].x;
        if(pts[i].y<ymin) ymin=pts[i].y;
        if(pts[i].x>xmax) xmax=pts[i].x;
        if(pts[i].y>ymax) ymax=pts[i].y;
    }


    //std::cout<<"xmin11: "<<xmin<<" "<<xmax<<" "<<ymin<<" "<<ymax<<endl;
    if(xmin<d->min().x||xmax>d->max().x||ymin<d->min().y||ymax>d->max().y)
    {
        if(p!=prepared)//what if prepare called?
        {
            return false;
        }
        if(!spreadMap(xmin,ymin,xmax,ymax))
        {
            return false;
        }
        else
        {
            std::shared_lock<shared_mutex> lock(mutex);
            if(p!=prepared)//what if prepare called?
            {
                return false;
            }
            d=data;//new data
        }
    }
    int xminInt=floor((xmin-d->min().x)*d->eleSizeInv());
    int yminInt=floor((ymin-d->min().y)*d->eleSizeInv());
    int xmaxInt= ceil((xmax-d->min().x)*d->eleSizeInv());
    int ymaxInt= ceil((ymax-d->min().y)*d->eleSizeInv());
    if(xminInt<0||yminInt<0||xmaxInt>d->w()||ymaxInt>d->h()||xminInt>=xmaxInt||yminInt>=ymaxInt)
    {
        cerr<<"MultiBandMap2DCPU::renderFrame:should never happen!\n";
        return false;
    }
    {
        xmin=d->min().x+d->eleSize()*xminInt;
        ymin=d->min().y+d->eleSize()*yminInt;
        xmax=d->min().x+d->eleSize()*xmaxInt;
        ymax=d->min().y+d->eleSize()*ymaxInt;
    }
    // 3.prepare weight and warp images
    cv::Mat weight_src;
    if(1)
    {

        auto t_start = std::chrono::high_resolution_clock::now();
        std::shared_lock<shared_mutex> lock(mutex);


        int w=frame.image.cols;
        int h=frame.image.rows;

        weightImage.create(h,w,CV_32FC1);
        float* p = (float*)weightImage.data;
        double fx = prepared->_camera.fx;
        double fy = prepared->_camera.fy;
        double cx = prepared->_camera.cx;
        double cy = prepared->_camera.cy;

        // 图像中心方向向量
        Eigen::Vector3d center_dir = Eigen::Vector3d((0.0 - cx) / fx, (0.0 - cy) / fy, 1.0).normalized();
        //std::cout << "[Debug] center_dir (unit vector): " << center_dir.transpose() << std::endl;

        // 地图分辨率（米/像素）
        double r = data->lengthPixel();
        //std::cout << "[Debug] map resolution r: " << r << " m/pixel" << std::endl;

        // 相机光轴方向（相对于世界坐标系）
        Eigen::Vector3d cam_z = frame.pose.so3() * Eigen::Vector3d(0, 0, 1);
        //std::cout << "[Debug] camera direction (cam_z): " << cam_z.transpose() << std::endl;

        // 地面法线方向
        Eigen::Vector3d normal_plane = prepared->_plane.so3() * Eigen::Vector3d(0, 0, 1);
        //std::cout << "[Debug] ground normal (from plane): " << normal_plane.transpose() << std::endl;

        // 相机高度

        Eigen::Vector3d t_cam = frame.pose.translation();
        Eigen::Vector3d t_plane = prepared->_plane.translation();

        // 垂直距离 = 法线 · (相机 - 地面)
        double Alt = std::abs(normal_plane.dot(t_cam - t_plane));
        //std::cout << "[Debug] Correct Alt (camera to plane): " << Alt << " meters" << std::endl;

        // 相机与地面法线夹角（弧度）
        double cos_theta = cam_z.dot(normal_plane);
        cos_theta = std::clamp(cos_theta, -1.0, 1.0);
        double theta = std::acos(cos_theta);
        //std::cout << "[Debug] cos(theta): " << cos_theta << ", theta (rad): " << theta
                //<< ", theta (deg): " << theta * 180.0 / M_PI << std::endl;

        // 图像对角线一半 Dm（单位：像素）
        double Dm = 0.5 * std::sqrt(w * w + h * h);
        //std::cout << "[Debug] Dm (half-diagonal in pixels): " << Dm << std::endl;

        // 计算全局正交性权重项
        double global_term = 1.0 - (Alt * std::tan(theta)) / (r * Dm);
        global_term = std::clamp(global_term, 0.0, 1.0);
        //std::cout << "[Debug] global_term (orthogonality weight): " << global_term << std::endl;

        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        //std::cout << "[Timing] Orthogonality weight calculation time: " << elapsed_ms << " ms" << std::endl;

        double gamma = 2.0;
        

        if (cached_max_dis < 0) {
            double max_dis = 0.0;
            for (int v = 0; v < h; ++v) {
                for (int u = 0; u < w; ++u) {
                    double dx = (u - cx) / fx;
                    double dy = (v - cy) / fy;
                    double d = std::sqrt(dx * dx + dy * dy);
                    if (d > max_dis) max_dis = d;
                }
            }
            cached_max_dis = max_dis;
        }

        
        for (int v = 0; v < h; ++v) {
            for (int u = 0; u < w; ++u) {
                double dx = (u - cx) / fx;
                double dy = (v - cy) / fy;
                double d = std::sqrt(dx * dx + dy * dy);
                float wgt = std::pow(1.0 - d / cached_max_dis, gamma);
                *p = std::max((wgt)*(float)global_term, 1e-5f);
                p++;
            }
        }
        weight_src=weightImage.clone();
        bool useSegMaskWeight = (!frame.segMask.empty() && frame.roofHeight > 0);

        if (useSegMaskWeight) {
        //std::cout << "[INFO] Adjusting weight for multi-homography (segmentation mask)...\n";
        cv::Mat maskWalls = (frame.segMask == 1);
        cv::Mat maskRoof  = (frame.segMask == 5);
        cv::Mat maskRoofFloat, maskWallsFloat;
        maskRoof.convertTo(maskRoofFloat, CV_32F, 1.0/255.0);
        maskWalls.convertTo(maskWallsFloat, CV_32F, 1.0/255.0);

        
        cv::GaussianBlur(maskRoofFloat, maskRoofFloat, cv::Size(31,31), 20);
        cv::GaussianBlur(maskWallsFloat, maskWallsFloat, cv::Size(31,31), 10);

        // 遍历墙体像素，把对应权重降低（例如乘0.5）
        for(int v=0; v < h; ++v) {
            float* wptr = weight_src.ptr<float>(v);
            // const uchar* mptr = maskWalls.ptr<uchar>(v);
            // const uchar* roofPtr = maskRoof.ptr<uchar>(v);
             const float* roofPtr = maskRoofFloat.ptr<float>(v);
        const float* wallPtr = maskWallsFloat.ptr<float>(v);
            for(int u=0; u < w; ++u) {
                    wptr[u] *= (1.0f + 0.2f * roofPtr[u]); // 屋顶区域轻微提升
            wptr[u] *= (1.0f - 0.2f * wallPtr[u]); // 墙体区域轻微抑制
                // if(mptr[u])
                //     wptr[u] *= 1.0f; // 
                // else if (roofPtr[u]) {
                //     wptr[u] *= 2.0f;  // 房顶 → 增强
                //     }
                // else{
                //     wptr[u] *= 1.0f;
                }
                
                
            }

        // cv::Mat weight_vis;
        // cv::normalize(weight_src, weight_vis, 0, 255, cv::NORM_MINMAX);
        // weight_vis.convertTo(weight_vis, CV_8UC1);
        // cv::Mat weight_color;
        // cv::applyColorMap(weight_vis, weight_color, cv::COLORMAP_JET);
        // cv::imshow("Weight Map", weight_color);
        // //cv::imwrite("weight_map_debug.png", weight_color);
        // cv::waitKey(1);
        cv::GaussianBlur(weight_src, weight_src, cv::Size(15,15), 5);
        }

    }
    else
    {
        std::shared_lock<shared_mutex> lock(mutex);
        weight_src=weightImage.clone();
        //cv::waitKey(1);
    }

    std::vector<cv::Point2f>  imgPtsCV;
    {
        imgPtsCV.reserve(imgPts.size());
        for(int i=0;i<imgPts.size();i++)
            imgPtsCV.push_back(cv::Point2f(imgPts[i].x,imgPts[i].y));
    }
    std::vector<cv::Point2f> destPoints;
    destPoints.reserve(imgPtsCV.size());
    for(int i=0;i<imgPtsCV.size();i++)
    {
        destPoints.push_back(cv::Point2f((pts[i].x-xmin)*d->lengthPixelInv(),
                             (pts[i].y-ymin)*d->lengthPixelInv()));
    }

    // cv::Mat transmtx = cv::getPerspectiveTransform(imgPtsCV, destPoints);

    // cv::Mat img_src;
    // //if(svar.GetInt("MultiBandMap2DCPU.ForceFloat",0))
    // if(0)
    //     frame.image.convertTo(img_src,CV_32FC3,1./255.);
    // else
    //     frame.image.convertTo(img_src,CV_16SC3);

    // cv::Mat weight_warped((ymaxInt-yminInt)*ELE_PIXELS,(xmaxInt-xminInt)*ELE_PIXELS,CV_32FC1);
    // cv::Mat image_warped((ymaxInt-yminInt)*ELE_PIXELS,(xmaxInt-xminInt)*ELE_PIXELS,img_src.type());
    // cv::warpPerspective(img_src, image_warped, transmtx, image_warped.size(),cv::INTER_LINEAR,cv::BORDER_REFLECT);
    // //cv::warpPerspective(img_src, image_warped, transmtx, image_warped.size(),cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));

    // cv::warpPerspective(weight_src, weight_warped, transmtx, weight_warped.size(),cv::INTER_NEAREST);

    cv::Mat img_src;
    frame.image.convertTo(img_src, CV_16SC3);

    cv::Mat image_warped, weight_warped;
    image_warped.create((ymaxInt - yminInt) * ELE_PIXELS, (xmaxInt - xminInt) * ELE_PIXELS, img_src.type());
    weight_warped.create((ymaxInt - yminInt) * ELE_PIXELS, (xmaxInt - xminInt) * ELE_PIXELS, CV_32FC1);
    image_warped.setTo(0);
    weight_warped.setTo(0);

    bool useMultiHomography = (!frame.segMask.empty() && frame.roofHeight > 0);

    if(0) {
        std::cout << "[INFO] Using Multi-Homography Warping with RoofHeight = " << frame.roofHeight << std::endl;

        cv::Mat groundMask = (frame.segMask == 0);
        cv::Mat roofMask   = (frame.segMask == 5);

        cv::Mat groundImage = img_src.clone();
        groundImage.setTo(cv::Scalar(0,0,0), ~groundMask);

        cv::Mat roofImage = img_src.clone();
        roofImage.setTo(cv::Scalar(0,0,0), ~roofMask);

        //cv::Mat transmtx_ground = cv::getPerspectiveTransform(imgPtsCV, destPoints);  // plane Z=0
        //cv::Mat transmtx_roof = cv::getPerspectiveTransform(imgPtsCV, destPoints);    // will adjust below

        double z_ground = 0.0;
        double z_roof = frame.roofHeight;

        for(int i = 0; i < 4 ;i++)
        {
            cv::Point3d p3d = p->UnProject(imgPts[i]);
            Eigen::Vector3d dir = frame.pose.so3() * Eigen::Vector3d(p3d.x, p3d.y, p3d.z);

            Eigen::Vector3d ground_point = frame.pose.translation() - dir * ((frame.pose.translation().z() + z_ground) / dir.z());
            destPoints[i] = cv::Point2f((ground_point.x() - xmin) * d->lengthPixelInv(),
                                    (ground_point.y() - ymin) * d->lengthPixelInv());
        }
        cv::Mat transmtx_ground = cv::getPerspectiveTransform(imgPtsCV, destPoints);

        for(int i=0; i<4; i++)
        {
            cv::Point3d p3d = p->UnProject(imgPts[i]);
            Eigen::Vector3d dir = frame.pose.so3() * Eigen::Vector3d(p3d.x, p3d.y, p3d.z);

            Eigen::Vector3d roof_point = frame.pose.translation() - dir * ((frame.pose.translation().z() + z_roof) / dir.z());
            destPoints[i] = cv::Point2f((roof_point.x() - xmin) * d->lengthPixelInv(),
                                    (roof_point.y() - ymin) * d->lengthPixelInv());
        }
        cv::Mat transmtx_roof = cv::getPerspectiveTransform(imgPtsCV, destPoints);

        cv::Mat groundWarped, roofWarped;
        cv::warpPerspective(groundImage, groundWarped, transmtx_ground, image_warped.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT);
        cv::warpPerspective(roofImage,   roofWarped,   transmtx_roof,   image_warped.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT);

        cv::Mat weight_ground, weight_roof;
        cv::warpPerspective(weight_src, weight_ground, transmtx_ground, weight_warped.size(), cv::INTER_NEAREST);
        cv::warpPerspective(weight_src, weight_roof,   transmtx_roof,   weight_warped.size(), cv::INTER_NEAREST);

        cv::Mat groundMaskWarped, roofMaskWarped;
        cv::warpPerspective(frame.segMask, groundMaskWarped, transmtx_ground, groundWarped.size(), cv::INTER_NEAREST);
        cv::warpPerspective(frame.segMask, roofMaskWarped, transmtx_roof, roofWarped.size(), cv::INTER_NEAREST);

        cv::Mat roofMaskInMap = (roofMaskWarped == 5);
        image_warped  = groundWarped.clone();
        roofWarped.copyTo(image_warped , roofMaskInMap);

        weight_warped = weight_ground.clone();
        weight_roof.copyTo(weight_warped, roofMaskInMap);
    }
    else{
        //std::cout << "[INFO] Using Single-Homography Warping (Z=0)" << std::endl;
        
        // Compute homography once
        // for(int i=0;i<4;i++)
        // {
        //     cv::Point3d p3d = p->UnProject(imgPts[i]);
        //     Eigen::Vector3d dir = frame.pose.so3() * Eigen::Vector3d(p3d.x, p3d.y, p3d.z);
        //     Eigen::Vector3d ground_point = frame.pose.translation() - dir * (frame.pose.translation().z() / dir.z());
        //     destPoints[i] = cv::Point2f((ground_point.x() - xmin) * d->lengthPixelInv(),
        //                                 (ground_point.y() - ymin) * d->lengthPixelInv());
        // }
        cv::Mat transmtx = cv::getPerspectiveTransform(imgPtsCV, destPoints);

        // Warp
        cv::warpPerspective(img_src, image_warped, transmtx, image_warped.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT);
        cv::warpPerspective(weight_src, weight_warped, transmtx, weight_warped.size(), cv::INTER_NEAREST);
    }


    if(0)//showed wraped!
    {
        cv::imshow("image_warped",image_warped);
       // cv::imshow("weight_warped",weight_warped);
        if(0)//Save  ImageWarped!
        {
            cout<<"Saving warped image.\n";
            cv::imwrite("image_warped.png",image_warped);
            cv::imwrite("weight_warped.png",weight_warped);
        }
        cv::waitKey(1);
    }

    // 4. blender dst to eles
    std::vector<cv::Mat> pyr_laplace;
    std::vector<cv::UMat> pyr_laplaceUMat;
    
    cv::detail::createLaplacePyr(image_warped, _bandNum, pyr_laplaceUMat);
    convertUMatToMat(pyr_laplaceUMat, pyr_laplace);

    std::vector<cv::Mat> pyr_weights(_bandNum+1);
    pyr_weights[0]=weight_warped;
    for (int i = 0; i < _bandNum; ++i)
        cv::pyrDown(pyr_weights[i], pyr_weights[i + 1]);

   
    std::vector<shared_ptr<MultiBandMap2DCPUEle> > dataCopy=d->data();
    for(int x=xminInt;x<xmaxInt;x++)
        for(int y=yminInt;y<ymaxInt;y++)
        {
            shared_ptr<MultiBandMap2DCPUEle> ele=dataCopy[y*d->w()+x];
            if(!ele.get())
            {
                ele=d->ele(y*d->w()+x);
            }
            {
                std::shared_lock<shared_mutex> lock(ele->mutexData);
                if(!ele->pyr_laplace.size())
                {
                    ele->pyr_laplace.resize(_bandNum+1);
                    ele->weights.resize(_bandNum+1);
                }

                int width=ELE_PIXELS,height=ELE_PIXELS;

                for (int i = 0; i <= _bandNum; ++i)
                {
                    if(ele->pyr_laplace[i].empty())
                    {
                        //fresh
                        cv::Rect rect(width*(x-xminInt),height*(y-yminInt),width,height);
                        pyr_laplace[i](rect).copyTo(ele->pyr_laplace[i]);
                        pyr_weights[i](rect).copyTo(ele->weights[i]);
                    }
                    else
                    {
                        if(pyr_laplace[i].type()==CV_32FC3)
                        {
                            int org =(x-xminInt)*width+(y-yminInt)*height*pyr_laplace[i].cols;
                            int skip=pyr_laplace[i].cols-ele->pyr_laplace[i].cols;
//注意！！！！！！！！
                            cv::Point3f *srcL=((cv::Point3f*)pyr_laplace[i].data)+org;
                            float       *srcW=((float*)pyr_weights[i].data)+org;

                            cv::Point3f *dstL=(cv::Point3f*)ele->pyr_laplace[i].data;
                            float       *dstW=(float*)ele->weights[i].data;

                            for(int eleY=0;eleY<height;eleY++,srcL+=skip,srcW+=skip)
                                for(int eleX=0;eleX<width;eleX++,srcL++,dstL++,srcW++,dstW++)
                                {
                                    if((*srcW)>=(*dstW))
                                    {
                                        *dstL=(*srcL);
                                        *dstW=*srcW;

                                        // float totalW = *srcW + *dstW + 1e-5f;  // 避免除0
                                        // *dstL = (*srcL * *srcW + *dstL * *dstW) / totalW;
                                        // *dstW = std::max(*dstW, *srcW);
                                    }
                                }
                        }
                        else if(pyr_laplace[i].type()==CV_16SC3)
                        {
                            
                            int org =(x-xminInt)*width+(y-yminInt)*height*pyr_laplace[i].cols;
                            int skip=pyr_laplace[i].cols-ele->pyr_laplace[i].cols;

                            cv::Point3_<short> *srcL=((cv::Point3_<short>*)pyr_laplace[i].data)+org;
                            float       *srcW=((float*)pyr_weights[i].data)+org;

                            cv::Point3_<short> *dstL=(cv::Point3_<short>*)ele->pyr_laplace[i].data;
                            float       *dstW=(float*)ele->weights[i].data;
                            
                            for(int eleY=0;eleY<height;eleY++,srcL+=skip,srcW+=skip)
                                for(int eleX=0;eleX<width;eleX++,srcL++,dstL++,srcW++,dstW++)
                                {
                                    if((*srcW)>=(*dstW))
                                    {
                                        *dstL=(*srcL);
                                        *dstW=*srcW;
                                    }
                                }
                            }
                    }
                    width/=2;height/=2;
                }
                ele->Ischanged=true;
            }
        }
    //pi::timer.leave("MultiBandMap2DCPU::Apply");

    return true;
}



void MultiBandMap2DCPU::RenderTime2File(){

    cout<<"sssss.txt"<<endl;
    ofstream f;
    f.open("RenderTimeStats.txt");
    f << fixed << setprecision(6);
    f << "#Stereo rect[ms], MP culling[ms], MP creation[ms], LBA[ms], KF culling[ms], Total[ms]" << endl;
    for(int i=0; i<vdrender_ms.size(); ++i)
    {
        f << vdrender_ms[i]<< endl;
    }

    f.close();
}



bool MultiBandMap2DCPU::spreadMap(double xmin,double ymin,double xmax,double ymax)
{
    //pi::timer.enter("MultiBandMap2DCPU::spreadMap");
    shared_ptr<MultiBandMap2DCPUData> d;
    {
        std::shared_lock<shared_mutex> lock(mutex);
        d=data;
    }
    int xminInt=floor((xmin-d->min().x)*d->eleSizeInv());
    int yminInt=floor((ymin-d->min().y)*d->eleSizeInv());
    int xmaxInt= ceil((xmax-d->min().x)*d->eleSizeInv());
    int ymaxInt= ceil((ymax-d->min().y)*d->eleSizeInv());
    xminInt=min(xminInt,0); yminInt=min(yminInt,0);
    xmaxInt=max(xmaxInt,d->w()); ymaxInt=max(ymaxInt,d->h());
    int w=xmaxInt-xminInt;
    int h=ymaxInt-yminInt;
    cv::Point2d min,max;
    {
        min.x=d->min().x+d->eleSize()*xminInt;
        min.y=d->min().y+d->eleSize()*yminInt;
        max.x=min.x+w*d->eleSize();
        max.y=min.y+h*d->eleSize();
    }
    std::vector<std::shared_ptr<MultiBandMap2DCPUEle> > dataOld=d->data();
    std::vector<std::shared_ptr<MultiBandMap2DCPUEle> > dataCopy;
    dataCopy.resize(w*h);
    {
        for(int x=0,xend=d->w();x<xend;x++)
            for(int y=0,yend=d->h();y<yend;y++)
            {
                dataCopy[x-xminInt+(y-yminInt)*w]=dataOld[y*d->w()+x];
            }
    }
    //apply
    {
        std::unique_lock<shared_mutex> lock(mutex);
        data=std::shared_ptr<MultiBandMap2DCPUData>(new MultiBandMap2DCPUData(d->eleSize(),d->lengthPixel(),
                                                 cv::Point3d(max.x,max.y,d->max().z),
                                                 cv::Point3d(min.x,min.y,d->min().z),
                                                 w,h,dataCopy));
    }
    //pi::timer.leave("MultiBandMap2DCPU::spreadMap");
    return true;
}

bool MultiBandMap2DCPU::getFrame(std::pair<cv::Mat,Sophus::SE3d>& frame)
{
    // pi::ReadMutex lock(mutex);
    // pi::ReadMutex lock1(prepared->mutexFrames);
    //std::shared_lock<shared_mutex> lock(mutex);
    //std::shared_lock<shared_mutex> lock1(prepared->mutexFrames);
    if(prepared->_frames.size())
    {
        frame=prepared->_frames.front();
        prepared->_frames.pop_front();
        return true;
    }
    else return false;
}

void MultiBandMap2DCPU::run()
{
    std::pair<cv::Mat,Sophus::SE3d> frame;
    while(1)
    {
        
            if(getFrame(frame))
            {
                //pi::timer.enter("MultiBandMap2DCPU::renderFrame");
                renderFrame(frame);
                //pi::timer.leave("MultiBandMap2DCPU::renderFrame");
            }
        
        //sleep(1);
    }
}

void MultiBandMap2DCPU::scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    void* ptr = glfwGetWindowUserPointer(window);
    if (!ptr) return;

    auto* viewer = static_cast<MultiBandMap2DCPU*>(ptr);
    if (yoffset > 0)
        viewer->zoom_scale *= 1.1;
    else if (yoffset < 0)
        viewer->zoom_scale /= 1.1;

    viewer->zoom_scale = std::max(0.1, std::min(10.0, viewer->zoom_scale));
    std::cout << "[Zoom] zoom_scale now: " << viewer->zoom_scale << std::endl;
}


void MultiBandMap2DCPU::key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (action != GLFW_PRESS && action != GLFW_REPEAT) return;

    MultiBandMap2DCPU* map = static_cast<MultiBandMap2DCPU*>(glfwGetWindowUserPointer(window));
    if (!map) return;

    double delta = 0.5 / map->zoom_scale; // 缩放越大，移动越精细
    switch (key)
    {
        case GLFW_KEY_LEFT:  map->pan_offset_x -= delta; break;
        case GLFW_KEY_RIGHT: map->pan_offset_x += delta; break;
        case GLFW_KEY_UP:    map->pan_offset_y += delta; break;
        case GLFW_KEY_DOWN:  map->pan_offset_y -= delta; break;
        case GLFW_KEY_R:     // Reset view
            map->pan_offset_x = 0.0;
            map->pan_offset_y = 0.0;
            map->zoom_scale = 1.0;
            return;
        case GLFW_KEY_S:
        {
            static int save_count = 0;
            std::ostringstream fname;
            fname << "save_" << std::setw(5) << std::setfill('0') << save_count << ".png";
            std::cout << "[INFO] Saving map to " << fname.str() << std::endl;
            map->save(fname.str());
            save_count++;
            return;
        }
    }

    std::cout << "[Pan] offset: (" << map->pan_offset_x << ", " << map->pan_offset_y << ")\n";
}

//原始代码：
// void MultiBandMap2DCPU::draw2D()
// {
//     shared_ptr<MultiBandMap2DCPUPrepare> p;
//     shared_ptr<MultiBandMap2DCPUData> d;
//     {
//         std::shared_lock<shared_mutex> lock(mutex);
//         p = prepared;
//         d = data;
//     }
//     std::chrono::steady_clock::time_point time_renderbegin = std::chrono::steady_clock::now();
//     GLint last_texture_ID;
//     glGetIntegerv(GL_TEXTURE_BINDING_2D, &last_texture_ID);//获取当前绑定在目标的纹理对象名称，并存储到last_texture_id中
//     //     glClearColor(1.0f, 1.0f, 1.0f, 1.0f); // 白色背景
//     // glClear(GL_COLOR_BUFFER_BIT);  // 清除颜色缓冲区，应用背景色
//     std::vector<shared_ptr<MultiBandMap2DCPUEle> > dataCopy = d->data();
//     int wCopy = d->w(),hCopy = d->h();

//     cout<<"wCopy: "<<d->w()<<" "<<d->h()<<endl;
//     glColor3ub(255,255,255);
//     for(int x = 0; x < wCopy; x++)
//         for(int y = 0; y < hCopy; y++)
//         {
//             int idxData = y*wCopy + x;

//             float x0 = d->min().x+x*d->eleSize();
//             float y0=d->min().y+y*d->eleSize();
//             float x1=x0+d->eleSize();
//             float y1=y0+d->eleSize();
//             //cout<<"map x0:"<<x0<<" map y0"<<y0<<endl;
//             //cout<<"wCopy1111"<<endl;
//             shared_ptr<MultiBandMap2DCPUEle> ele=dataCopy[idxData];//获得该元素块
//             if(!ele.get()) continue;
//             //if(ele->img.empty()) continue;
//             {
//                 {
//                     std::shared_lock<shared_mutex> lock(mutex);
//                     if(!(ele->pyr_laplace.size()&&ele->weights.size()
//                          &&ele->pyr_laplace.size()==ele->weights.size())) continue;

//                     if(ele->Ischanged)
//                     {
//                         bool updated = false, inborder = false;
//                         if(_highQualityShow)
//                         {
//                             vector<shared_ptr<MultiBandMap2DCPUEle> > neighbors;
//                             neighbors.reserve(9);
//                             for(int yi=y-1;yi<=y+1;yi++)
//                                 for(int xi=x-1;xi<=x+1;xi++)
//                                 {
//                                     if(yi<0||yi>=hCopy||xi<0||xi>=wCopy)
//                                     {
//                                         neighbors.push_back(shared_ptr<MultiBandMap2DCPUEle>());
//                                         inborder=true;
//                                     }
//                                     else neighbors.push_back(dataCopy[yi*wCopy+xi]);
//                                 }
//                             updated=ele->updateTexture(neighbors);
//                         }
//                         else
//                             updated = ele->updateTexture();
                        
//                     }

//                 }
//                 if(ele->texName)
//                 {
//                     glBindTexture(GL_TEXTURE_2D, ele->texName);
//                     float x_min = -10;
//                     float x_max = 10;
//                     float y_min = -10;
//                     float y_max = 10;

//                     // float x_min = -500;
//                     // float x_max = 1000;
//                     // float y_min = -500;
//                     // float y_max = 1000;

//                     //phantom3
//                     // float x_min = -1000;
//                     // float x_max = 1000;
//                     // float y_min = -700;
//                     // float y_max = 1000;

//                     // float x_min = -1000;
//                     // float x_max = 1000;
//                     // float y_min = -1000;
//                     // float y_max = 1000;
//                     float x0_n = 2*(x0 - x_min) / (x_max - x_min)-1;
//                     float x1_n = 2*(x1 - x_min) / (x_max - x_min)-1;

//                     float y0_n = 2*(y0 - y_min) / (y_max - y_min)-1;
//                     float y1_n = 2*(y1 - y_min) / (y_max - y_min)-1;
//                     //cout<<"x0_n: "<<x0_n<<" x1_n"<<x1_n<<" y0_n"<<y0_n<<" y1_n"<<y1_n<<endl;
//                     //cout<<"x0: "<<x0<<" x1"<<x1<<" y0"<<y0<<" y1"<<y1<<endl;
//                     glBegin(GL_QUADS);
//                     glTexCoord2f(0.0f, 0.0f); glVertex2f(x0_n,y0_n);
//                     glTexCoord2f(0.0f, 1.0f); glVertex2f(x0_n,y1_n);
//                     glTexCoord2f(1.0f, 1.0f); glVertex2f(x1_n,y1_n);
//                     glTexCoord2f(1.0f, 0.0f); glVertex2f(x1_n,y0_n);
//                     glEnd();
//                 }

//             }


//         }
//         glBindTexture(GL_TEXTURE_2D, last_texture_ID);

//         std::chrono::steady_clock::time_point time_renderend = std::chrono::steady_clock::now();
//         double timerender = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_renderend - time_renderbegin).count();
//         cout<<"write render success!"<<endl;
//         std::ofstream file("RenderTimeStats.txt", std::ios::app);
//         vdrender_ms.push_back(timerender);
//         file << std::fixed << std::setprecision(6);
//         file << timerender << std::endl;
//         file.close();
// }


//带缩放的代码：
void MultiBandMap2DCPU::draw2D()
{
    shared_ptr<MultiBandMap2DCPUPrepare> p;
    shared_ptr<MultiBandMap2DCPUData> d;
    {
        std::shared_lock<shared_mutex> lock(mutex);
        p = prepared;
        d = data;
    }

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f); // 设置清屏背景色
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // 清除颜色和深度缓冲

    std::chrono::steady_clock::time_point time_renderbegin = std::chrono::steady_clock::now();
    GLint last_texture_ID;
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &last_texture_ID);//获取当前绑定在目标的纹理对象名称，并存储到last_texture_id中
    //     glClearColor(1.0f, 1.0f, 1.0f, 1.0f); // 白色背景
    // glClear(GL_COLOR_BUFFER_BIT);  // 清除颜色缓冲区，应用背景色
    std::vector<shared_ptr<MultiBandMap2DCPUEle> > dataCopy = d->data();
    int wCopy = d->w(),hCopy = d->h();
    ensureTilePublisherStarted();

    

    // 设置缩放后的视图矩阵
    // float x_center = 0.5f * (d->min().x + d->max().x) + pan_offset_x;
    // float y_center = 0.5f * (d->min().y + d->max().y) + pan_offset_y;

    // float width = d->max().x - d->min().x;
    // float height = d->max().y - d->min().y;

    // float view_width = width / zoom_scale;
    // float view_height = height / zoom_scale;

  
    // glMatrixMode(GL_PROJECTION);
    // glLoadIdentity();
    // glOrtho(x_center - view_width/2, x_center + view_width/2,
    //         y_center - view_height/2, y_center + view_height/2,
    //         -1.0, 1.0);


    // 1️⃣ 计算地图中心
float x_center = 0.5f * (d->min().x + d->max().x) + pan_offset_x;
float y_center = 0.5f * (d->min().y + d->max().y) + pan_offset_y;

// 2️⃣ 计算地图世界坐标的宽高
float width = d->max().x - d->min().x;
float height = d->max().y - d->min().y;

// 3️⃣ 默认缩放范围（未修正比例时）
float view_width_raw = width / zoom_scale;
float view_height_raw = height / zoom_scale;

// 4️⃣ 查询当前窗口像素尺寸
int win_w=2200;
int win_h = 1600;


// 5️⃣ 计算窗口的像素长宽比
float window_aspect = static_cast<float>(win_w) / static_cast<float>(win_h);
float map_aspect = view_width_raw / view_height_raw;

// 6️⃣ 按窗口比例修正视口范围
float view_width, view_height;
if (window_aspect > map_aspect) {
    // 窗口更宽 → 按高度铺满，左右留黑边
    view_height = view_height_raw;
    view_width = view_height * window_aspect;
} else {
    // 窗口更高 → 按宽度铺满，上下留黑边
    view_width = view_width_raw;
    view_height = view_width / window_aspect;
}

// 7️⃣ 设置正交投影
glMatrixMode(GL_PROJECTION);
glLoadIdentity();
glOrtho(x_center - view_width/2, x_center + view_width/2,
        y_center - view_height/2, y_center + view_height/2,
        -1.0, 1.0);
glMatrixMode(GL_MODELVIEW);
glLoadIdentity();



    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glColor3ub(255,255,255);
    for(int x = 0; x < wCopy; x++)
        for(int y = 0; y < hCopy; y++)
        {
            int idxData = y*wCopy + x;

            float x0 = d->min().x+x*d->eleSize();
            float y0=d->min().y+y*d->eleSize();
            float x1=x0+d->eleSize();
            float y1=y0+d->eleSize();


            //cout<<"map x0:"<<x0<<" map y0"<<y0<<endl;
            //cout<<"wCopy1111"<<endl;
            shared_ptr<MultiBandMap2DCPUEle> ele=dataCopy[idxData];//获得该元素块
            if(!ele.get()) continue;
            //if(ele->img.empty()) continue;
            {
                {
                    std::shared_lock<shared_mutex> lock(mutex);
                    if(!(ele->pyr_laplace.size()&&ele->weights.size()
                         &&ele->pyr_laplace.size()==ele->weights.size())) continue;

                    if(ele->Ischanged)
                    {
                        bool updated = false, inborder = false;
                        vector<shared_ptr<MultiBandMap2DCPUEle> > neighbors;
                        if(_highQualityShow)
                        {
                            neighbors.reserve(9);
                            for(int yi=y-1;yi<=y+1;yi++)
                                for(int xi=x-1;xi<=x+1;xi++)
                                {
                                    if(yi<0||yi>=hCopy||xi<0||xi>=wCopy)
                                    {
                                        neighbors.push_back(shared_ptr<MultiBandMap2DCPUEle>());
                                        inborder=true;
                                    }
                                    else neighbors.push_back(dataCopy[yi*wCopy+xi]);
                                }
                            updated=ele->updateTexture(neighbors);
                        }
                        else
                            updated = ele->updateTexture();

                        if (updated) {
                            if (neighbors.empty()) {
                                neighbors.push_back(ele);
                            }
                            publishTileIfNeeded(x, y, ele, neighbors, d);
                        }
                        
                    }

                }
                if(ele->texName)
                {
                    glBindTexture(GL_TEXTURE_2D, ele->texName);
                    float x_min = -10;
                    float x_max = 10;
                    float y_min = -10;
                    float y_max = 10;

                    
                    // float x0_n = 2*(x0 - x_min) / (x_max - x_min)-1;
                    // float x1_n = 2*(x1 - x_min) / (x_max - x_min)-1;

                    // float y0_n = 2*(y0 - y_min) / (y_max - y_min)-1;
                    // float y1_n = 2*(y1 - y_min) / (y_max - y_min)-1;


                    //cout<<"x0_n: "<<x0_n<<" x1_n"<<x1_n<<" y0_n"<<y0_n<<" y1_n"<<y1_n<<endl;
                    //cout<<"x0: "<<x0<<" x1"<<x1<<" y0"<<y0<<" y1"<<y1<<endl;

                    glBegin(GL_QUADS);
                    glTexCoord2f(0.0f, 0.0f); glVertex2f(x0, y0);
                    glTexCoord2f(0.0f, 1.0f); glVertex2f(x0, y1);
                    glTexCoord2f(1.0f, 1.0f); glVertex2f(x1, y1);
                    glTexCoord2f(1.0f, 0.0f); glVertex2f(x1, y0);
                    glEnd();

                    // glBegin(GL_QUADS);
                    // glTexCoord2f(0.0f, 1.0f); glVertex2f(x0, y0);
                    // glTexCoord2f(0.0f, 0.0f); glVertex2f(x0, y1);
                    // glTexCoord2f(1.0f, 0.0f); glVertex2f(x1, y1);
                    // glTexCoord2f(1.0f, 1.0f); glVertex2f(x1, y0);
                    // glEnd();
                }

            }


        }
        glBindTexture(GL_TEXTURE_2D, last_texture_ID);

        std::chrono::steady_clock::time_point time_renderend = std::chrono::steady_clock::now();
        double timerender = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_renderend - time_renderbegin).count();
        cout<<"write render success!"<<endl;
        std::ofstream file("RenderTimeStats.txt", std::ios::app);
        vdrender_ms.push_back(timerender);
        file << std::fixed << std::setprecision(6);
        file << timerender << std::endl;
        file.close();
}







bool MultiBandMap2DCPU::save(const std::string& filename)
{

    cout<<"begin save: "<<endl;
    // determin minmax
    shared_ptr<MultiBandMap2DCPUPrepare> p;
    shared_ptr<MultiBandMap2DCPUData>    d;
    {
        p=prepared;d=data;
    }
    if(d->w()==0||d->h()==0) return false;

    cv::Point2i minInt(1e6,1e6),maxInt(-1e6,-1e6);
    int contentCount=0;
    for(int x=0;x<d->w();x++)
        for(int y=0;y<d->h();y++)
        {
            shared_ptr<MultiBandMap2DCPUEle> ele=d->data()[x+y*d->w()];
            if(!ele.get()) continue;
            {
                std::shared_lock<shared_mutex> lock(ele->mutexData);
                if(!ele->pyr_laplace.size()) continue;
            }
            contentCount++;
            minInt.x=min(minInt.x,x); minInt.y=min(minInt.y,y);
            maxInt.x=max(maxInt.x,x); maxInt.y=max(maxInt.y,y);
        }

    maxInt=maxInt+cv::Point2i(1,1);
    int border = 2;
    minInt.x = std::max(0, minInt.x - border);
    minInt.y = std::max(0, minInt.y - border);
    maxInt.x = std::min(d->w(), maxInt.x + border);
    maxInt.y = std::min(d->h(), maxInt.y + border);
    //minInt=minInt-cv::Point2i(10,10);
    cv::Point2i wh=maxInt-minInt;
    vector<cv::Mat> pyr_laplace(_bandNum+1);
    vector<cv::Mat> pyr_weights(_bandNum+1);
    for(int i=0;i<=0;i++)
        pyr_weights[i]=cv::Mat::zeros(wh.y*ELE_PIXELS,wh.x*ELE_PIXELS,CV_32FC1);

    for(int x=minInt.x;x<maxInt.x;x++)
        for(int y=minInt.y;y<maxInt.y;y++)
        {
            shared_ptr<MultiBandMap2DCPUEle> ele=d->data()[x+y*d->w()];
            if(!ele.get()) continue;
            {
                std::shared_lock<shared_mutex> lock(ele->mutexData);
                //pi::ReadMutex lock(ele->mutexData);
                if(!ele->pyr_laplace.size()) continue;
                int width=ELE_PIXELS,height=ELE_PIXELS;

                for (int i = 0; i <= _bandNum; ++i)
                {
                    cv::Rect rect(width*(x-minInt.x),height*(y-minInt.y),width,height);
                    if(pyr_laplace[i].empty())
                        pyr_laplace[i]=cv::Mat::zeros(wh.y*height,wh.x*width,ele->pyr_laplace[i].type());
                    ele->pyr_laplace[i].copyTo(pyr_laplace[i](rect));
                    if(i==0)
                        ele->weights[i].copyTo(pyr_weights[i](rect));
                    height>>=1;width>>=1;
                }
            }
        }
    std::vector<cv::UMat> pyr_laplaceUMat;
    convertMatToUMat(pyr_laplace, pyr_laplaceUMat);
    cv::detail::restoreImageFromLaplacePyr(pyr_laplaceUMat);
    convertUMatToMat(pyr_laplaceUMat, pyr_laplace);

    cv::Mat result=pyr_laplace[0];
    if(result.type()==CV_16SC3) result.convertTo(result,CV_8UC3);
    //result.setTo(cv::Scalar::all(svar.GetInt("Result.BackGroundColor")),pyr_weights[0]==0);设置背景颜色，将 pyr_weights[0] 为0的区域设置为指定的背景颜色

    result.setTo(cv::Scalar(255, 255, 255), pyr_weights[0]==0);  
    cv::imwrite(filename,result);
    cout<<"Resolution:["<<result.cols<<" "<<result.rows<<"]";
    // if(svar.exist("GPS.Origin"))
    //       cout<<",_lengthPixel:"<<d->lengthPixel()
    //    <<",Area:"<<contentCount*d->eleSize()*d->eleSize()<<endl;
    return true;
}


bool MultiBandMap2DCPU::save_1(const std::string& filename)
{
        cout << "begin save: " << endl;

    // === 1️⃣ 拿到data指针 ===
    shared_ptr<MultiBandMap2DCPUPrepare> p;
    shared_ptr<MultiBandMap2DCPUData>    d;
    {
        p = prepared;
        d = data;
    }
    if (d->w() == 0 || d->h() == 0) return false;

    // === 2️⃣ 地图分辨率 ===
    double resolution = d->lengthPixel();

    // === 3️⃣ 精确的世界坐标范围 ===
    double minWorldX = d->min().x;
    double minWorldY = d->min().y;
    double maxWorldX = d->max().x;
    double maxWorldY = d->max().y;

    cout << "World Bounds: [" << minWorldX << ", " << minWorldY << "] to ["
         << maxWorldX << ", " << maxWorldY << "]" << endl;

    // === 4️⃣ 输出大图尺寸（像素） ===
    int outWidth = static_cast<int>(std::ceil((maxWorldX - minWorldX) / resolution));
    int outHeight = static_cast<int>(std::ceil((maxWorldY - minWorldY) / resolution));

    cout << "Output Image Size: " << outWidth << " x " << outHeight << endl;

    // === 5️⃣ 创建拉普拉斯金字塔拼接画布 ===
    vector<cv::Mat> pyr_laplace(_bandNum + 1);
    vector<cv::Mat> pyr_weights(_bandNum + 1);

    for (int i = 0; i <= _bandNum; ++i)
    {
        int scale = 1 << i;
        pyr_laplace[i] = cv::Mat::zeros(outHeight / scale, outWidth / scale, CV_16SC3);
        pyr_weights[i] = cv::Mat::zeros(outHeight / scale, outWidth / scale, CV_32FC1);
    }

    // === 6️⃣ 逐tile复制 ===
    for (int x = 0; x < d->w(); x++)
    for (int y = 0; y < d->h(); y++)
    {
        shared_ptr<MultiBandMap2DCPUEle> ele = d->data()[x + y * d->w()];
        if (!ele) continue;

        {
            std::shared_lock<shared_mutex> lock(ele->mutexData);
            if (!ele->pyr_laplace.size()) continue;

            // ✅ Tile左上角世界坐标
            double tileWorldX = d->min().x + x * d->eleSize();
            double tileWorldY = d->min().y + y * d->eleSize();

            // ✅ 真实偏移像素
            double offsetX_f = (tileWorldX - minWorldX) / resolution;
            double offsetY_f = (tileWorldY - minWorldY) / resolution;

            int offsetX = static_cast<int>(std::round(offsetX_f));
            int offsetY = static_cast<int>(std::round(offsetY_f));

            

            int width = ELE_PIXELS;
            int height = ELE_PIXELS;

            for (int i = 0; i <= _bandNum; ++i)
            {
                cv::Rect roi(offsetX, offsetY, width, height);

                // 🟠 检查是否超出边界
                if (roi.x < 0 || roi.y < 0 || roi.x + roi.width > pyr_laplace[i].cols || roi.y + roi.height > pyr_laplace[i].rows)
                {
                    continue;
                }

                ele->pyr_laplace[i].copyTo(pyr_laplace[i](roi));
                if (i == 0)
                    ele->weights[i].copyTo(pyr_weights[i](roi));

                width >>= 1;
                height >>= 1;
            }
        }
    }

    // === 7️⃣ 拉普拉斯金字塔复原 ===
    std::vector<cv::UMat> pyr_laplaceUMat;
    convertMatToUMat(pyr_laplace, pyr_laplaceUMat);
    cv::detail::restoreImageFromLaplacePyr(pyr_laplaceUMat);
    convertUMatToMat(pyr_laplaceUMat, pyr_laplace);

    cv::Mat result = pyr_laplace[0];

    // === 8️⃣ 类型转换
    if (result.type() == CV_16SC3)
        result.convertTo(result, CV_8UC3);

    // === 9️⃣ 背景可选填充
    // result.setTo(cv::Scalar::all(背景色), pyr_weights[0]==0);

    // === 10️⃣ 写入
    cv::imwrite(filename, result);

    cout << "Saved image. Resolution: [" << result.cols << " x " << result.rows << "]" << endl;

    return true;
}
