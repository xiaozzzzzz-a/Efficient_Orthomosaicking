#include "Map2DCPU.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Core>
#include <iostream>
#include "type.h"
#include <thread>
#include <mutex>
#include <unistd.h>
#include <GL/glut.h> 

using namespace std;

/**

  __________max
  |    |    |
  |____|____|
  |    |    |
  |____|____|
 min
 */

// pcl::PointCloud<pcl::PointXYZRGB>::Ptr Map2DCPU::imageToPointCloud(const cv::Mat& image,int x0, int y0)
// {
//     pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
//     for (int i = 0; i < image.rows; ++i){
//         for (int j = 0; j < image.cols; ++j){
//             cv::Vec3b color = image.at<cv::Vec3b>(i, j);
//             pcl::PointXYZRGB point;
//             point.x = j+x0;
//             point.y = i+y0;
//             point.z = 0;
//             point.r = color[2];
//             point.g = color[1];
//             point.b = color[0];
//             cloud->points.push_back(point);
//         }
//     }
//     // cloud->width = image.cols;
//     // cloud->height = image.rows;
//     cloud->is_dense = true;
//     return cloud;
// }

bool Map2DCPU::Map2DCPUData::prepare(std::shared_ptr<Map2DCPUPrepare> prepared)
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

    {
        double minh;
        if(_min.z>0) minh=_min.z;
        else minh=-_max.z;
        cout<<"prepared min height: "<<minh<<endl;
        cv::Point3d line=prepared->UnProject(cv::Point2d(prepared->_camera.w,prepared->_camera.h))
                -prepared->UnProject(cv::Point2d(0,0));
        double radius=0.5*minh*sqrt((line.x*line.x+line.y*line.y));
        _lengthPixel=2*radius/sqrt(prepared->_camera.w*prepared->_camera.w
                                   +prepared->_camera.h*prepared->_camera.h);//计算出每个像素所对应的实际大小

        //_lengthPixel/=svar.GetDouble("Map2D.Scale",1);
        _lengthPixel/=0.5;

        _lengthPixelInv=1./_lengthPixel;
        //这部分代码通过将 radius 加到 _max 并从 _min 中减去 radius 来扩大场景的边界。然后计算场景的中心点 center，并调整 _min 和 _max 使得场景以新的中心点对称。
        _min=_min-cv::Point3d(radius,radius,0);
        _max=_max+cv::Point3d(radius,radius,0);
        cv::Point3d center=0.5*(_min+_max);
        _min=2*_min-center;_max=2*_max-center;

        _eleSize=ELE_PIXELS*_lengthPixel;
        _eleSizeInv=1./_eleSize;
        {
            _w=ceil((_max.x-_min.x)/_eleSize);//计算在宽和高中的单元格的数量
            _h=ceil((_max.y-_min.y)/_eleSize);

            cout<<"_w:"<<_w<<"_h"<<_h<<endl;
            _max.x=_min.x+_eleSize*_w;
            _max.y=_min.y+_eleSize*_h;
            _data.resize(_w*_h);
        }
    }
    return true;
}

Map2DCPU::Map2DCPUEle::~Map2DCPUEle()
{
    // if(texName) pi::gl::Signal_Handle::instance().delete_texture(texName);
}

Map2DCPU::Map2DCPU(bool thread)
    :alpha(0),
     _valid(false),_thread(thread)
{
}



bool Map2DCPU::prepare(const Sophus::SE3d& plane,const PinHoleParameters& camera,
                const std::deque<std::pair<cv::Mat,Sophus::SE3d> >& frames)
{
    //insert frames
    std::shared_ptr<Map2DCPUPrepare> p(new Map2DCPUPrepare);
    std::shared_ptr<Map2DCPUData>    d(new Map2DCPUData);

    if(p->prepare(plane,camera,frames))
        if(d->prepare(p))
        {
            std::unique_lock<std::shared_mutex> lock(mutex);
            prepared=p;
            data=d;
            weightImage.release();
            cout<<"create new thread Map2DCPU::run"<<endl;
            run_thread = std::thread(&Map2DCPU::run, this);
            cout<<"create success new thread Map2DCPU::run"<<endl;
            
            _valid=true;
            return true;
        }
    return false;
}

bool Map2DCPU::feed(cv::Mat img,const Sophus::SE3d& pose)
{
    if(!_valid) return false;
    std::shared_ptr<Map2DCPUPrepare> p;
    std::shared_ptr<Map2DCPUData>    d;
    {
        std::shared_lock<std::shared_mutex> lock(mutex);
        p=prepared;d=data;
    }
    std::pair<cv::Mat,Sophus::SE3d> frame(img,p->_plane.inverse()*pose);
    if(_thread)
    {
        std::unique_lock<std::shared_mutex> lock(p->mutexFrames);
        p->_frames.push_back(frame);
        if(p->_frames.size()>20) p->_frames.pop_front();
        return true;
    }
    else
    {
        return renderFrame(frame);
    }
}



bool Map2DCPU::renderFrame(const std::pair<cv::Mat,Sophus::SE3d>& frame)
{
    std::shared_ptr<Map2DCPUPrepare> p;
    std::shared_ptr<Map2DCPUData>    d;
    {
        std::shared_lock<shared_mutex> lock(mutex);
        p=prepared;d=data;
    }
    if(frame.first.cols!=p->_camera.w||frame.first.rows!=p->_camera.h||frame.first.type()!=CV_8UC3)
    {
        //cerr<<"Map2DCPU::renderFrame: frame.first.cols!=p->_camera.w||frame.first.rows!=p->_camera.h||frame.first.type()!=CV_8UC3\n";

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
    std::vector<cv::Point2d>   imgPts;
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
        cv::Point3d point3d = p->UnProject(imgPts[i]);
        Eigen::Vector3d axis_e=frame.second.so3() *Eigen::Vector3d(point3d.x, point3d.y, point3d.z);
        cv::Point3d axis(axis_e.x(), axis_e.y(), axis_e.z());
        if(axis.dot(downLook)<0.4)
        {
            return false;
        }
        axis_e=frame.second.translation()
                -axis_e*(frame.second.translation().z()/axis_e.z());//比例因子来缩放 axis_e。这一步的目的是将 axis_e 沿z轴缩放，使其与 frame.second.translation() 的z坐标相同,并将其与相机位移相机相减，得到该点在相机平面上的坐标
        cv::Point3d axis1(axis_e.x(), axis_e.y(), axis_e.z());
        // if(i==0)
        // cout<<axis_e.x()<<","<<axis_e.y()<<","<<axis_e.z()<<endl;
        pts.push_back(cv::Point2d(axis1.x,axis1.y));//整个过程是将图像中的四个角点反投影到3D点，然后通过变换矩阵将其转换到世界坐标系中，检查点的方向，最后将其投影相机位姿所在的新的平面上，并存储在2D点向量中
        //cout<<pts[i].x<<" "<<pts[i].y<<" ";
    }

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
    if(xmin<d->min().x||xmax>d->max().x||ymin<d->min().y||ymax>d->max().y)//如果当前图像的边角坐标超过map2d的边界，则需要扩展图像
    {
        //cout<<xmin<<" "<<xmax<<" "<<ymin<<" "<<ymax<<endl;
        //cout<<"d->minx: "<<d->min().x<<"d->maxx" <<d->max().x<<"d->miny"<<d->min().y<<"d->maxy"<<d->max().y<<endl;
        if(p!=prepared)
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
        //cout<<"sssd->minx: "<<d->min().x<<"d->maxx" <<d->max().x<<"d->miny"<<d->min().y<<"d->maxy"<<d->max().y<<endl;
    }

    //1.计算离散网格索引
    int xminInt = floor((xmin-d->min().x)*d->eleSizeInv());
    int yminInt = floor((ymin-d->min().y)*d->eleSizeInv());
    int xmaxInt = ceil((xmax-d->min().x)*d->eleSizeInv());
    int ymaxInt = ceil((ymax-d->min().y)*d->eleSizeInv());
    //cout<<"xminInt: "<<xminInt<<"yminInt: "<<yminInt<<endl;
    //2.检查索引的有效性,将世界坐标系中的最小和最大点映射到离散的网格索引上，如果超出了允许范围就返回false
    if(xminInt<0||yminInt<0||xmaxInt>d->w()||ymaxInt>d->h()||xminInt>=xmaxInt||yminInt>=ymaxInt)
    {
        cerr<<"Map2DCPU::renderFrame:should never happen!\n";
        return false;
    }
    {
        xmin=d->min().x+d->eleSize()*xminInt;//获得取整之后的坐标，也就是该点所在网格的角点
        ymin=d->min().y+d->eleSize()*yminInt;
        xmax=d->min().x+d->eleSize()*xmaxInt;
        ymax=d->min().y+d->eleSize()*ymaxInt;
        //cout<<xmin<<" "<<ymin<<" "<<xmax<<" "<<ymax<<" "<<endl;
    }

    //根据输入的图像帧frame.first，创建权重图像weightImage，并将其复制到src中
    cv::Mat src;
    if(weightImage.empty()||weightImage.cols!=frame.first.cols||weightImage.rows!=frame.first.rows)
    {
        std::shared_lock<shared_mutex> lock(mutex);
        int w=frame.first.cols;
        int h=frame.first.rows;
        weightImage.create(h,w,CV_8UC4);//创建了一个新的权重图像

        //计算中心点和最大距离
        uchar *p=(weightImage.data);
        float x_center=w/2;
        float y_center=h/2;
        float dis_max=sqrt(x_center*x_center+y_center*y_center);
        // int weightType=svar.GetInt("Map2D.WeightType",0);
        int weightType = 0;

        for(int i=0;i<h;i++)
            for(int j=0;j<w;j++)
            {
                float dis=(i-y_center)*(i-y_center)+(j-x_center)*(j-x_center);//计算每个像素距离中心的距离
                dis=1-sqrt(dis)/dis_max;//一种归一化距离的方式，从而计算权重。
                p[1]=p[2]=p[0]=0;//将前三个维度设为0，RGB
                if(0==weightType)//第四个维度为权重，通过距离计算得出
                    p[3]=dis*254.;
                else p[3]=dis*dis*254;
                if(p[3]<2) p[3]=2;
                p+=4;
            }
        src=weightImage.clone();//将其拷贝至src中
    }
    else
    {
        std::shared_lock<shared_mutex> lock(mutex);
        src = weightImage.clone();
    }

    //下面部分需要修改
    pi::Array_<pi::byte, 4> *psrc = (pi::Array_<pi::byte,4>*)src.data;
    pi::Array_<pi::byte, 3> *pimg = (pi::Array_<pi::byte,3>*)frame.first.data;

    for(int i = 0,iend = weightImage.cols * weightImage.rows ; i < iend ; i++)
    {
        *((pi::Array_<pi::byte,3>*)psrc)=*pimg;
        psrc++;
        pimg++;
    }

    //if(svar.GetInt("ShowSRC",0))
    if(0)
    {
        cv::namedWindow("src", cv::WINDOW_NORMAL);
        cv::resizeWindow("src", 800, 600);
        cv::imshow("src",src);
        //cv::waitKey(100);

    }

    //将原图src通过透视变换，变换到目标图dst上
    cv::Mat dst((ymaxInt-yminInt)*ELE_PIXELS,(xmaxInt-xminInt)*ELE_PIXELS,src.type());
    std::vector<cv::Point2f>   imgPtsCV;
    {
        imgPtsCV.reserve(imgPts.size());
        for(int i=0;i<imgPts.size();i++)
            imgPtsCV.push_back(cv::Point2f(imgPts[i].x,imgPts[i].y));//imgpts就是相机坐标系上的四个角点
    }
    std::vector<cv::Point2f> destPoints;
    destPoints.reserve(imgPtsCV.size());
    for(int i=0;i<imgPtsCV.size();i++)
    {
        destPoints.push_back(cv::Point2f((pts[i].x-xmin)*d->lengthPixelInv(),
                             (pts[i].y-ymin)*d->lengthPixelInv()));//对于四个角点，都以左上角为坐标，说明目标坐标是以网格左上角角点为基准的。
    }
    cv::Mat transmtx = cv::getPerspectiveTransform(imgPtsCV, destPoints);
    cv::warpPerspective(src, dst, transmtx, dst.size(),cv::INTER_LINEAR);

    // if(1)
    // {
        
    //     cv::namedWindow("dst", cv::WINDOW_NORMAL);
    //     cv::resizeWindow("dst", 800, 600);
    //     cv::imshow("dst",dst);
    //     cv::waitKey(1);
    // }

//将透视变换后的图像dst数据合并到一个2D地图的特定区域内，每个小块(element)表示地图的一部分.代码逐一遍历这些小块并根据条件将dst图像中的数据融合到对应的小块中。

    std::vector<std::shared_ptr<Map2DCPUEle> > dataCopy=d->data();
    
    for(int x = xminInt ; x <xmaxInt ; x++)
        for(int y = yminInt; y < ymaxInt;y++)
        {
            std::shared_ptr<Map2DCPUEle> ele=dataCopy[y*d->w()+x];
            if(!ele.get())
            {
                ele=d->ele(y*d->w()+x);
            }
            {
                std::unique_lock lock(ele->mutex_data);
                if(ele->img.empty())
                    ele->img=cv::Mat::zeros(ELE_PIXELS,ELE_PIXELS,dst.type());
                pi::Array_<pi::byte,4> *eleP=(pi::Array_<pi::byte,4>*)ele->img.data;
                pi::Array_<pi::byte,4> *dstP=(pi::Array_<pi::byte,4>*)dst.data;
                dstP+=(x-xminInt)*ELE_PIXELS+(y-yminInt)*ELE_PIXELS*dst.cols;
                int skip=dst.cols-ele->img.cols;
                for(int eleY=0;eleY<ELE_PIXELS;eleY++,dstP+=skip)
                    for(int eleX=0;eleX<ELE_PIXELS;eleX++,dstP++,eleP++)
                    {
                        if(eleP->data[3] < dstP->data[3])
                            *eleP=*dstP;
                    }
                ele->Ischanged=true;


            }            
        }
    
    //std::vector<shared_ptr<Map2DCPUEle> > dataCopy = d->data();
    //====================保存成点云代码
    // int wCopy = d->w(),hCopy = d->h();
    // for(int x = 0 ; x < wCopy ; x++)
    //     for(int y = 0; y <hCopy ;y++)
    //     {
    //         int idxData = y*wCopy+x;
    //         float x0 = d->min().x +x*d->eleSize();
    //         float y0 = d->min().y + y*d->eleSize();
    //         float x1 = x0 + d->eleSize();
    //         float y1 = y0 + d->eleSize();

    //         shared_ptr<Map2DCPUEle> ele = dataCopy[idxData];
    //         if(!ele.get()) continue;
    //         if(ele->img.empty()) continue;

    //         if(ele->Ischanged)
    //         {
    //             std::mutex mtx;
    //             std::lock_guard<std::mutex> lock(mtx);
    //             pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud = imageToPointCloud(ele->img, x0, y0);
    //             *globalpoint += *cloud;
    //             ele->Ischanged=false;
                
    //         }
    //     }
    // pcl::PointCloud<pcl::PointXYZRGB>::Ptr tmp(new pcl::PointCloud<pcl::PointXYZRGB>());
    // voxel.setInputCloud(globalpoint);
    // voxel.filter(*tmp);
    // globalpoint->swap(*tmp);
    // viewer.showCloud(globalpoint);
    //  cout << "show global map, size=" << globalpoint->points.size() << endl;
    //=============================

    //cout<<"111ss"<<endl;
    return true;
}

bool Map2DCPU::spreadMap(double xmin,double ymin,double xmax,double ymax)
{
    //pi::timer.enter("Map2DCPU::spreadMap");
    std::shared_ptr<Map2DCPUData> d;
    {
        std::shared_lock<shared_mutex> lock(mutex);
        d = data;
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
    std::vector<std::shared_ptr<Map2DCPUEle> > dataOld=d->data();
    std::vector<std::shared_ptr<Map2DCPUEle> > dataCopy;
    dataCopy.resize(w*h);
    {
        for(int x=0,xend=d->w();x<xend;x++)
            for(int y=0,yend=d->h();y<yend;y++)
            {
                dataCopy[x-xminInt+(y-yminInt)*w]=dataOld[y*d->w()+x];
            }
    }
    {
        std::unique_lock<shared_mutex> lock(mutex);
        data=std::shared_ptr<Map2DCPUData>(new Map2DCPUData(d->eleSize(),d->lengthPixel(),
                                                 cv::Point3d(max.x,max.y,d->max().z),
                                                 cv::Point3d(min.x,min.y,d->min().z),
                                                 w,h,dataCopy));
    }
    //pi::timer.leave("Map2DCPU::spreadMap");
    return true;
}

bool Map2DCPU::getFrame(std::pair<cv::Mat,Sophus::SE3d>& frame)
{
    //std::shared_lock<shared_mutex> lock(mutex);
    //std::shared_lock<shared_mutex> lock1(prepared->mutexFrames);
    //cout<<"getFrame: "<<prepared->_frames.size()<<" "<<endl;
    if(prepared->_frames.size())
    {
        frame=prepared->_frames.front();
        cout<<"prepared.deque size"<<prepared->_frames.size()<<endl;
        prepared->_frames.pop_front();
        return true;
    }
    else return false;
}

void Map2DCPU::run()
{
    std::pair<cv::Mat, Sophus::SE3d> frame;
    cout<<_valid<<"aaa"<<endl;
    //while(!shouldStop())
    while(1)
    {
        
        if(getFrame(frame))
        {
            renderFrame(frame);
        }
       
        // sleep(1);
    }
}
void Map2DCPU::draw2D()
{
    shared_ptr<Map2DCPUPrepare> p;
    shared_ptr<Map2DCPUData> d;
    {
        std::shared_lock<shared_mutex> lock(mutex);
        p = prepared;
        d = data;
    }
    // glEnable(GL_TEXTURE_2D);
    // glEnable(GL_BLEND);
    // if(alpha)
    // {
    //     glEnable(GL_ALPHA_TEST);//启用透明度测试。
    //     glAlphaFunc(GL_GREATER, 0.1f);//设置透明度函数，只有透明度值大于0.1的像素才会被绘制。
    //     glBlendFunc(GL_SRC_ALPHA,GL_ONE);//设置混合函数，使用源颜色的 alpha 值来进行混合。
    // }
    GLint last_texture_ID;
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &last_texture_ID);//获取当前绑定在目标的纹理对象名称，并存储到last_texture_id中
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f); // 白色背景
    glClear(GL_COLOR_BUFFER_BIT);  // 清除颜色缓冲区，应用背景色
    std::vector<shared_ptr<Map2DCPUEle> > dataCopy=d->data();
    int wCopy=d->w(),hCopy=d->h();
    glColor3ub(255,255,255);
    for(int x = 0; x <wCopy; x++)
        for(int y = 0; y<hCopy; y++)
        {
            int idxData = y*wCopy + x;

            float x0=d->min().x+x*d->eleSize();
            float y0=d->min().y+y*d->eleSize();
            float x1=x0+d->eleSize();
            float y1=y0+d->eleSize();

            shared_ptr<Map2DCPUEle> ele=dataCopy[idxData];//获得该元素块
            if(!ele.get())  continue;
            if(ele->img.empty()) continue;

            if(ele->texName==0)
            {
                //如果 ele->texName 为 0，则为当前元素生成一个新的纹理序号，第一个参数为生成纹理对象的数目
                glGenTextures(1, &ele->texName);
            }

            if(ele->Ischanged)
            {
                //如果当前元素发生了改变，则利用该元素生成平面里
                std::shared_lock<shared_mutex> lock(ele->mutex_data);
                glBindTexture(GL_TEXTURE_2D,ele->texName);
                
                //使用 glTexImage2D 更新纹理图像数据，设置纹理的宽度、高度、格式和数据指针
                glTexImage2D(GL_TEXTURE_2D, 0,
                                 GL_RGBA, ele->img.cols,ele->img.rows, 0,
                                 GL_BGRA, GL_UNSIGNED_BYTE,ele->img.data);
                
                //设置纹理参数，分别指定纹理放大和缩小过滤方式
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,  GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,GL_NEAREST);
                ele->Ischanged=false;
                //pi::timer.leave("glTexImage2D");
            }
            glBindTexture(GL_TEXTURE_2D, ele->texName);
            // glBegin(GL_QUADS);
            // glTexCoord2f(0.0f, 0.0f); glVertex3f(x0,y0,0);
            // glTexCoord2f(0.0f, 1.0f); glVertex3f(x0,y1,0);
            // glTexCoord2f(1.0f, 1.0f); glVertex3f(x1,y1,0);
            // glTexCoord2f(1.0f, 0.0f); glVertex3f(x1,y0,0);
            // glEnd();

            
            // float x_min = -1000;
            // float x_max = 1000;
            // float y_min = -1000;
            // float y_max = 500;
            float x_min = -5;
            float x_max = 5;
            float y_min = -5;
            float y_max = 5;
            float x0_n = 2*(x0 - x_min) / (x_max - x_min)-1;
            float x1_n = 2*(x1 - x_min) / (x_max - x_min)-1;

            float y0_n = 2*(y0 - y_min) / (y_max - y_min)-1;
            float y1_n = 2*(y1 - y_min) / (y_max - y_min)-1;
            glBegin(GL_QUADS);
            glTexCoord2f(0.0f, 0.0f); glVertex2f(x0_n,y0_n);
            glTexCoord2f(0.0f, 1.0f); glVertex2f(x0_n,y1_n);
            glTexCoord2f(1.0f, 1.0f); glVertex2f(x1_n,y1_n);
            glTexCoord2f(1.0f, 0.0f); glVertex2f(x1_n,y0_n);
            glEnd();

            //==============绘制到三维平面上
            // glBegin(GL_QUADS);
            // glTexCoord2f(0.0f, 0.0f); glVertex3f(x0,y0,0);
            // glTexCoord2f(0.0f, 1.0f); glVertex3f(x0,y1,0);
            // glTexCoord2f(1.0f, 1.0f); glVertex3f(x1,y1,0);
            // glTexCoord2f(1.0f, 0.0f); glVertex3f(x1,y0,0);
            // glEnd();
        }
        glBindTexture(GL_TEXTURE_2D, last_texture_ID);

}
void Map2DCPU::draw()
{
    if(!_valid) return;

    shared_ptr<Map2DCPUPrepare> p;
    shared_ptr<Map2DCPUData>    d;
    
    {
        std::shared_lock<shared_mutex> lock(mutex);
        p = prepared;
        d = data;
    }

    //定义模型视图矩阵模式
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();// 将当前的矩阵（模型视图矩阵、投影矩阵或纹理矩阵）保存到一个矩阵堆栈中，这样可以在后续的操作中临时修改矩阵而不会丢失原始的矩阵状态。

    std::cout << "Starting to draw map." << std::endl;
    //获取平面矩阵，并转换为OpenGL矩阵格式，并应用该矩阵
    Eigen::Matrix4d mat = p->_plane.matrix();
    GLdouble plane[16];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            plane[j * 4 + i] = mat(i, j);
        }
    }

    //     GLfloat plane1[16] = { // 一个简单的变换矩阵示例，这里是一个单位矩阵，即无变换效果
    //         2.0f, 0.0f, 0.0f, 0.0f,
    //         0.0f, 2.0f, 0.0f, 0.0f,
    //         0.0f, 0.0f, 1.0f, 0.0f,
    //         0.0f, 0.0f, 0.0f, 1.0f
    //     };
    glMultMatrixd(plane);
    glEnable(GL_BLEND);

    glBegin(GL_TRIANGLES);
    glColor3f(1.0f, 0.0f, 0.0f);   glVertex2f(-0.1f, -0.1f);
    glColor3f(0.0f, 1.0f, 0.0f);   glVertex2f(0.1f, -0.1f);
    glColor3f(0.0f, 0.0f, 1.0f);   glVertex2f(0.0f, 0.5f);
    glEnd();
    glPopMatrix();
    glFlush();
    //该段代码遍历所有的帧数据，并在 OpenGL 窗口中绘制每个帧的坐标轴。这些坐标轴以不同的颜色表示 X、Y 和 Z 轴，分别为红色、绿色和蓝色。
    {
        std::deque<std::pair<cv::Mat,Sophus::SE3d> > frames=p->getFrames();
        glDisable(GL_LIGHTING);
        glBegin(GL_LINES);
        for(std::deque<std::pair<cv::Mat,Sophus::SE3d> >::iterator it=frames.begin();it!=frames.end();it++)
        {
            Sophus::SE3d& pose=it->second;
            glColor3ub(255,0,0);
            glVertex3d(pose.translation()[0], pose.translation()[1], pose.translation()[2]);
            Eigen::Vector3d eigenPoint = Eigen::Vector3d(1,0,0);
            eigenPoint = pose * eigenPoint;
            glVertex3d(eigenPoint[0], eigenPoint[1], eigenPoint[2]);
            glColor3ub(0,255,0);
            glVertex3d(pose.translation()[0], pose.translation()[1], pose.translation()[2]);
            eigenPoint = Eigen::Vector3d(0,1,0);
            eigenPoint = pose * eigenPoint;
            glVertex3d(eigenPoint[0], eigenPoint[1], eigenPoint[2]);
            glColor3ub(0,0,255);
            glVertex3d(pose.translation()[0], pose.translation()[1], pose.translation()[2]);
            eigenPoint = Eigen::Vector3d(0,0,1);
            eigenPoint = pose * eigenPoint;
            glVertex3d(eigenPoint[0], eigenPoint[1], eigenPoint[2]);
        }
        glEnd();
        glPopMatrix();
        glFlush();
    }


    //代码在 OpenGL 窗口中绘制了一个红色的矩形框,表示为二维地图的边界。    
    {
        cv::Point3d _min = d->min();
        cv::Point3d _max = d->max();
        glColor3ub(255,0,0);
        glBegin(GL_LINES);
        glVertex3d(_min.x,_min.y,0);
        glVertex3d(_min.x,_max.y,0);
        glVertex3d(_min.x,_min.y,0);
        glVertex3d(_max.x,_min.y,0);
        glVertex3d(_max.x,_min.y,0);
        glVertex3d(_max.x,_max.y,0);
        glVertex3d(_min.x,_max.y,0);
        glVertex3d(_max.x,_max.y,0);
        glEnd();
    }


    glEnable(GL_TEXTURE_2D);//启用二维纹理功能，使得后续的绘制操作能够使用纹理。
    glEnable(GL_BLEND);//启用混合功能，使得可以进行透明度处理。

    //如果 alpha 变量为真，则启用透明度测试并设置透明度函数和混合函数：
    if(alpha)
    {
        glEnable(GL_ALPHA_TEST);//启用透明度测试。
        glAlphaFunc(GL_GREATER, 0.1f);//设置透明度函数，只有透明度值大于0.1的像素才会被绘制。
        glBlendFunc(GL_SRC_ALPHA,GL_ONE);//设置混合函数，使用源颜色的 alpha 值来进行混合。
    }

    //保存当前绑定的纹理 ID，以便在操作完成后恢复它。
    GLint last_texture_ID;
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &last_texture_ID);

    //从 d 对象中获取数据副本以及宽度和高度。设置颜色为白色，确保绘制时不会对纹理颜色进行修改。
    std::vector<shared_ptr<Map2DCPUEle> > dataCopy=d->data();
    int wCopy=d->w(),hCopy=d->h();
    glColor3ub(255,255,255);

    
    for(int x=0;x<wCopy;x++)
        for(int y=0;y<hCopy;y++)
        {
            int idxData=y*wCopy+x;
            
            //计算当前元素块在地图中的左上角(x0,y0),右下角(x1,y1)的坐标。
            float x0=d->min().x+x*d->eleSize();
            float y0=d->min().y+y*d->eleSize();
            float x1=x0+d->eleSize();
            float y1=y0+d->eleSize();

            shared_ptr<Map2DCPUEle> ele=dataCopy[idxData];//获得该元素块
            if(!ele.get())  continue;
            if(ele->img.empty()) continue;

            if(ele->texName==0)
            {
                //如果 ele->texName 为 0，则为当前元素生成一个新的纹理序号
                glGenTextures(1, &ele->texName);
            }
            //if(ele->Ischanged&&ticTac.Tac()<0.02)
            if(ele->Ischanged)
            {
                //如果当前元素发生了改变，则利用该元素生成平面里
                std::shared_lock<shared_mutex> lock(ele->mutex_data);
                glBindTexture(GL_TEXTURE_2D,ele->texName);
                
                //使用 glTexImage2D 更新纹理图像数据，设置纹理的宽度、高度、格式和数据指针
                glTexImage2D(GL_TEXTURE_2D, 0,
                                 GL_RGBA, ele->img.cols,ele->img.rows, 0,
                                 GL_BGRA, GL_UNSIGNED_BYTE,ele->img.data);
                
                //设置纹理参数，分别指定纹理放大和缩小过滤方式
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,  GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,GL_NEAREST);
                ele->Ischanged=false;
                //pi::timer.leave("glTexImage2D");
            }

            //绘制纹理矩形
            glBindTexture(GL_TEXTURE_2D,ele->texName);
            glBegin(GL_QUADS);
            glTexCoord2f(0.0f, 0.0f); glVertex3f(x0,y0,0);
            glTexCoord2f(0.0f, 1.0f); glVertex3f(x0,y1,0);
            glTexCoord2f(1.0f, 1.0f); glVertex3f(x1,y1,0);
            glTexCoord2f(1.0f, 0.0f); glVertex3f(x1,y0,0);
            glEnd();
        }
    glBindTexture(GL_TEXTURE_2D, last_texture_ID);
    glPopMatrix();
}

bool Map2DCPU::save(const std::string& filename)
{
    
    std::shared_ptr<Map2DCPUPrepare> p;
    std::shared_ptr<Map2DCPUData> d;
    {
        // std::shared_lock<shared_mutex> lock(mutex);
        p=prepared;d=data;
    }
    if(d->w()==0||d->h()==0)
    {
        cout<<"2111";
        return false;
    } 
    cv::Point2i minInt(1e6,1e6),maxInt(-1e6,-1e6);
    for(int x=0;x<d->w();x++)
        for(int y=0;y<d->h();y++)
        {
            std::shared_ptr<Map2DCPUEle> ele=d->data()[x+y*d->w()];
            if(!ele.get()) continue;
            {
                std::shared_lock<shared_mutex> lock(ele->mutex_data);
                if(ele->img.empty()) continue;
            }
            minInt.x=min(minInt.x,x); minInt.y=min(minInt.y,y);
            maxInt.x=max(maxInt.x,x); maxInt.y=max(maxInt.y,y);
        }

    maxInt=maxInt+cv::Point2i(1,1);
    cv::Point2i wh=maxInt-minInt;
    cv::Mat result(wh.y*ELE_PIXELS,wh.x*ELE_PIXELS,CV_8UC4);
    for(int x=minInt.x;x<maxInt.x;x++)
        for(int y=minInt.y;y<maxInt.y;y++)
        {
            std::shared_ptr<Map2DCPUEle> ele=d->data()[x+y*d->w()];
            
            if(!ele.get()) continue;
            {
                if(!ele->img.data) continue;
                std::shared_lock<shared_mutex> lock(ele->mutex_data);
                ele->img.copyTo(result(cv::Rect(ELE_PIXELS*(x-minInt.x),ELE_PIXELS*(y-minInt.y),ELE_PIXELS,ELE_PIXELS)));
            }
        }

    
    cv::imwrite(filename,result);
    return true;
}