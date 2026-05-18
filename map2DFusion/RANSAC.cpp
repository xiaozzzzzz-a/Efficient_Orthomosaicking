#include <cmath>
#include <ctime>
#include "RANSAC.h"
#include "DataTrans.h"
#include "opencv2/opencv.hpp"
#include "eigen3/Eigen/Geometry"

using namespace Eigen;

RANSAC::RANSAC()
{
    this->points.clear();
    this->finished = false;
}

double RANSAC::solve_distance(cv::Point3d M, cv::Point3d P, cv::Point3d N)//计算点M到平面的距离
{
    double A = N.x;
    double B = N.y;
    double C = N.z;
    double D = -A * P.x - B * P.y - C * P.z;

    return fabs(A * M.x + B * M.y + C * M.z + D) / sqrt(A * A + B * B + C * C);
}


cv::Point3d normalize(const cv::Point3d& p) {
    double length = std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
    if (length > 0) {
        return cv::Point3d(p.x / length, p.y / length, p.z / length);
    } else {
        // 向量长度为零时返回零向量
        return cv::Point3d(0, 0, 0);
    }
}

cv::Point3d crossProduct(const cv::Point3d& a, const cv::Point3d& b) {
    return cv::Point3d(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

double dotProduct(const cv::Point3d& a, const cv::Point3d& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

void RANSAC::solve_plane(cv::Point3d A, cv::Point3d B, cv::Point3d C)
{
    //定义两个常量
    const double pi = 3.1415926535;
    cv::Point3d N(0, 0, 1);

    //计算平面的单位法向量，即BC与BA的叉积
    cv::Point3d Nx = crossProduct((B - C) , (B - A));
    Nx = normalize(Nx);

    //计算单位旋转向量与旋转角(范围0到pi)
    cv::Point3d Nv = crossProduct(Nx , N);
    double angle = acos(dotProduct(Nx , N));

    //两个向量的夹角不大于pi/2,这里单独处理一下
    if (angle > pi / 2.0)
    {
        angle = pi - angle;
        Nv = Nv * (-1);
    }

    //利用私有变量把返回值保存出来,不太优美...
    this->plane_P = B;
    //pi::SO3d根据direction和angle初始化好像不太对，这里只能换一个形式
    //this->plane_Q = pi::SO3d(Nv * sin(angle / 2), angle);
    //std::cout<<"sss1ss"<<std::endl;
    // Eigen::Matrix3d R1 = AngleAxisd(M_PI/2, Vector3d(0,0,1)).matrix();
    // Sophus::SO3d so3d1(R1);
    // Eigen::Matrix3d R = AngleAxisd(angle, Vector3d(Nv.x, Nv.y, Nv.z).normalized()).matrix();
    // Sophus::SO3d so3d(R);
    // this->plane_Q = so3d;
    Nv = Nv * sin(angle / 2);
    Eigen::Vector3d eigenNv = Eigen::Vector3d(Nv.x, Nv.y, Nv.z);
    Eigen::Vector3d rotation_vector = eigenNv * cos(angle / 2);
    this->plane_Q = Sophus::SO3d::exp(rotation_vector);
    this->plane_N = Nx;
}

void RANSAC::ransac_core()
{
    //数据规模
    unsigned long size = this->points.size();
    //迭代的最大次数,每次得到更好的估计会优化iters的数值,默认10000
    int iters = 10000;
    //数据和模型之间可接受的差值,默认0.25
    //double sigma = 0.25;

    double sigma = 1;
    //内点数目
    int pretotal = 0;
    //希望的得到正确模型的概率,默认0.99
    double per = 0.99;
    srand((unsigned) time(nullptr));
    for (int i = 0; i < iters; i++)
    {
        //随机从数据中算则三个点去求解模型,这里取得点可能是重复的
        int index[3];
        
        index[0] = rand() % (size);
        index[1] = rand() % (size);
        index[2] = rand() % (size);
        //如果取的点是重复的,就重新取一次
        if (index[0] == index[1] || index[1] == index[2] || index[2] == index[0])
        {
            i--;
            continue;
        }
        std::cout<<index[0]<<" "<<index[1]<<" "<<index[2]<<" "<<rand()<<std::endl;
        solve_plane(this->points[index[0]], this->points[index[1]], this->points[index[2]]);

        //计算内点的数目
        int total_inlier = 0;
        for (int j = 0; j < size; j++)
        {
            if (solve_distance(points[j], this->plane_P, this->plane_N) < sigma)
            {
                total_inlier = total_inlier + 1;
            }
        }

        //std::cout<<"total_inlier: "<<total_inlier<< "pretotal: "<<pretotal<<std::endl;

        //判断当前的模型是否比之前估算的模型更好
        if (total_inlier > pretotal)
        {
            // std::cout<<log(1.0 - per)<<std::endl;
            // std::cout<<log(1.0 - pow(total_inlier / size, 2))<<std::endl;
            // std::cout<<"pow(total_inlier / size, 2): "<<pow(float(total_inlier) / size, 2)<<std::endl;
            // std::cout<<total_inlier<<" "<<size<<std::endl;
            // std::cout<<"pow(total_inlier / size, 2): "<<(total_inlier / size)*(total_inlier / size)<<std::endl;
            // std::cout<<total_inlier/size<<std::endl;
            // std::cout<<"size: "<<size<<std::endl;
            iters = int(log(1.0 - per) / log(1.0 - pow(float(total_inlier) / size, 2)));
            std::cout<<"iters: "<<iters<<std::endl;
            pretotal = total_inlier;
        }

        //std::cout<<"test mdoelsla "<<std::endl;

        //判断是否当前模型已经符合超过一半的点
        if (total_inlier > size / 1.5)
        {   std::cout<<"total_inlier_size > 1.75"<<std::endl;
            break;
        }
    }
}

void RANSAC::improve_ransac()
{
    unsigned int nPoints = points.size();
    int nRansacs =100;
    double thresholdZ = 0.05;
    Eigen::Vector3d bestMean = Eigen::Vector3d::Zero();
    Eigen::Vector3d bestNormal = Eigen::Vector3d::Zero();
    double bestError = std::numeric_limits<double>::max();
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> dist(0, nPoints - 1);
    std::vector<int> outliers;
    for(int i=0; i<nRansacs; i++)
    {
        int idxA = dist(rng), idxB = idxA, idxC = idxA;
        while(idxB == idxA) idxB = dist(rng);
        while(idxC == idxA || idxC == idxB) idxC = dist(rng);

        Eigen::Vector3d A(points[idxA].x, points[idxA].y, points[idxA].z);
        Eigen::Vector3d B(points[idxB].x, points[idxB].y, points[idxB].z);
        Eigen::Vector3d C(points[idxC].x, points[idxC].y, points[idxC].z);

        Eigen::Vector3d mean = (A + B + C) / 3.0;
        Eigen::Vector3d normal = (C - A).cross(B - A);

        if (normal.norm() < 1e-8) continue;
        normal.normalize();

        double sumError = 0.0;
        for (const auto& p : points)
        {
            Eigen::Vector3d P(p.x, p.y, p.z);
            double distZ = std::abs((P - mean).dot(normal));
            if (distZ > thresholdZ) distZ = thresholdZ;
            sumError += distZ;
        }

        if (sumError < bestError)
        {
            bestError = sumError;
            bestMean = mean;
            bestNormal = normal;
        }
       
    }

    std::vector<Eigen::Vector3d> inliers;
    outliers.clear();
    inliers.reserve(nPoints);
    outliers.reserve(nPoints);

    for (size_t i = 0; i < nPoints; ++i)
    {
        Eigen::Vector3d P(points[i].x, points[i].y, points[i].z);
        double distZ = std::abs((P - bestMean).dot(bestNormal));
        if (distZ < thresholdZ)
            inliers.push_back(P);
        else
            outliers.push_back(static_cast<int>(i));
    }

    if (inliers.size() < 3)
    {
        std::cout << "[WARN] Too few inliers after RANSAC." << std::endl;
        return ;
    }

    Eigen::Vector3d meanInliers = Eigen::Vector3d::Zero();
    for (const auto& p : inliers) meanInliers += p;
    meanInliers /= inliers.size();

    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    for (const auto& p : inliers)
    {
        Eigen::Vector3d d = p - meanInliers;
        cov += d * d.transpose();
    }

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov);
    Eigen::Vector3d normal = solver.eigenvectors().col(0);
    if (normal.z() < 0) normal = -normal;

    // ============================
    // 4️⃣ Build SE3 pose (x,y,z axes)
    // ============================
    Eigen::Vector3d vz = normal;
    Eigen::Vector3d vx = vz.cross(Eigen::Vector3d(0, -1, 0));
    if (vx.norm() < 1e-8)
        vx = vz.cross(Eigen::Vector3d(-1, 0, 0));
    vx.normalize();
    Eigen::Vector3d vy = vz.cross(vx);

    Eigen::Matrix3d R;
    R.col(0) = vx;
    R.col(1) = vy;
    R.col(2) = vz;

    this->plane_P = cv::Point3d(meanInliers.x(), meanInliers.y(), meanInliers.z());
    this->plane_Q = Sophus::SO3d(R);
    


}
void RANSAC::solve(cv::Point3d point3D)
{
    //将目前的点存入
    this->points.emplace_back(point3D);
   
    if (this->points.size() < 1500) //1000
        return;

    std::cout<<"Ransac point num:"<< this->points.size();



    this->improve_ransac();
    
    this->finished = true;
    Eigen::Vector3d eigen_P = Eigen::Vector3d(this->plane_P.x, this->plane_P.y, this->plane_P.z);
    
    Trans_Plane.product(Sophus::SE3d(this->plane_Q, eigen_P));
    std::vector<double> planeVec;
    planeVec.push_back(this->plane_P.x);
    planeVec.push_back(this->plane_P.y);
    planeVec.push_back(this->plane_P.z);


    planeVec.push_back(this->plane_Q.unit_quaternion().x());
    planeVec.push_back(this->plane_Q.unit_quaternion().y());
    planeVec.push_back(this->plane_Q.unit_quaternion().z());
    planeVec.push_back(this->plane_Q.unit_quaternion().w());

    Trans_PlaneVec.product(planeVec);
    std::cout<<"plane so3: "<<this->plane_Q.unit_quaternion().coeffs().transpose()<<std::endl;
    std::cout<<"plane transtition: "<<eigen_P<<std::endl;
}

bool RANSAC::is_finished() const 
{
    return this->finished;
}

RANSAC &RANSAC::Instance()
{
    static RANSAC instance;
    return instance;
}