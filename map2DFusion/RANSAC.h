#ifndef RANSAC_RANSAC_H
#define RANSAC_RANSAC_H

#include <string>
#include <vector>
#include <cmath>
#include <sophus/se3.hpp>
#include <opencv2/opencv.hpp>

#define ransac RANSAC::Instance()

class RANSAC
{
public:
    // 获取单实例对象
    static RANSAC &Instance();

    //进行求解
    void solve(cv::Point3d point3D);

    //是否完成计算
    bool is_finished() const;
    bool begin = false;

public:
    RANSAC();

    /**
     * 求解点M到平面的距离
     * @param M 点M
     * @param P 平面上一点P
     * @param N 平面的法向量
     */
    static double solve_distance(cv::Point3d M, cv::Point3d P, cv::Point3d N);

    /**
     * 根据三点求解平面方程
     * @param A 点A
     * @param B 点B
     * @param C 点C
     */
    void solve_plane(cv::Point3d A, cv::Point3d B, cv::Point3d C);

    /**
     * RANSAC核心算法
     */
    void ransac_core();
    void improve_ransac();

public:
    //保存的点集
    std::vector<cv::Point3d> points;
    //计算得到的平面上的一点
    cv::Point3d plane_P;
    //计算得到的平面的四元数
    Sophus::SO3d plane_Q;
    //计算得到的平面的法向量
    cv::Point3d plane_N;
    //是否完成计算
    bool finished;
};

#endif //RANSAC_RANSAC_H
