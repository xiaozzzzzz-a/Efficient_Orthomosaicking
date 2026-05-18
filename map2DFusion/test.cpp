#include <opencv2/opencv.hpp>
#include <opencv2/stitching/detail/blenders.hpp>

int main() {
    // 假设已有 lappyr 和 lowResImg
    std::vector<cv::Mat> lappyr;
    cv::Mat lowResImg;
    cv::Mat dst;

    // 从拉普拉斯金字塔恢复图像
    cv::detail::restoreImageFromLaplacePyr(lappyr, lowResImg, dst);

    // 显示结果
    cv::imshow("Restored Image", dst);
    cv::waitKey(0);

    return 0;
}
