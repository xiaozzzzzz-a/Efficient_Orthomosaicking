1#include <iostream>
#include <fstream>
#include <sophus/so3.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include "Map2D.h"
#include <unistd.h>
// #include <yaml-cpp/yaml.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>


using namespace std;
class TrajectoryLengthCalculator
{

public:
    TrajectoryLengthCalculator():length(-1){}
    ~TrajectoryLengthCalculator()
    {
        cout<<"TrajectoryLength:"<<length<<endl;
    }

    void feed(cv::Point3d position)
    {
        if(length < 0)
        {
            length = 0;
            lastPosition = position;
        }
        else
        {
            length+=cv::norm(position-lastPosition);
            lastPosition=position;
        }
    }

    

private:
    double      length;
    cv::Point3d lastPosition;

};



class TestSystem
{

public:
    TestSystem()
    {
        
        datapath = "/media/xiao/data2/image_stitching/myMap2DFusion/config.yaml";
        //config = YAML::LoadFile(datapath+"/config.yaml");

        config = cv::FileStorage(datapath.c_str(), cv::FileStorage::READ);
    }

    struct TextureElement {
            GLuint texName;
            cv::Mat img;
    };



    ~TestSystem()
    {
        stop();
        
        if(map.get())
            map->save("resultss.png");
    }



    void stop()
    {
        myStopFLag = true;
    }

    bool isRunning() const
    {
        return myRunningFlag;
    }

    

    bool obtainFrame(std::pair<cv::Mat,Sophus::SE3d>& frame)
    {
        string line;
        if(!getline(*in, line)) return false;
        stringstream ifs(line);
        string imgfile;
        ifs>>imgfile;
        imgfile = datapath+"/rgb/"+imgfile+".jpg";
        //pi::timer.enter("obtainFrame");
        frame.first=cv::imread(imgfile);
        if(frame.first.empty()) return false;

        double x,y,z;
        double rx,ry,rz,rw;
        ifs>>x >> y >> z >> rx >> ry >> rz >> rw;
        Eigen::Quaterniond rotation(rw, rx, ry, rz);
        Eigen::Vector3d translation(x, y, z);
        frame.second.setQuaternion(rotation);
        frame.second.translation() = translation;

        if(gps_origin)
        {
            if(!lengthCalculator.get()) lengthCalculator=std::shared_ptr<TrajectoryLengthCalculator>(
                        new TrajectoryLengthCalculator());
        }

        return true;
    }

    int testMap2D()
    {
        cout<<"Act=TestMap2D\n";

        //datapath=svar.GetString("Map2D.DataPath","");
        datapath = "/media/xiao/data1/datasets_machine/phantom3-npu/";//修改
        if(!datapath.size())
        {
            cerr<<"Map2D.DataPath is not seted!\n";
            return -1;
        }
        
        if(config["Plane"].empty())
        {
            cerr<<"Plane is not defined!\n";
            return -2;
        }

        if(!in.get())
            in = std::shared_ptr<ifstream>(new ifstream((datapath+"/trajectory_slam.txt").c_str()));

        if(!in->is_open())
        {
            cerr<<"Can't open file "<<(datapath+"/trajectory.txt")<<endl;
            return -3;
        }
        deque<std::pair<cv::Mat,Sophus::SE3d>> frames;
        int iEnd ;
        config["PrepareFrameNum"] >> iEnd;
        for(int i = 0, iend=iEnd;  i<iend; i++)
        //for(int i = 0, iend=10;  i<iend; i++)
        {
            std::pair<cv::Mat, Sophus::SE3d> frame;
            if(!obtainFrame(frame)) break;
            frames.push_back(frame);
        }
        cout<<"Loaded "<<frames.size()<<" frames.\n";

        if(!frames.size()) return -4;
        int type;
        config["Map2D.Type"]>>type;
        map = Map2D::create(type);

        if(!map.get())
        {
            cerr<<"No map2d created!\n";
            return -5;
        }

        //  提取config文件中的se(3)
        std::vector<double> plane;
        config["Plane"] >> plane;

        //std::vector<double> plane = config["Plane"].as<std::vector<double>>();
        Eigen::Quaterniond quat(plane[3], plane[4], plane[5], plane[6]);
        Eigen::Vector3d trans(plane[0], plane[1], plane[2]);
        Sophus::SE3d se3(quat, trans);
        
        //std::vector<double> vecP = config["Camera.Paraments"].as<std::vector<double>>();//此处config为数据集的配置文件
        std::vector<double> vecP;
        config["Camera.Paraments"] >>vecP;
        map->prepare(se3, PinHoleParameters(vecP[0],vecP[1],vecP[2],vecP[3],vecP[4],vecP[5]),
                    frames);
        int autofeedframes;
        config["AutoFeedFrames"] >> autofeedframes;
        if(autofeedframes == 1)
        {

            if(!glfwInit())
            {
                std::cerr << "Failed to initialize GLFW" << std::endl;
            }

            GLFWwindow* window = glfwCreateWindow(2200, 1600, "2D Map Stitching", NULL, NULL);

            if(!window) 
            {
                std::cerr << "Failed to create GLFW window" << std::endl;
                glfwTerminate();
            }

            glfwMakeContextCurrent(window);
            glewExperimental = GL_TRUE;
            if (glewInit() != GLEW_OK) 
            {
                std::cerr << "Failed to initialize GLEW" << std::endl;
            }
            glEnable(GL_TEXTURE_2D);
            while (!glfwWindowShouldClose(window)) 
            {
                if(map->queueSize()<2)
                {
                    std::pair<cv::Mat, Sophus::SE3d> frame;
                    if(!obtainFrame(frame)) break;
                    map->feed(frame.first, frame.second);
                }
                map->draw2D();
                glBindTexture(GL_TEXTURE_2D, 0);
                glfwSwapBuffers(window);
                glfwPollEvents();
            }
        }
    }

    virtual void run()
    {
        string act;
        config["Act"] >> act;
        //string act = config["Act"].as<string>();
        if(act == "TestMap2D" || act == "Default") testMap2D();
    }

    string        datapath;
    std::shared_ptr<ifstream>      in;
    std::shared_ptr<Map2D>       map;
    std::shared_ptr<TrajectoryLengthCalculator> lengthCalculator;
    bool myStopFLag = false;
    bool myRunningFlag = true;
    bool gps_origin = true;
    cv::FileStorage config;


    

};




int main(int argc, char** argv)
{
    TestSystem sys;
    sys.run();


    // glutInit(&argc, argv);
    // glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
    // glutInitWindowSize(800, 600);
    // glutCreateWindow("OpenGL Example");
    // init();
    // glutDisplayFunc(TestSystem::displayCallback);
    // glutMainLoop();
    return 0;

}