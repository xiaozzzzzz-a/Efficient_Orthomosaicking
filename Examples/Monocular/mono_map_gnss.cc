/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include<cmath>
#include<limits>
#include<exiv2/exiv2.hpp>
#include<opencv2/core/core.hpp>
#include<System.h>

#include "../../map2DFusion/Map2D.h"
#include "../../map2DFusion/Map2DCPU.h"
#include "../../map2DFusion/Map2DFusion.h"

using namespace std;

void LoadImages(const string &strFile, vector<string> &vstrImageFilenames,
                vector<double> &vTimestamps);


double dmsToDecimal(const Exiv2::Rational& degrees, const Exiv2::Rational& minutes, const Exiv2::Rational& seconds, const std::string& direction) {
    double decDegrees = degrees.first / static_cast<double>(degrees.second);
    double decMinutes = minutes.first / static_cast<double>(minutes.second);
    double decSeconds = static_cast<double>(seconds.first) / static_cast<double>(seconds.second);
    //cout<<(double)seconds.first<<" "<<(double)seconds.second<<endl;
    //std::cout<<"decSecond: "<<decSeconds / 3600.0<<std::endl;
    //std::cout<<"decMinutes: "<<decMinutes / 60.00<<std::endl;
    double decimal = decDegrees + (decMinutes / 60.00) + (decSeconds / 3600.0);
    // 根据方向，南纬和西经需要取负值
    if (direction == "S" || direction == "W") {
        decimal = -decimal;
    }

    return decimal;
}

struct GpsData {
    std::string imageName;
    double timestamp;
    double latitude;
    double longitude;
    double altitude;

    GpsData(const std::string &img, double ts, double lat, double lon, double alt)
        : imageName(img), timestamp(ts), latitude(lat), longitude(lon), altitude(alt) {}
};

bool LoadGPSDataFromFile(const std::string &gpsFilePath, std::vector<GpsData> &gpsDataVector) {
    std::ifstream file(gpsFilePath);
    if (!file.is_open()) {
        std::cerr << "Failed to open GPS file: " << gpsFilePath << std::endl;
        return false;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string imageName;
        double timestamp, latitude, longitude, altitude;

        std::vector<std::string> tokens;
        std::string token;
        std::stringstream ss(line);
        while(std::getline(ss, token, ' ')){
            tokens.push_back(token);
        }

        if(tokens.size() < 5){
            std::cerr << "Invalid GPS data in line: " << line << std::endl;
            continue;
        }

        try{
            imageName = tokens[1];
            timestamp = std::stod(tokens[0]);
            longitude = std::stod(tokens[2]);
            latitude = std::stod(tokens[3]);
            altitude = std::stod(tokens[4]);

            // imageName = tokens[0];
            // timestamp = std::stod(tokens[1]);
            // longitude = std::stod(tokens[2]);
            // latitude = std::stod(tokens[3]);
            // altitude = std::stod(tokens[4]);
            
            gpsDataVector.emplace_back(imageName, timestamp, latitude, longitude, altitude);
        }catch (const std::exception &e) {
            std::cerr << "Error parsing line: " << line << " (" << e.what() << ")" << std::endl;
            continue;
        }
        
    }

    file.close();
    return true;
}

const GpsData* FindGpsDataForFrame(const std::vector<GpsData>& gpsDataVector,
                                   const std::string& imageName,
                                   const double timestamp,
                                   size_t& gpsCursor)
{
    if(gpsDataVector.empty())
        return nullptr;

    while(gpsCursor < gpsDataVector.size() &&
          gpsDataVector[gpsCursor].timestamp + 1e-6 < timestamp)
    {
        if(gpsDataVector[gpsCursor].imageName == imageName)
            return &gpsDataVector[gpsCursor];
        gpsCursor++;
    }

    for(size_t i = gpsCursor; i < gpsDataVector.size(); ++i)
    {
        if(gpsDataVector[i].timestamp > timestamp + 0.5)
            break;
        if(gpsDataVector[i].imageName == imageName)
        {
            gpsCursor = i;
            return &gpsDataVector[i];
        }
    }

    size_t bestIdx = gpsCursor;
    double bestDt = std::numeric_limits<double>::max();
    const size_t begin = gpsCursor > 2 ? gpsCursor - 2 : 0;
    const size_t end = std::min(gpsDataVector.size(), gpsCursor + 3);
    for(size_t i = begin; i < end; ++i)
    {
        const double dt = std::abs(gpsDataVector[i].timestamp - timestamp);
        if(dt < bestDt)
        {
            bestDt = dt;
            bestIdx = i;
        }
    }

    if(bestDt <= 0.03)
    {
        gpsCursor = bestIdx;
        return &gpsDataVector[bestIdx];
    }

    return nullptr;
}

Eigen::Matrix3d CreateGpsCovariance(ORB_SLAM3::Settings* settings)
{
    const double sigma_xy = settings ? settings->gpsSigmaXY() : 2.0;
    const double sigma_z = settings ? settings->gpsSigmaZ() : 5.0;

    Eigen::Matrix3d covariance;
    covariance.setZero();
    covariance(0, 0) = sigma_xy * sigma_xy;
    covariance(1, 1) = sigma_xy * sigma_xy;
    covariance(2, 2) = sigma_z * sigma_z;
    return covariance;
}

int main(int argc, char **argv)
{
    if(argc != 4)
    {
        cerr << endl << "Usage: ./mono_tum path_to_vocabulary path_to_settings path_to_sequence" << endl;
        return 1;
    }

    // Retrieve paths to images

    cout<<"ssss"<<endl;
    vector<string> vstrImageFilenames;
    vector<double> vTimestamps;
    string strFile = string(argv[3])+"/rgb_sim.txt";
    //string strFile = string(argv[3])+"/rgb.txt";
    LoadImages(strFile, vstrImageFilenames, vTimestamps);

    int nImages = vstrImageFilenames.size();
    //启动map2dfusion线程
    Map2DFusion::TestSystem mapsys;
    std::thread* map = new thread(&Map2DFusion::TestSystem::run, mapsys);
    
    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM3::System SLAM(argv[1],argv[2],ORB_SLAM3::System::MONOCULAR,true);

    
    float imageScale = SLAM.GetImageScale();

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    double t_resize = 0.f;
    double t_track = 0.f;

    // Main loop
    cv::Mat im;

    bool bUsefile = true;
   
    vector<GpsData> gpsDataVector;
    if(bUsefile)
    {
        
        
        string gpsFilePath = string(argv[3])+"/frames_with_gps_aligned.txt";
        ifstream alignedFile(gpsFilePath);
        if(!alignedFile.is_open())
        {
            gpsFilePath = string(argv[3])+"/frames_with_gps.txt";
        }
        if (!LoadGPSDataFromFile(gpsFilePath, gpsDataVector)) {
        std::cerr << "Failed to load GPS data from file." << std::endl;
    }
    }
    size_t gpsCursor = 0;
    for(int ni=0; ni<nImages; ni++)
    {
        ORB_SLAM3::GlobalPosition::GpsMeasurement *g;
        
        
        if(!bUsefile)
        {
            im = cv::imread(string(argv[3])+"/"+vstrImageFilenames[ni],cv::IMREAD_UNCHANGED); //,cv::IMREAD_UNCHANGED);

            //读取图片的gps数据
            double latitude, longitude, altitude;
            Exiv2::Image::AutoPtr im_exiv = Exiv2::ImageFactory::open(string(argv[3])+"/"+vstrImageFilenames[ni]);
            //Exiv2::Image::AutoPtr im_exiv = Exiv2::ImageFactory::open("/media/xiao/data2/image_stitching/Reslut/2023_03_06-09-50-16-68850m-119-5ms/00002.jpg");
            
            im_exiv->readMetadata();
            Exiv2::ExifData &exifData = im_exiv->exifData();
            if (exifData.empty()) {
                throw Exiv2::Error(Exiv2::kerErrorMessage, string(argv[3])+"/"+vstrImageFilenames[ni],cv::IMREAD_UNCHANGED); + ": No EXIF data found in the file";
            }
            auto gpsLatRef = exifData.findKey(Exiv2::ExifKey("Exif.GPSInfo.GPSLatitudeRef"));
            auto gpsLat = exifData.findKey(Exiv2::ExifKey("Exif.GPSInfo.GPSLatitude"));
            auto gpsLonRef = exifData.findKey(Exiv2::ExifKey("Exif.GPSInfo.GPSLongitudeRef"));
            auto gpsLon = exifData.findKey(Exiv2::ExifKey("Exif.GPSInfo.GPSLongitude"));
            auto gpsAlt = exifData.findKey(Exiv2::ExifKey("Exif.GPSInfo.GPSAltitude"));
            if (gpsLatRef != exifData.end() && gpsLat != exifData.end() &&
                gpsLonRef != exifData.end() && gpsLon != exifData.end() && gpsAlt != exifData.end()) {

                // 提取纬度和经度的度、分、秒
                Exiv2::Rational latDegrees = gpsLat->value().toRational(0);
                Exiv2::Rational latMinutes = gpsLat->value().toRational(1);
                Exiv2::Rational latSeconds = gpsLat->value().toRational(2);
                Exiv2::Rational lonDegrees = gpsLon->value().toRational(0);
                Exiv2::Rational lonMinutes = gpsLon->value().toRational(1);
                Exiv2::Rational lonSeconds = gpsLon->value().toRational(2);
                //std::cout<<latSeconds.first<<" "<<latSeconds.second<<" "<<std::endl;
                // 获取纬度和经度的方向 (N/S, E/W)
                std::string latDirection = gpsLatRef->value().toString();
                std::string lonDirection = gpsLonRef->value().toString();

                // 转换为十进制格式
                latitude = dmsToDecimal(latDegrees, latMinutes, latSeconds, latDirection);
                longitude = dmsToDecimal(lonDegrees, lonMinutes, lonSeconds, lonDirection);
                Exiv2::Rational altitudeValue = gpsAlt->value().toRational(0);
                altitude = static_cast<double>(altitudeValue.first) / altitudeValue.second;

                std::cout << std::fixed << std::setprecision(6); 
                // 显示十进制格式的 GPS 信息
                //std::cout << "Latitude (Decimal): " << latitude << "Longitude (Decimal): " << longitude << "altitude: "<<altitude<<std::endl;
            } else {
                std::cout << "No GPS information found in the image." << std::endl;
            }

            Eigen::Matrix3d covariance = CreateGpsCovariance(SLAM.settings_);
            g = new ORB_SLAM3::GlobalPosition::GpsMeasurement(ni, latitude, longitude, altitude, vTimestamps[ni], covariance);
        }
        // Read image from file
       
        else if(bUsefile){
            const GpsData* gpsData = FindGpsDataForFrame(gpsDataVector,
                                                         vstrImageFilenames[ni],
                                                         vTimestamps[ni],
                                                         gpsCursor);
            if(!gpsData)
            {
                cerr << "No matching GPS data for image index " << ni
                     << " timestamp " << vTimestamps[ni]
                     << " image " << vstrImageFilenames[ni] << endl;
                return 1;
            }
            im = cv::imread(string(argv[3])+"/"+vstrImageFilenames[ni],cv::IMREAD_UNCHANGED);
            double latitude = gpsData->latitude;
            double longitude = gpsData->longitude;
            double altitude = gpsData->altitude;
            Eigen::Matrix3d covariance = CreateGpsCovariance(SLAM.settings_);
            std::cout << std::fixed << std::setprecision(6);
            g = new ORB_SLAM3::GlobalPosition::GpsMeasurement(ni, latitude, longitude, altitude, vTimestamps[ni], covariance);
            //std::cout << "image timestamp:  " <<vTimestamps[ni]<<" Latitude (Decimal): " << latitude << "Longitude (Decimal): " << longitude << "altitude: "<<altitude<<std::endl;

        }
        else{
            std::cout << std::fixed << std::setprecision(6);
            cout<<"error timestamp: "<<vTimestamps[ni]<<" "<<gpsDataVector[ni].timestamp<<endl;
            cout<<"image num: "<<ni<<"read gps information failed!"<<endl;
        }

        double tframe = vTimestamps[ni];
        cv::Mat im_ori = im.clone();
        if(im.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << string(argv[3]) << "/" << vstrImageFilenames[ni] << endl;
            return 1;
        }

        if(imageScale != 1.f)
        {
#ifdef REGISTER_TIMES
    #ifdef COMPILEDWITHC11
            std::chrono::steady_clock::time_point t_Start_Resize = std::chrono::steady_clock::now();
    #else
            std::chrono::steady_clock::time_point t_Start_Resize = std::chrono::steady_clock::now();
    #endif
#endif
            int width = im.cols * imageScale;
            int height = im.rows * imageScale;
            cv::resize(im, im, cv::Size(width, height));
#ifdef REGISTER_TIMES
    #ifdef COMPILEDWITHC11
            std::chrono::steady_clock::time_point t_End_Resize = std::chrono::steady_clock::now();
    #else
            std::chrono::steady_clock::time_point t_End_Resize = std::chrono::steady_clock::now();
    #endif
            t_resize = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t_End_Resize - t_Start_Resize).count();
            SLAM.InsertResizeTime(t_resize);
#endif
        }

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#endif

        // Pass the image to the SLAM system
        SLAM.TrackMonocular(im, im_ori,tframe,{},g);
        
        //cout<<"track success!"<<endl;
#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#endif

#ifdef REGISTER_TIMES
            t_track = t_resize + std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t2 - t1).count();
            SLAM.InsertTrackTime(t_track);
#endif

        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        vTimesTrack[ni]=ttrack;

        // Wait to load the next frame
        double T=0;
        if(ni<nImages-1)
            T = vTimestamps[ni+1]-tframe;
        else if(ni>0)
            T = tframe-vTimestamps[ni-1];

        if(ttrack<T)
            usleep((T-ttrack)*1e3);
            //usleep((T-ttrack)*2e6);
    }

    // map->join();
    // delete map;

    // Stop all threads
    SLAM.Shutdown();

    // Tracking time statistics
    sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    for(int ni=0; ni<nImages; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;
    cout << "mean tracking time: " << totaltime/nImages << endl;
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");
    SLAM.SaveTrajectoryTUM("CameraTrajectory_map.txt");

    //    Map2DFusion::TestSystem mapsys;
    // std::thread* map = new thread(&Map2DFusion::TestSystem::run, mapsys);
    
    // Save camera trajectory
    

    return 0;
}

void LoadImages(const string &strFile, vector<string> &vstrImageFilenames, vector<double> &vTimestamps)
{
    ifstream f;
    f.open(strFile.c_str());

    // skip first three lines
    string s0;
    getline(f,s0);
    getline(f,s0);
    getline(f,s0);

    while(!f.eof())
    {
        string s;
        getline(f,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            string sRGB;
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenames.push_back(sRGB);
        }
    }
}
