// Batched PS OpenCV operations. Optical flow and pixel preprocessing remain PL.
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <vector>

extern "C" int tracking_cv_patches(const unsigned char* current,
        const unsigned char* previous, int width, int height, const float* points,
        int count, const double* h, unsigned char* first, unsigned char* second) {
    if (!current || !previous || !points || !h || !first || !second ||
        width < 24 || height < 24 || count < 1 || count > 240) return -1;
    try {
        cv::Mat coordinates(count*16,16,CV_32FC2);
        for (int p=0; p<count; ++p) {
            float ox=std::floor(points[2*p]-7), oy=std::floor(points[2*p+1]-7);
            for (int y=0; y<16; ++y) for (int x=0; x<16; ++x)
                coordinates.at<cv::Vec2f>(p*16+y,x)=cv::Vec2f(
                    std::max(0.f,std::min(float(width-1),ox+x)),
                    std::max(0.f,std::min(float(height-1),oy+y)));
        }
        cv::Mat source,atlas;
        cv::perspectiveTransform(coordinates,source,cv::Mat(3,3,CV_64F,const_cast<double*>(h)));
        cv::remap(cv::Mat(height,width,CV_8U,const_cast<unsigned char*>(current)),
            atlas,source,cv::Mat(),cv::INTER_LINEAR,cv::BORDER_CONSTANT,0);
        cv::Mat old(height,width,CV_8U,const_cast<unsigned char*>(previous));
        for (int p=0; p<count; ++p) {
            cv::Point2f point(points[2*p],points[2*p+1]);
            cv::Point2f offset(point.x-std::floor(point.x-7),point.y-std::floor(point.y-7));
            cv::Mat a(15,15,CV_8U,first+p*225), b(15,15,CV_8U,second+p*225);
            cv::getRectSubPix(atlas.rowRange(p*16,(p+1)*16),cv::Size(15,15),offset,a);
            cv::getRectSubPix(old,cv::Size(15,15),point,b);
        }
        return 0;
    } catch (const cv::Exception&) { return -1; }
}

extern "C" int tracking_cv_skin(const unsigned char* rgba, int width, int height,
        const float* quad, float px, float py, int right_only, float* out) {
    if (!rgba || !quad || !out || width < 1 || height < 1 || width > 160 || height > 120) return -1;
    try {
        std::vector<cv::Point2f> polygon;
        for (int i=0; i<4; ++i) polygon.push_back(cv::Point2f(quad[2*i],quad[2*i+1]));
        cv::Point2f axis=(polygon[1]+polygon[2]-polygon[0]-polygon[3])*.5;
        float norm=std::max(float(cv::norm(axis)),1.f);
        cv::Point2f normal(-axis.y/norm,axis.x/norm);
        int hint=std::max(8,int(cv::norm(polygon[3]-polygon[0])*.8));
        std::vector<cv::Point2f> expanded;
        for (int sign : {1,-1}) for (const auto& p : polygon)
            expanded.push_back(p+normal*(hint*1.25f*sign));
        std::vector<cv::Point2f> hull;
        cv::convexHull(expanded,hull);
        std::vector<cv::Point> search,neck;
        for (const auto& p : hull) search.push_back(cv::Point(cvRound(p.x),cvRound(p.y)));
        for (const auto& p : polygon) neck.push_back(cv::Point(cvRound(p.x),cvRound(p.y)));
        cv::Mat rgb,ycrcb,skin;
        cv::cvtColor(cv::Mat(height,width,CV_8UC4,const_cast<unsigned char*>(rgba)),rgb,cv::COLOR_RGBA2RGB);
        cv::cvtColor(rgb,ycrcb,cv::COLOR_RGB2YCrCb);
        cv::inRange(ycrcb,cv::Scalar(36,150,75),cv::Scalar(255,180,135),skin);
        cv::Mat mask=cv::Mat::zeros(height,width,CV_8U), neck_mask=mask.clone();
        cv::fillConvexPoly(mask,search,255);
        cv::fillConvexPoly(neck_mask,neck,255);
        if (right_only) mask.colRange(0,width/2).setTo(0);
        cv::bitwise_and(skin,mask,skin);
        cv::morphologyEx(skin,skin,cv::MORPH_OPEN,cv::Mat::ones(3,3,CV_8U));
        cv::Mat labels,stats,centroids;
        int count=cv::connectedComponentsWithStats(skin,labels,stats,centroids);
        struct Sum { double x=0,y=0,warm=0; int contact=0; };
        std::vector<Sum> sums(count);
        for (int y=0; y<height; ++y) for (int x=0; x<width; ++x) {
            int id=labels.at<int>(y,x);
            if (!id) continue;
            sums[id].warm += ycrcb.at<cv::Vec3b>(y,x)[1];
            if (neck_mask.at<unsigned char>(y,x)) {
                sums[id].x+=x; sums[id].y+=y; ++sums[id].contact;
            }
        }
        double best=-1;
        for (int i=1; i<count; ++i) {
            const int* s=stats.ptr<int>(i);
            int w=s[2],h=s[3],area=s[4];
            if (area < 18 || area > width*height*.2 ||
                double(std::max(w,h))/std::max(1,std::min(w,h)) > 5 ||
                double(area)/std::max(1,w*h) > .92 || sums[i].contact < 8) continue;
            cv::Point2f center(sums[i].x/sums[i].contact,sums[i].y/sums[i].contact);
            if (right_only && center.x < (width-1)*.5) continue;
            double distance=std::max(0.,-cv::pointPolygonTest(polygon,center,true))/hint;
            double temporal=1;
            if (std::isfinite(px) && std::isfinite(py))
                temporal=.5+.5*std::max(0.,1.-cv::norm(center-cv::Point2f(px,py))/(hint*1.5));
            double score=std::min(1.,sums[i].contact/180.)*std::max(0.,1.-distance)*
                temporal*(sums[i].warm/area/180.);
            if (score <= best) continue;
            best=score;
            out[0]=center.x; out[1]=center.y; out[2]=score;
            out[3]=s[0];out[4]=s[1];out[5]=s[0]+w-1;out[6]=s[1]+h-1;
        }
        return best >= 0;
    } catch (const cv::Exception&) { return -1; }
}
