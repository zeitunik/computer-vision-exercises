#include <iostream>
#include <fstream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"

using namespace std;
using namespace cv;

class cEpipolar{
public:
    cEpipolar(){};
    ~cEpipolar(){};
    int loadImages();
    int loadCorresp();
    int displayCorresp();
    int computeFundMat();
    int drawEpipolar();
    int rectify();
    int applyHomography();

private:
    Mat             img1, img2;
    vector<Point2f> corr1, corr2;

    Mat F_Mat;
    Mat left_Ep;
};

int cEpipolar::loadImages(){
    img1 = imread("images/apt1.jpg");
    img2 = imread("images/apt2.jpg");
    if(0==img1.data || 0==img2.data){
        cout<<"error reading stereo images"<<endl;
        exit(-1);
    }
    return 0;
}

int cEpipolar::loadCorresp(){
    int numPts, dummy;

    ifstream iStream("images/corresp.txt");
    if(!iStream){
        cout<<"error reading the correspondence file"<<endl;
        exit(-1);
    }
    iStream >> numPts;
    iStream >> dummy;
    corr1.resize(numPts);
    corr2.resize(numPts);
    for(int idx=0; idx<numPts; ++idx){
        iStream >> corr1[idx].x;
        iStream >> corr1[idx].y;
        iStream >> corr2[idx].x;
        iStream >> corr2[idx].y;
    }
    return 0;
}

int cEpipolar::displayCorresp(){
    Mat i1, i2;
    img1.copyTo(i1);
    img2.copyTo(i2);

    cout<<"displaying corresponding points"<<endl;
    for(unsigned int idx=0; idx<corr1.size(); ++idx){
        circle(i1,corr1[idx],3,Scalar(255,0,0),2);
        circle(i2,corr2[idx],3,Scalar(255,0,0),2);
    }
    imshow("left_image",i1);
    imshow("right_image",i2);
    waitKey(0);
}

int cEpipolar::computeFundMat()
{
    // implement the function

    //find the mean and the mean squared distance
    Point2f mean1(0.,0.), mean2(0.,0.);
    float dist1=0., dist2=0.;
    mean1.x = sum(corr1)[0] / corr1.size();
    mean1.y = sum(corr1)[1] / corr1.size();

    mean2.x = sum(corr2)[0] / corr2.size();
    mean2.y = sum(corr2)[1] / corr2.size();
  
    for (int i = 0; i < corr1.size(); ++i)
    {
        dist1 += sqrt(pow(corr1[i].x-mean1.x,2) + pow(corr1[i].y-mean1.y,2));
        dist2 += sqrt(pow(corr2[i].x-mean2.x,2) + pow(corr2[i].y-mean2.y,2));
    }
    dist1 /= corr1.size();
    dist2 /= corr2.size();

    // scale the data
    vector<Point2f> scaled1(corr1.size());
    vector<Point2f> scaled2(corr2.size());

    for (int i = 0; i < corr1.size(); ++i)
    {
        scaled1[i].x = (corr1[i].x - mean1.x)*sqrt(2)/dist1;
        scaled1[i].y = (corr1[i].y - mean1.y)*sqrt(2)/dist1;
        scaled2[i].x = (corr2[i].x - mean2.x)*sqrt(2)/dist2;
        scaled2[i].y = (corr2[i].y - mean2.y)*sqrt(2)/dist2;
    }

    // create the matrix
    Mat A(corr1.size(), 9, CV_32F);
    for (int i = 0; i < A.rows; ++i)
    {
        A.at<float>(i, 0) = scaled2[i].x * scaled1[i].x;
        A.at<float>(i, 1) = scaled2[i].x * scaled1[i].y;
        A.at<float>(i, 2) = scaled2[i].x;
        A.at<float>(i, 3) = scaled2[i].y * scaled1[i].x;
        A.at<float>(i, 4) = scaled2[i].y * scaled1[i].y;
        A.at<float>(i, 5) = scaled2[i].y;
        A.at<float>(i, 6) = scaled1[i].x;
        A.at<float>(i, 7) = scaled1[i].y;
        A.at<float>(i, 8) = 1;

    }

    // solve the homogeneous system A*F = 0 
    // using svd we get that F is the last column of the right singular vector, corresponding to the smallest singular value
    Mat S, U, Vt, F(3,3,CV_32F);
    SVD svd;
    svd.compute(A, S, U, Vt);

    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            F.at<float>(i,j) = Vt.at<float>(8, 3*i + j);
        }
    }

    // enforce rank 2 on F - by taking the svd of F and setting 0 to the smallest singular value
    svd.compute(F, S, U, Vt);
    Mat S_new = Mat::zeros(3,3,CV_32F);
    S_new.at<float>(0,0) = S.at<float>(0);
    S_new.at<float>(1,1) = S.at<float>(1);

    // calculate the fundamental matrix
    F_Mat = U*S_new*Vt;    

    // revert to the original units - calculate the normalizing transformation matrices
    Mat T_t = Mat::zeros(3,3,CV_32F), T = Mat::zeros(3,3,CV_32F);

    T.at<float>(0,0) = T_t.at<float>(1,1) = sqrt(2)/dist1;        T_t.at<float>(0,0) = T.at<float>(1,1) = sqrt(2)/dist2;
    T.at<float>(2,2) = 1;                                         T_t.at<float>(2,2) = 1;
    T.at<float>(0,2) = -mean1.x*sqrt(2)/dist1;                    T_t.at<float>(0,2) = -mean2.x*sqrt(2)/dist2;
    T.at<float>(1,2) = -mean1.y*sqrt(2)/dist1;                    T_t.at<float>(1,2) = -mean2.y*sqrt(2)/dist2;
    
    F_Mat = T_t.t() * F_Mat * T;
    cout << "fundamental matrix\n" << F_Mat << endl;

    // compute the left epipole - solve F_Mat*left_Ep = 0, similar to above
    svd.compute(F_Mat, S, U, Vt);
    Mat e(3,1,CV_32F);
    for (int i = 0; i < 3; ++i)
    {
        e.at<float>(i) = Vt.at<float>(2,i);
    }

    // convert from homogeneous coordinates
    for (int i = 0; i < 3; ++i)
    {
        e.at<float>(i) /= e.at<float>(2);
    }    
    e.copyTo(left_Ep);
}

int cEpipolar::drawEpipolar(){
    // implement the function
}

int cEpipolar::rectify(){
    // implement the function
}

int cEpipolar::applyHomography(){
    // implement the function
}

class cDisparity{
public:
    cDisparity(){};
    ~cDisparity(){};
    int loadImages();
    int computeDisparity();
private:
    Mat img1, img2;
};

int cDisparity::loadImages(){
    img1 = imread("images/aloe1.png");
    img2 = imread("images/aloe2.png");
    if(0==img1.data || 0==img2.data){
        cout<<"error reading stereo images for disparity"<<endl;
        exit(-1);
    }
    return 0;
}

int cDisparity::computeDisparity(){
    return 0;
}

int main()
{
    // Q1: Fundamental Matrix
    cout<<"Q1 and Q2....."<<endl;
    cEpipolar epipolar;
    epipolar.loadImages();
    epipolar.loadCorresp();
    epipolar.displayCorresp();
    epipolar.computeFundMat();
    epipolar.drawEpipolar();

    // Q3 Disparity map
    cout<<endl<<endl<<"Q3....."<<endl;
    cDisparity disparity;
    disparity.loadImages();
    disparity.computeDisparity();

    // Q4: Rectifying image pair
    cout<<endl<<endl<<"Q4....."<<endl;
    epipolar.rectify();
    epipolar.applyHomography();

    return 0;
}
