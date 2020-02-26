#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>

using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
  ///=======================================================///
  ///                        Task-1                         ///
  ///=======================================================///

  /// read and show images
  Mat image1 = imread("./images/mountain1.png", IMREAD_COLOR);
  Mat image2 = imread("./images/mountain2.png", IMREAD_COLOR);
  imshow("Image-1", image1);
  imshow("Image-2", image2);
  waitKey(0);
  destroyAllWindows();

  /// compute keypoints and descriptors
  SIFT sift;



  /// show keypoints



  /// compute nearest matches
  BFMatcher matcher;




  /// filter matches by ratio test





  /// determine two-way matches




  /// visualize matching key-points


  waitKey(0);
  destroyAllWindows();


  ///=======================================================///
  ///                        Task-2                         ///
  ///=======================================================///

  /// Implement RANSAC here







  ///  Transform and stitch the images here






  /// visualize stitched image
  waitKey(0);
  destroyAllWindows();

}
