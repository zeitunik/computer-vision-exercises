#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

void readImage(const char* file, Mat& result) {
    //TODO: implement your solution here
    result = imread(file);      //read 3 channel color image "file" into result

    if ( !result.data )
    {
        printf("No image data \n");
    }
}

void display(const char* windowTitle, Mat& img) {
    //TODO: implement your solution here
    namedWindow(windowTitle, WINDOW_AUTOSIZE );     //create a window windowTitle
    imshow(windowTitle, img);                       //display image img in window windowTitle in original size
    waitKey(0);                                     //keep the window active until a key is pressed
}

void convertToGrayImg(Mat& img, Mat& result) {
    //TODO: implement your solution here
    cvtColor(img, result, CV_BGR2GRAY);             //convert from BGR to grayscale, 
                                                    //because the imread stores the channels in BGR order
}

void subtractIntensityImage(Mat& bgrImg, Mat& grayImg, Mat& result) {
    //TODO: implement your solution here
    Mat gray3Chan;
    cvtColor(0.5*grayImg, gray3Chan, CV_GRAY2BGR);  //convert 1 channel gray image to 3 channeled one
    subtract(bgrImg, gray3Chan, result);            //both images have the same number of channels
}

void pixelwiseSubtraction(Mat& bgrImg, Mat& grayImg, Mat& result) {
    //TODO: implement your solution here
    
    Size size = bgrImg.size();
    result.create(size, bgrImg.type());             //create the resulting image

    if(bgrImg.isContinuous() && grayImg.isContinuous()) //check if both are continuous
    {
        size.width *= size.height;
        size.height = 1;
    }
    size.width *= 3;                                //the image is 3-channeled

    for (int i = 0; i < size.height; ++i)           //the loop will be implemented only once if continuous
    {
        uchar* ptr1 = bgrImg.ptr(i);                //extract the i-th row
        uchar* ptr2 = grayImg.ptr(i);
        uchar* resptr = result.ptr(i);

        for (int j = 0; j < size.width; ++j)
        {
            resptr[j] = saturate_cast<uchar>(ptr1[j] - 0.5*ptr2[(int)(j/3)]); //make sure that there are no negative values
        }
    }
}

void extractPatch(Mat& img, Mat& result) {
    //TODO: implement your solution here
    int centerx = img.size().width / 2;                                 //center of the image
    int centery = img.size().height / 2;
    result = img(Rect(centerx-8, centery-8, 16, 16));                   //get the patch via a Rectangle
        //Range(centery-8,centery+8), Range(centerx-8, centerx+8));     //get the patch via Ranges
}

void copyPatchToRandomLocation(Mat& img, Mat& patch, Mat& result) {
    //TODO: implement your solution here
    RNG rng(0xFFFFFFF);                             //initialize the random number generator
    Point p;
    p.x = rng.uniform(0, img.size().width-16);      //get random coordinates on the image: since the patch is of size 16,
    p.y = rng.uniform(0, img.size().height-16);     //hence the smaller range for the random coordinates
    Rect rect(p, Size(16,16));
    
    result = img.clone();                           //clone the image, otherwise the patch would be copied on the original image
    patch.copyTo(result(rect));
}

void drawRandomRectanglesAndEllipses(Mat& img) {
    //TODO: implement your solution here
    RNG rng(0xFFFFFFF);         
    int width = img.size().width;
    int height = img.size().height;

    Point pt1, pt2, center;                         //Points for the rectangle and the ellipse
    Size axes;

    for(int i = 0; i < 10; ++i)
    {
        pt1.x = rng.uniform(0, width);
        pt1.y = rng.uniform(0, height);
        pt2.x = rng.uniform(0, width);
        pt2.y = rng.uniform(0, height);
        Scalar col1(rng.uniform(0,256), rng.uniform(0,256), rng.uniform(0,256));
        rectangle(img, pt1, pt2, col1, 2);          //draw a rectangle with points pt1 and pt2 of col1 with line thickness 2

        center.x = rng.uniform(0, width);
        center.y = rng.uniform(0, height);
        axes.width = rng.uniform(0, width);
        axes.height = rng.uniform(0, height);

        double angle = rng.uniform(0, 180);
        Scalar col2(rng.uniform(0,256), rng.uniform(0,256), rng.uniform(0,256));

        ellipse(img, center, axes, angle, 0, 360, col2, 2); //draw an ellipse with col2 and line thickness 2

    }

    
}

int main(int argc, char* argv[])
{
    // check input arguments
    if(argc!=2) {
        cout << "usage: " << argv[0] << " <path_to_image>" << endl;
        return -1;
    }

    // read and display the input image
    Mat bgrImg;
    readImage(argv[1], bgrImg);
    display("(a) inputImage", bgrImg);

    // (b) convert bgrImg to grayImg
    Mat grayImg;
    convertToGrayImg(bgrImg, grayImg);
    display("(b) grayImage", grayImg);    

    // (c) bgrImg - 0.5*grayImg
    Mat imgC;
    subtractIntensityImage(bgrImg, grayImg, imgC);
    display("(c) subtractedImage", imgC);

    // (d) pixelwise operations
    Mat imgD;
    pixelwiseSubtraction(bgrImg, grayImg, imgD);
    display("(d) subtractedImagePixelwise", imgD);

    // (e) copy 16x16 patch from center to a random location
    Mat patch;
    extractPatch(bgrImg, patch);
    display("(e) random patch", patch);
    Mat imgE;
    copyPatchToRandomLocation(bgrImg, patch, imgE);
    display("(e) afterRandomCopying", imgE);

    // (f) drawing random rectanges on the image
    drawRandomRectanglesAndEllipses(bgrImg);
    display("(f) random elements", bgrImg);

    return 0;
}
