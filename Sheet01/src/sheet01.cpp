#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <math.h>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    
//	==================== Load image =========================================
  	// Read the image file - bonn.png is of type CV_8UC3
	const Mat bonn = imread(argv[1], IMREAD_COLOR);

	// Create gray version of bonn.png
	Mat bonn_gray;
	cvtColor(bonn, bonn_gray, CV_BGR2GRAY);

//	=========================================================================
//	==================== Solution of task 1 =================================
//	=========================================================================
	cout << "Task 1:" << endl;
	imshow("gray bonn", bonn_gray);
	//====(a)==== draw random rectangles ====
	
	// Create a copy of the input image to draw the rectangles into
	Mat img_task1;
	bonn_gray.copyTo(img_task1);
	
	// Create a random number generator
	RNG rng(getTickCount());
	Rect rects[100];
	for (auto& r : rects) {
		// Random coordinates of the rectangle
		const int x1 = rng.uniform(1, bonn_gray.cols-1);
		const int x2 = rng.uniform(1, bonn_gray.cols-1);
		const int y1 = rng.uniform(1, bonn_gray.rows-1);
		const int y2 = rng.uniform(1, bonn_gray.rows-1);
		
		//TODO: Draw the rectangle into the image "img_task1"
		r = Rect(Point(x1, y1), Point(x2, y2));
		
        rectangle(img_task1, r, Scalar(rng.uniform(0,256), rng.uniform(0,256), rng.uniform(0,256))); 
	}
	//TODO: Display the image "img_task1" with the rectangles
	imshow("Image with the rectangles", img_task1);	
	
	//====(b)==== summing up method ====
	// Variables for time measurements
	int64_t tick, tock;
	double sumOfPixels;
	
	// Measure the time
	tick = getTickCount();
	
	// Repeat 10 times
	for (size_t n = 0; n < 10; ++n) 
	{
		sumOfPixels = 0.0;
		for (auto& r : rects) 
		{
			double avgInt = 0.0;
			for (int y = r.y; y < r.y + r.height; ++y) 
				for (int x = r.x; x < r.x+r.width; ++x)	
				  	//TODO: Sum up all pixels at "bonn_gray.at<uchar>(y,x)" inside the rectangle and store it in "sumOfPixels" 
				  	//sum of all the pixels in rectangle r
					avgInt += bonn_gray.at<uchar>(y,x);		

			if(r.area() != 0)
				sumOfPixels += avgInt/double(r.area());			//average intensity in the rectangle r, i.e. the sum divided by the area
		}
	}
	tock = getTickCount();

	cout << "Summing up each pixel gives " << sumOfPixels << " computed in " << (tock-tick)/getTickFrequency() << " seconds." << endl;

	// //====(c)==== integral image method - using OpenCV function "integral" ==== 
	
	//TODO: implement your solution of here
	tick = getTickCount();
	Mat int_img_cv;
	//calculate the integral image with opencv function
	integral(bonn_gray, int_img_cv);
	double sumOfRects_cv;

	for (size_t n = 0; n < 10; ++n)
	{
		sumOfRects_cv = 0.0;
		for(auto& r : rects)
		{
			if(r.area() != 0)
				sumOfRects_cv += (int_img_cv.at<int>(r.y+r.height, r.x+r.width) - int_img_cv.at<int>(r.y+r.height, r.x) -
								int_img_cv.at<int>(r.y, r.x+r.width) + int_img_cv.at<int>(r.y, r.x))/(double)r.area();
		}		
	}
	tock = getTickCount();
	cout << "OpenCV integral function gives " << sumOfRects_cv << " computed in " << (tock-tick)/getTickFrequency() << " seconds." << endl;
	
	//====(b)==== integral image method - custom implementation====
	//TODO: implement your solution here
	tick = getTickCount();
	int rows = bonn_gray.rows;
	int cols = bonn_gray.cols;

	Mat int_img(rows, cols, CV_32S);

	int_img.at<int>(0,0) = bonn_gray.at<uchar>(0,0);

	//initialization of the first row
	for (int j = 1; j < cols; ++j)
		int_img.at<int>(0, j) = int_img.at<int>(0, j-1) + bonn_gray.at<uchar>(0,j);

	//initialization of the first column
	for (int i = 1; i < rows; ++i)
		int_img.at<int>(i, 0) = int_img.at<int>(i-1, 0) + bonn_gray.at<uchar>(i,0);
	
	//initialization of the rest of the matrix -> not very efficient, can be improved
	for (int i = 1; i < rows; ++i)
		for (int j = 1; j < cols; ++j)
			int_img.at<int>(i,j) = int_img.at<int>(i-1, j) + int_img.at<int>(i, j-1) - 
										int_img.at<int>(i-1, j-1) + (int)bonn_gray.at<uchar>(i,j);


	double sumOfRects;
	for (size_t n = 0; n < 10; ++n) 
	{
		sumOfRects = 0.0;
		int l =0;
		for (auto& r : rects) 
		{
			if(r.area() != 0)
				sumOfRects += (int_img.at<int>(r.y+r.height-1, r.x+r.width-1) - int_img.at<int>(r.y+r.height-1, r.x-1) -
								int_img.at<int>(r.y-1, r.x+r.width-1) + int_img.at<int>(r.y-1, r.x-1))/(double)r.area();
		}		
	}
	tock = getTickCount();
	cout << "My implementation gives " << sumOfRects << " computed in " << (tock-tick)/getTickFrequency() << " seconds." << endl;

	waitKey(0); // waits until the user presses a button and then continues with task 2 -> uncomment this
	destroyAllWindows(); // closes all open windows -> uncomment this
	
//	=========================================================================	
//	==================== Solution of task 2 =================================
//	=========================================================================
	cout << "\nTask 2:" << endl;
	
	imshow("gray bonn", bonn_gray);
	waitKey(0);
	//====(a)==== Histogram equalization - using opencv "equalizeHist" ====
	Mat ocvHistEqualization;
	//TODO: implement your solution of here
	equalizeHist(bonn_gray, ocvHistEqualization);	//opencv equalizeHist function
	
	//====(b)==== Histogram equalization - custom implementation ====
	Mat myHistEqualization(bonn_gray.size(), bonn_gray.type()); 

	// Count the frequency of intensities
	Vec<double,256> hist(0.0);
	for (int y=0; y < bonn_gray.rows; ++y) {
		for (int x=0;  x < bonn_gray.cols; ++x) {
			++hist[bonn_gray.at<uchar>(y,x)];
		}
	}
	// Normalize vector of new intensity values to sum up to a sum of 255
	hist *= 255. / sum(hist)[0];
	
	// Compute integral histogram - representing the new intensity values
	for (size_t i=1; i < 256; ++i) {
		hist[i] += hist[i-1];
	}
	
	// TODO: Fill the matrix "myHistEqualization" -> replace old intensity values with new intensities taken from the integral histogram
	if(bonn_gray.isContinuous())
	{
		cols *= rows;
		rows = 1;
	}
	uchar* row_src, *row_dst;
	for (int i = 0; i < rows; ++i)
	{
        row_src = bonn_gray.ptr<uchar>(i);
        row_dst = myHistEqualization.ptr<uchar>(i);
		for (int j = 0; j < cols; ++j)
			row_dst[j] = hist[row_src[j]];		//get the new intensity value by using the integral histogram as a look-up table
	}
	
	// TODO: Show the results of (a) and (b)
	imshow("ocvHistEqualization", ocvHistEqualization);
	waitKey(0);
	imshow("myHistEqualization", myHistEqualization);
	
	//====(c)==== Difference between openCV implementation and custom implementation ====
	// TODO: Compute absolute differences between pixel intensities of (a) and (b) using "absdiff"
	// ... Just uncomment the following lines:
	Mat diff;
	absdiff(myHistEqualization, ocvHistEqualization, diff);
	double minVal, maxVal;
	minMaxLoc(diff, &minVal, &maxVal);
	cout << "maximum pixel error: " << maxVal << endl;

	waitKey(0);
	destroyAllWindows();
	
//	=========================================================================	
//	==================== Solution of task 4 =================================
//	=========================================================================
	cout << "\nTask 4:" << endl;
	imshow("gray bonn", bonn_gray);
	waitKey(0);

	Mat img_gb;
	Mat img_f2D;
	Mat img_sepF2D;
	const double sigma = 2. * sqrt(2.);
	double sigma2 = sigma*sigma;


    //  ====(a)==== 2D Filtering - using opencv "GaussianBlur()" ====
	tick = getTickCount();
	// TODO: implement your solution here:
	Size size(2*(int)(3.5*sigma)+1, 2*(int)(3.5*sigma)+1); // half-size of the kernel is [3.5*sigma], the size has to be odd
	GaussianBlur(bonn_gray, img_gb, size, sigma);

	tock = getTickCount();
	cout << "OpenCV GaussianBlur() method takes " << (tock-tick)/getTickFrequency() << " seconds." << endl;
	
    //  ====(b)==== 2D Filtering - using opencv "filter2D()" ==== 
	// Compute gaussian kernel manually
	const int k_width = 3.5 * sigma;
	tick = getTickCount();

	Matx<float, 2*k_width+1, 2*k_width+1> kernel2D;
	for (int y = 0; y < kernel2D.rows; ++y) {
		const int dy = abs(k_width - y);
		for (int x = 0; x < kernel2D.cols; ++x) {
			const int dx = abs(k_width-x);
			//TODO: Fill kernel2D matrix with values of a gaussian
			kernel2D(y,x) = exp((-dx*dx-dy*dy)/(2.*sigma2));
		}
	}
	kernel2D *= 1. / sum(kernel2D)[0];
	
	// TODO: implement your solution here - use "filter2D"
	filter2D(bonn_gray, img_f2D, -1, kernel2D);	
	tock = getTickCount();
	cout << "OpenCV filter2D() method takes " << (tock-tick)/getTickFrequency() << " seconds." << endl;

    //  ====(c)==== 2D Filtering - using opencv "sepFilter2D()" ====

   	tick = getTickCount();
	// TODO: implement your solution here

	Vec<float, 2*k_width+1> kernel1D;
	for (int x = 0; x < kernel1D.rows; ++x)
	{
		const int dx = abs(x-k_width);
		kernel1D(x) = exp(-dx*dx/(2.*sigma2));
	}

	kernel1D *= 1. / sum(kernel1D)[0];
	sepFilter2D(bonn_gray, img_sepF2D, -1, kernel1D, kernel1D);

	tock = getTickCount();
	cout << "OpenCV sepFilter2D() method takes " << (tock-tick)/getTickFrequency() << " seconds." << endl;

	// TODO: Show result images
	imshow("GaussianBlur", img_gb);
	waitKey(0);
	imshow("Filter2D", img_f2D);
	waitKey(0);
	imshow("sepFilter2D", img_sepF2D);

	// compare blurring methods
	// TODO: Compute absolute differences between pixel intensities of (a), (b) and (c) using "absdiff"
	Mat diff_gbf2D, diff_gbsepf, diff_f2Dsepf;
	double min_gbf2D, max_gbf2D, min_gbsepf, max_gbsepf, min_f2Dsepf, max_f2Dsepf;
	
	absdiff(img_gb, img_f2D, diff_gbf2D);
	absdiff(img_gb, img_sepF2D, diff_gbsepf);
	absdiff(img_f2D, img_sepF2D, diff_f2Dsepf);
			
	// TODO: Find the maximum pixel error using "minMaxLoc"
	minMaxLoc(diff_gbf2D, &min_gbf2D, &max_gbf2D);
	minMaxLoc(diff_gbsepf, &min_gbsepf, &max_gbsepf);
	minMaxLoc(diff_f2Dsepf, &min_f2Dsepf, &max_f2Dsepf);
	cout << "maximum pixel error between \n"
		<< "\t\tGaussianBlur and Filter2D:\t" << max_gbf2D << endl
		<< "\t\tGaussianBlur and sepFilter2D:\t" << max_gbsepf << endl
		<< "\t\tFilter2D and sepFilter2D:\t" << max_f2Dsepf << endl;

	waitKey(0);
	destroyAllWindows();	
	
//	=========================================================================	
//	==================== Solution of task 6 =================================
//	=========================================================================
	cout << "\nTask 6:" << endl;
	imshow("gray bonn", bonn_gray);
	waitKey(0);
	const double sigma3 = 2.;
    //  ====(a)==================================================================
	// TODO: implement your solution here
	Mat img_bl2;
	Size size1(2*(int)(3.5*sigma3)+1, 2*(int)(3.5*sigma3)+1); 	//size of the kernel, has to be odd
	GaussianBlur(bonn_gray, img_bl2, size1, sigma3);			// blur the image twice with sigma3, in-place filtering is possible
	GaussianBlur(img_bl2, img_bl2, size1, sigma3);

    //  ====(b)==================================================================	
	// TODO: implement your solution here
	Mat img_bl;
	GaussianBlur(bonn_gray, img_bl, size, sigma);		// blur once with sigma=2*sqrt(2.)
	
	imshow("Blurred twice", img_bl2);
	waitKey(0);
	imshow("Blurred once", img_bl);

	absdiff(img_bl2, img_bl, diff);
	minMaxLoc(diff, &minVal, &maxVal);
	cout << "maximum pixel error between once and twice blurred: " << maxVal << endl;
	
	waitKey(0);
	destroyAllWindows();	

//	=========================================================================	
//	==================== Solution of task 7 =================================
//	=========================================================================
	cout << "\nTask 7:" << endl;
	// Create an image with salt and pepper noise
	Mat bonn_salt_pepper(bonn_gray.size(), bonn_gray.type());
	randu(bonn_salt_pepper, 0, 100); // Generates an array of random numbers
	
	for (int y = 0; y < bonn_gray.rows; ++y) {
		for (int x=0; x < bonn_gray.cols; ++x) {
			uchar& pix = bonn_salt_pepper.at<uchar>(y,x);
			if (pix < 15) {
			  // TODO: Set set pixel "pix" to black
				pix=0;
			}else if (pix >= 85) { 
			  // TODO: Set set pixel "pix" to white
				pix=255;
			}else { 
			  // TODO: Set set pixel "pix" to its corresponding intensity value in bonn_gray
				pix = bonn_gray.at<uchar>(y,x);
			}
		}
	}
	imshow("bonn.png with salt and pepper", bonn_salt_pepper);	
	waitKey(0);

    //  ====(a)==================================================================
	// TODO: implement your solution here
	Mat saltpep_blurred;
	const double sigma4 = 1.5;		// higher sigma values blur the image too much, without much reduction of the noise
	const int k_width4 = 3.5*sigma4;
	Size size4(2*k_width4+1, 2*k_width4+1);
	GaussianBlur(bonn_salt_pepper, saltpep_blurred, size4, sigma4);
	imshow("Salt and pepper GaussianBlur", saltpep_blurred);
	waitKey(0);

    //  ====(b)==================================================================	
	// TODO: implement your solution here
	Mat saltpep_median;
	int ksize = 3;		//higher values make the image lose the details
	medianBlur(bonn_salt_pepper, saltpep_median, ksize);
	imshow("Salt and pepper MedianBlur", saltpep_median);
	waitKey(0);

    //  ====(c)==================================================================	
	// TODO: implement your solution here
	Mat saltpep_bilat;
	int d = 7;
	double sigmaCol = 120;
	double sigmaSpace = 120;
	bilateralFilter(bonn_salt_pepper, saltpep_bilat, d, sigmaCol, sigmaSpace);
	imshow("Salt and pepper Bilateral Filter", saltpep_bilat);

	cout << "Parameters for:\n" <<
			"\t\tGaussian Blur sigma: " << sigma4 << " kernel size: 6*sigma" << endl <<
			"\t\tMedian Filter aperture: " << ksize << endl << 
			"\t\tBilateral Filter size: " << d << " sigmaColor: " << sigmaCol << " sigmaSpace: " << sigmaSpace << endl;
	waitKey(0);
	destroyAllWindows();	

//	=========================================================================	
//	==================== Solution of task 8 =================================
//	=========================================================================	
	cout << "\nTask 8:" << endl;
	imshow("gray bonn", bonn_gray);
	waitKey(0);
	// Declare Kernels
	Mat kernel1 = (Mat_<float>(3,3) << 0.0113, 0.0838, 0.0113, 0.0838, 0.6193, 0.0838, 0.0113, 0.0838, 0.0113);	
	Mat kernel2 = (Mat_<float>(3,3) << -0.8984, 0.1472, 1.1410, -1.9075, 0.1566, 2.1359, -0.8659, 0.0573, 1.0337);
	
    //  ====(a)==================================================================
	// TODO: implement your solution here
	Mat filter1, filter2;
	filter2D(bonn_gray, filter1, -1, kernel1);
	filter2D(bonn_gray, filter2, -1, kernel2);

    //  ====(b)==================================================================	
	// TODO: implement your solution here
	SVD svd1(kernel1), svd2(kernel2);
	double factor1 = sqrt(svd1.w.at<float>(0));
	double factor2 = sqrt(svd2.w.at<float>(0));
	
	Mat kernel11(factor1*svd1.u.col(0));	//vertical filter
	Mat kernel12(factor1*svd1.vt.row(0));	//horizontal filter

	Mat kernel21(factor2*svd2.u.col(0));	//vertical filter
	Mat kernel22(factor2*svd2.vt.row(0));	//horizontal filter
	
	Mat filter1_SVD, filter2_SVD;
	sepFilter2D(bonn_gray, filter1_SVD, -1, kernel12, kernel11);
	sepFilter2D(bonn_gray, filter2_SVD, -1, kernel22, kernel21);
	
	imshow("Bonn filtered with Kernel1", filter1);
	waitKey(0);
	imshow("Bonn filtered with Kernel1 SVD", filter1_SVD);
	waitKey(0);
	imshow("Bonn filtered with Kernel2", filter2);
	waitKey(0);
	imshow("Bonn filtered with Kernel2 SVD", filter2_SVD);
	waitKey(0);

	//  ====(c)==================================================================	
	// TODO: implement your solution here
	absdiff(filter1, filter1_SVD, diff);
	minMaxLoc(diff, &minVal, &maxVal);
	cout << "maximum pixel error between (a) and (b)\n" <<
			"\t\tfor kernel1: " << maxVal << endl;
	absdiff(filter2, filter2_SVD, diff);
	minMaxLoc(diff, &minVal, &maxVal);
	cout << "\t\tfor kernel2: " << maxVal << endl;	
	
	cout << "Program finished successfully" << endl;
	destroyAllWindows(); 
	return 0;
}