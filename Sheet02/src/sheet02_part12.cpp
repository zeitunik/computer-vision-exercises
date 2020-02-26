void part1()
{
    std::cout <<                                                            std::endl;
    std::cout << "/////////////////////////////////////////////////////" << std::endl;
    std::cout << "/////////////////////////////////////////////////////" << std::endl;
    std::cout << "////    Part 1    ///////////////////////////////////" << std::endl;
    std::cout << "/////////////////////////////////////////////////////" << std::endl;
    std::cout << "/////////////////////////////////////////////////////" << std::endl;

    // read the image file
    cv::Mat im_Traffic_BGR = cv::imread("./images/traffic.jpg", cv::IMREAD_COLOR);
    // gray version of bonn.png
    cv::Mat                      im_Traffic_Gray;
    cv::cvtColor(im_Traffic_BGR, im_Traffic_Gray, CV_BGR2GRAY);

    // construct Gaussian pyramids
    std::vector<cv::Mat>   gpyr;    // this will hold the Gaussian Pyramid created with OpenCV
    std::vector<cv::Mat> myGpyr;    // this will hold the Gaussian Pyramid created with your custom way

    // Please implement the pyramids as described in the exercise sheet, using the containers given above.
    const int num_levels = 4;
    cv::buildPyramid(im_Traffic_Gray, gpyr, num_levels);

    myGpyr.push_back(im_Traffic_Gray);
    
    for (int i = 1; i < num_levels; ++i)
    {
        cv::Mat temp, temp1;
        cv::GaussianBlur(myGpyr[i-1], temp, cv::Size(5,5), -1);
        cv::resize(temp, temp1, cv::Size(std::round(temp.cols/2.), std::round(temp.rows/2.)));
        myGpyr.push_back(temp1);
    }

    // Perform the computations asked in the exercise sheet and show them using **std::cout**
    cv::Mat diff;
    double minVal, maxVal;
    std::cout << "maximum pixel error for layer:\n";
    for (int i = 0; i < num_levels; ++i)
    {
        cv::absdiff(gpyr[i], myGpyr[i], diff); 
        cv::minMaxLoc(diff, &minVal, &maxVal);
        std::cout << "\t" << i << "\t" << maxVal << std::endl;
    }

    // Show every layer of the pyramid
    // using **cv::imshow and cv::waitKey()** and when necessary **std::cout**
    // In the end, after the last cv::waitKey(), use **cv::destroyAllWindows()**
    for (int i = 0; i < num_levels; ++i)
    {
        cv::imshow("Gaussian Pyramid: OpenCV", gpyr[i]);
        cv::waitKey(0);
        cv::imshow("Gaussian Pyramid: own implementation", myGpyr[i]);
        cv::waitKey(0);
    }
         
    cv::destroyAllWindows();

    // For the laplacian pyramid you should define your own container.
    // If needed perform normalization of the image to be displayed
    std::vector<cv::Mat>    lpyr;
    for (int i = 0; i < num_levels-1; ++i)
    {
        cv::Mat temp;
        cv::pyrUp(gpyr[i+1], temp, cv::Size(gpyr[i].cols, gpyr[i].rows));
        lpyr.push_back(gpyr[i]-temp);
    }
    lpyr.push_back(gpyr[num_levels-1]);

    for (int i = 0; i < num_levels; ++i)
    {
        cv::imshow("Laplacian Pyramid " + std::to_string(i), lpyr[i]);
        cv::waitKey(0);
    }

    cv::destroyAllWindows();
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void part2()
{
    std::cout <<                                                            std::endl;
    std::cout << "/////////////////////////////////////////////////////" << std::endl;
    std::cout << "/////////////////////////////////////////////////////" << std::endl;
    std::cout << "////    Part 2    ///////////////////////////////////" << std::endl;
    std::cout << "/////////////////////////////////////////////////////" << std::endl;
    std::cout << "/////////////////////////////////////////////////////" << std::endl;

    // apple and orange are CV_32FC3
    
    cv::Mat im_Apple, im_Orange;
    cv::imread("./images/apple.jpg",  cv::IMREAD_COLOR).convertTo(im_Apple,  CV_32FC3, (1./255.));
    cv::imread("./images/orange.jpg", cv::IMREAD_COLOR).convertTo(im_Orange, CV_32FC3, (1./255.));
    cv::imshow("orange", im_Orange);
    cv::imshow("apple",  im_Apple );
    std::cout << "\n" << "Input images" << "   \t\t\t\t\t\t\t" << "Press any key..." << std::endl;
    cv::waitKey(0);
    
    
    // Perform the blending using a Laplacian Pyramid

    //Build the Gaussian pyramids
    std::vector<cv::Mat>    gApple, gOrange;
    std::vector<cv::Mat>    lApple, lOrange, lFinal;
    std::vector<cv::Mat>    gMask;

    cv::Mat Mask(im_Apple.size(), im_Apple.type());
    int blending_size = im_Apple.cols/4;
    int left_edge = (Mask.cols - blending_size)/2;
    int right_edge = (Mask.cols + blending_size)/2;

    for (int i = 0; i < Mask.rows; ++i)
    {
        float *data = Mask.ptr<float>(i);
        for (int j = 0; j < Mask.cols; ++j)
        {
            if(j < left_edge)
                data[3*j] = data[3*j+1] = data[3*j+2] = 1.;
            else if (j >= right_edge)
                data[3*j] = data[3*j+1] = data[3*j+2] = 0.;
            else 
                data[3*j] = data[3*j+1] = data[3*j+2] = float(right_edge-j)/(float)blending_size;
        }
    }

    cv::imshow("Mask image", Mask);
    cv::waitKey(0);

    const int num_levels = 4;
    cv::buildPyramid(im_Apple, gApple, num_levels);
    cv::buildPyramid(im_Orange, gOrange, num_levels);
    cv::buildPyramid(Mask, gMask, num_levels);

    lApple.resize(num_levels);
    lOrange.resize(num_levels);
    lFinal.resize(num_levels);

    lApple[num_levels-1] = gApple[num_levels-1];
    lOrange[num_levels-1] = gOrange[num_levels-1];

    //Build the Laplacian pyriamids
    for (int i = num_levels-1; i > 0; --i)
    {
        cv::pyrUp(gApple[i], lApple[i-1], gApple[i-1].size());
        lApple[i-1] = gApple[i-1] - lApple[i-1];

        cv::pyrUp(gOrange[i], lOrange[i-1], gOrange[i-1].size());
        lOrange[i-1] = gOrange[i-1] - lOrange[i-1];
    }

    lFinal[num_levels-1] = gMask[num_levels-1].mul(lApple[num_levels-1]) 
        + (cv::Scalar::all(1.)-gMask[num_levels-1]).mul(lOrange[num_levels-1]);

    //collapse to get final blended image
    for (int i = num_levels-1; i > 0; --i)
    {
        //cv::Mat temp;
        cv::pyrUp(lFinal[i], lFinal[i-1], lApple[i-1].size());
        lFinal[i-1] += gMask[i-1].mul(lApple[i-1]) + (cv::Scalar::all(1.)-gMask[i-1]).mul(lOrange[i-1]);
    }

    // Show the blending results @ several layers
    // using **cv::imshow and cv::waitKey()** and when necessary **std::cout**
    // In the end, after the last cv::waitKey(), use **cv::destroyAllWindows()**
    
    for (int i = 0; i < num_levels/2; ++i)
    {
        cv::imshow("Blended image " + std::to_string(i), lFinal[i]);
        cv::waitKey(0);
    }
    cv::destroyAllWindows();
}
