void part2_1()
{
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////    Part 2 - 1    //////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;

    // read the image file flower
    cv::Mat                                              flower;
    cv::imread( PATH_Flower, cv::IMREAD_COLOR).convertTo(flower, CV_32FC3, (1./255.)); // image normalized [0,255] -> [0,1]
    cv::imshow("original image",                         flower);
    // gray version of flower
    cv::Mat              flower_gray;
    cv::cvtColor(flower, flower_gray, CV_BGR2GRAY);
    cv::imshow("original image", flower_gray);

    // Perform the steps described in the exercise sheet
    int K[] = {2,4,6,8,10};
    
    int rows = flower_gray.rows;
    int cols = flower_gray.cols;
    cv::Mat intensities = flower_gray.reshape(0, rows*cols);

    cv::Mat labels, centers, new_intensities(flower_gray.size(), flower_gray.type(), 0.);
    int attempts = 3;
    for (int k = 0; k < 5; ++k)
    {
        cv::kmeans(intensities, K[k], labels, 
            cv::TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 1.0), 
            attempts, cv::KMEANS_RANDOM_CENTERS, centers);  //number of iterations = 10, epsilon = 1.0 
        
        labels = labels.reshape(0, rows);
        std::cout << centers << "\n";
        for (int i = 0; i < centers.rows; ++i)
        {
            cv::Mat mask = (labels == i);
            new_intensities.setTo(centers.at<float>(i), mask);
        }

        // Show results
        // using **cv::imshow and cv::waitKey()** and when necessary **std::cout**
        // In the end, after the last cv::waitKey(), use **cv::destroyAllWindows()**
        cv::imshow("Intensities Kmeans with k = " + std::to_string(K[k]), new_intensities);//.reshape(0, rows));
        cv::waitKey(0);
    }

    cv::destroyAllWindows();
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void part2_2()
{
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////    Part 2 - 2    //////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;

    std::cout << "\n" << "kmeans with color" << std::endl;

    // read the image file flower
    cv::Mat                                              flower;
    cv::imread( PATH_Flower, cv::IMREAD_COLOR).convertTo(flower, CV_32FC3, (1./255.)); // image normalized [0,255] -> [0,1]
    // gray version of flower
    // cv::Mat              flower_gray;
    // cv::cvtColor(flower, flower_gray, CV_BGR2GRAY);
    cv::imshow("original image", flower);

    // Perform the steps described in the exercise sheet
    cv::Mat     flower_luv;
    cv::cvtColor(flower, flower_luv, CV_BGR2Luv);
    cv::imshow("Luv image", flower_luv);
    
    int K[] = {2,4,6,8,10};
    
    int rows = flower_luv.rows;
    int cols = flower_luv.cols;
    cv::Mat colors = flower_luv.reshape(0, rows*cols);

    cv::Mat labels, centers, new_colors(flower_luv.size(), flower_luv.type(), cv::Scalar::all(0.));
    int attempts = 3;

    for (int k = 0; k < 5; ++k)
    {
        cv::kmeans(colors, K[k], labels, 
            cv::TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 1.0), 
            attempts, cv::KMEANS_RANDOM_CENTERS, centers);  //number of iterations = 10, epsilon = 1.0 
        
        labels = labels.reshape(0, rows);

        for (int i = 0; i < centers.rows; ++i)
        {
            cv::Mat mask = (labels == i);
            cv::Vec3b col(centers.at<float>(i, 0), centers.at<float>(i, 1), centers.at<float>(i, 2));
            new_colors.setTo(col, mask);
        }

        // Show results
        // using **cv::imshow and cv::waitKey()** and when necessary **std::cout**
        // In the end, after the last cv::waitKey(), use **cv::destroyAllWindows()**
        cv::Mat im;
        cv::cvtColor(new_colors, im, CV_Luv2BGR);

        cv::imshow("Color Kmeans with " + std::to_string(K[k]), im);
        cv::waitKey(0);

    }
    cv::destroyAllWindows();
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void part2_3()
{
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////    Part 2 - 3    //////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;

    std::cout << "\n" << "kmeans with gray and pixel coordinates" << std::endl;

    // read the image file flower
    cv::Mat                                              flower;
    cv::imread( PATH_Flower, cv::IMREAD_COLOR).convertTo(flower, CV_32FC3, (1./255.)); // image normalized [0,255] -> [0,1]
    // gray version of flower
    cv::Mat              flower_gray;
    cv::cvtColor(flower, flower_gray, CV_BGR2GRAY);
    cv::imshow("original image", flower);

    // Perform the steps described in the exercise sheet
    cv::imshow("gray image", flower_gray);

    int K[] = {2,4,6,8,10};
    
    int rows = flower_gray.rows;
    int cols = flower_gray.cols;
    cv::Mat points(flower_gray.size(), CV_32FC3);

    for (int i = 0; i < rows; ++i)
    {
        uchar* data = flower_gray.ptr<uchar>(i);
        float* new_data = points.ptr<float>(i);
        for (int j = 0; j < cols; ++j)
        {
            new_data[3*j] = (float)data[j];
            new_data[3*j+1] = (float)i;
            new_data[3*j+2] = (float)j;
        }
    }

    points = points.reshape(3, rows*cols);

    cv::Mat labels, centers, new_intensities(flower_gray.size(), CV_32FC3, 0.);
    int attempts = 3;
    
    for (int k = 0; k < 5; ++k)
    {
        cv::kmeans(points, K[k], labels, 
            cv::TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 1.0), 
            attempts, cv::KMEANS_RANDOM_CENTERS, centers);  //number of iterations = 10, epsilon = 1.0 
        
        labels = labels.reshape(0, rows);

        for (int i = 0; i < centers.rows; ++i)
        {
            cv::Mat mask = (labels == i);
            cv::Vec3b value(centers.at<float>(i, 0), centers.at<float>(i, 1), centers.at<float>(i, 2));
            new_intensities.setTo(value, mask);
        }

        cv::Mat mv[3], im(flower_gray.size(), CV_8UC1, 0.);
        cv::split(new_intensities, mv);
                
        mv[0].convertTo(im, CV_8UC1);
        cv::imshow("Intensities + position Kmeans with k = " + std::to_string(K[k]), im);
        cv::waitKey(0);
    }
    
    cv::destroyAllWindows();
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void part3()
{
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////    Part 3    //////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;

    // read the image file flower
    cv::Mat                                              flower;
    cv::imread( PATH_Flower, cv::IMREAD_COLOR).convertTo(flower, CV_8UC3); //CV_32FC3, (1./255.)); // image normalized [0,255] -> [0,1]
    cv::imshow("original image",                         flower);
    cv::waitKey(0);
    // BGR -> LUV
    cv::Mat              flower_luv;
    cv::cvtColor(flower, flower_luv, CV_BGR2Luv);

    // Perform the steps described in the exercise sheet

    cv::Mat seg_flower;
    double sp_radius = 8.;
    double col_radius = 8.;

    cv::pyrMeanShiftFiltering(flower_luv, seg_flower, sp_radius, col_radius);
    
    cv::imshow("pyrMeanShiftFiltering", seg_flower);
    cv::waitKey(0);

    //floodFillPostprocess( seg_flower, Scalar::all(2) );
    cv::Scalar colorDiff = cv::Scalar::all(2);
    cv::RNG rng = cv::theRNG();
    cv::Mat mask( seg_flower.rows+2, seg_flower.cols+2, CV_8UC1, cv::Scalar::all(0) );
    for(int y = 0; y < seg_flower.rows; ++y)
    {
        for(int x = 0; x < seg_flower.cols; ++x)
        {
            if( mask.at<uchar>(y+1, x+1) == 0 )
            {
                cv::Scalar new_color(rng(256), rng(256), rng(256));
                cv::floodFill(seg_flower, mask, cv::Point(x,y), new_color, 0, colorDiff, colorDiff);
            }
        }
    }
    // Show results
    // using **cv::imshow and cv::waitKey()** and when necessary **std::cout**
    // In the end, after the last cv::waitKey(), use **cv::destroyAllWindows()**
    cv::imshow("Mean-Shift segmented image", seg_flower);
    cv::waitKey(0);

    cv::destroyAllWindows();
}