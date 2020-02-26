#include <iostream>

#include <opencv2/opencv.hpp>


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void part1();
void part2();

std::string PATH_Ball   = "./images/ball.png";
std::string PATH_Coffee = "./images/coffee.png";


//////////////////////////////////////
// function declarations for task 1 //
//////////////////////////////////////
void  drawSnake(             cv::Mat  img, const std::vector<cv::Point2i>& snake);
void  snakes(          const cv::Mat& img, const cv::Point2i center, const int radius, std::vector<cv::Point2i>& snake);


//////////////////////////////////////
// function declarations for task 2 //
//////////////////////////////////////
void    showGray(          const cv::Mat& img, const std::string title="Image", const int t=0);
void    showContour(       const cv::Mat& img, const cv::Mat& contour,          const int t=0);
void    levelSetContours(  const cv::Mat& img, const cv::Point2f center,        const float radius, cv::Mat& phi);
cv::Mat computeContour(    const cv::Mat& phi, const float level );


//////////////////////////////////////
// my function declerations         //
//////////////////////////////////////

cv::Mat compute_w                   (const cv::Mat& img);
void    compute_grad_w              (const cv::Mat& w, cv::Mat& grad_w_x, cv::Mat& grad_w_y);
void    compute_grad_w_sobel        (const cv::Mat& w, cv::Mat& grad_w_x, cv::Mat& grad_w_y);
cv::Mat compute_mean_curvature      (const cv::Mat& phi, const cv::Mat& w, float eps);
cv::Mat compute_mean_curvature_sobel(const cv::Mat& phi, const cv::Mat& w, float eps);
cv::Mat compute_mean_curvature_div  (const cv::Mat& phi, const cv::Mat& w, float eps);
cv::Mat compute_front_propagation   (const cv::Mat& phi, const cv::Mat& grad_w_x, const cv::Mat& grad_w_y);
cv::Mat compute_front_propagation_sobel (const cv::Mat& phi, const cv::Mat& grad_w_x, const cv::Mat& grad_w_y);

void renormalize    (cv::Mat& phi);

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


int main()
{

    // Uncomment the part of the exercise that you wish to implement.
    // For the final submission all implemented parts should be uncommented.

    //part1();
    part2();

    std::cout <<                                                                                                   std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////    END    /////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout <<                                                                                                   std::endl;

}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void part1()
{
    std::cout <<                                                                                                   std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////    Part 1    //////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout <<                                                                                                   std::endl;

    cv::Mat                                               ball;
    cv::imread( PATH_Ball  , cv::IMREAD_COLOR).convertTo( ball,   CV_32FC3, (1./255.) );
    cv::Mat                                               coffee;
    cv::imread( PATH_Coffee, cv::IMREAD_COLOR).convertTo( coffee, CV_32FC3, (1./255.) );

    std::vector<cv::Point2i>    snake;
    size_t                      radius;
    cv::Point2i                 center;

    std::cout << "ball image" << std::endl;
    // for snake initialization
    center = cv::Point2i( ball.cols/2, ball.rows/2 );
    radius = std::min(    ball.cols/3, ball.rows/3 );
    //////////////////////////////////////
    snakes( ball, center, radius, snake );
    //////////////////////////////////////

    std::cout << "coffee image" << std::endl;
    // for snake initialization
    center = cv::Point2i( coffee.cols/2, coffee.rows/2 );
    radius = std::min(    coffee.cols/3, coffee.rows/3 );
    ////////////////////////////////////////
    snakes( coffee, center, radius, snake );
    ////////////////////////////////////////

    cv::destroyAllWindows();
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void part2()
{
    std::cout <<                                                                                                   std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////    Part 2    //////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout <<                                                                                                   std::endl;

    cv::Mat                                               ball;
    cv::imread( PATH_Ball  , cv::IMREAD_COLOR).convertTo( ball,   CV_32FC3, (1./255.) );
    cv::Mat                                               coffee;
    cv::imread( PATH_Coffee, cv::IMREAD_COLOR).convertTo( coffee, CV_32FC3, (1./255.) );

    cv::Mat                     phi;
    size_t                      radius;
    cv::Point2i                 center;

    std::cout << "ball image" << std::endl;
    center = cv::Point2i( ball.cols/2, ball.rows/2 );
    radius = std::min(    ball.cols/3, ball.rows/3 );
    /////////////////////////////////////////////////////////
    levelSetContours(     ball,    center, radius, phi     );
    /////////////////////////////////////////////////////////

    /*std::cout << "coffee image" << std::endl;
    center = cv::Point2f( coffee.cols/2.f, coffee.rows/2.f );
    radius =    std::min( coffee.cols/3.f, coffee.rows/3.f );
    /////////////////////////////////////////////////////////
    levelSetContours(     coffee,  center, radius, phi     );
    /////////////////////////////////////////////////////////
*/
    cv::destroyAllWindows();
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////
// apply the snake algorithm to an image //
///////////////////////////////////////////
void snakes( const cv::Mat&                     img,
             const cv::Point2i                  center,
             const int                          radius,
                   std::vector<cv::Point2i>&    snake )
{
    // initialize snake with a circle
    const int     vvvTOTAL =  radius*CV_PI/7; // defines number of snake vertices // adaptive based on the circumference
    snake.resize( vvvTOTAL );
    float angle = 0;
    for (cv::Point2i& vvv: snake)
    {
        vvv.x = round( center.x + cos(angle)*radius );
        vvv.y = round( center.y + sin(angle)*radius );

        angle += 2*CV_PI/vvvTOTAL;
    }

    // visualization
    cv::Mat     vis;
    img.copyTo( vis );
    drawSnake(  vis, snake);
    ///////////////////////////////////////////////////////////
    std::cout << "Press any key to continue...\n" << std::endl;
    ///////////////////////////////////////////////////////////
    cv::imshow("Snake", vis);
    cv::waitKey();

    // Perform optimization of the initialized snake as described in the exercise sheet and the slides.
    // You might want to apply some GaussianBlur on the edges so that the snake sidles up better.
    // Iterate until
    // - optimal solution for every point is the center of a 3x3 (or similar) box, OR
    // - until maximum number of iterations is reached

    // At each step visualize the current result
    // using **drawSnake() and cv::waitKey(10)** as in the example above and when necessary **std::cout**
    // In the end, after the last visualization, use **cv::destroyAllWindows()**

    cv::destroyAllWindows();
}


////////////////////////////////
// draws a snake on the image //
////////////////////////////////
void drawSnake(       cv::Mat                   img,
                const std::vector<cv::Point2i>& snake )
{
    const size_t siz = snake.size();

    for (size_t iii=0; iii<siz; iii++)
        cv::line( img, snake[iii], snake[(iii+1)%siz], cv::Scalar(0,0,1) );

    for (const cv::Point2i& p: snake)
        cv::circle( img, p, 2, cv::Scalar(1,0,0), -1 );
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////
// computes the metric for geodesic active contours     //
//////////////////////////////////////////////////////////
cv::Mat compute_w (const cv::Mat& img)
{
    cv::Mat blurred_img;
    cv::GaussianBlur(img, blurred_img, cv::Size(3,3), 0);
    cv::Mat grad_img_x  (img.size(), img.type()),
            grad_img_y  (img.size(), img.type()),
            grad        (img.size(), img.type());
            //w           (phi.size(), phi.type())

    cv::Mat gray_img;
    cv::cvtColor(blurred_img, gray_img, cv::COLOR_BGR2GRAY);
    cv::Sobel(gray_img, grad_img_x, -1, 1, 0);
    cv::Sobel(gray_img, grad_img_y, -1, 0, 1);

    cv::pow(grad_img_x, 2, grad_img_x);
    cv::pow(grad_img_y, 2, grad_img_y);

    cv::sqrt(grad_img_x + grad_img_y, grad);

    return 1./(1. + grad);
}

///////////////////////////////////////////////////////////
// computes the gradients of w                           //
///////////////////////////////////////////////////////////
void compute_grad_w(const cv::Mat& w, cv::Mat& grad_w_x, cv::Mat& grad_w_y)
{
    cv::Matx31f kernel_x    (-0.5,  0.,   0.5);     
    cv::filter2D(w, grad_w_x, -1, kernel_x);
    cv::filter2D(w, grad_w_y, -1, kernel_x.t());
}

///////////////////////////////////////////////////////////
// computes the gradients of w with cv::Sobel            //
///////////////////////////////////////////////////////////
void compute_grad_w_sobel(const cv::Mat& w, cv::Mat& grad_w_x, cv::Mat& grad_w_y)
{
    cv::Sobel(w, grad_w_x, -1, 1, 0);
    cv::Sobel(w, grad_w_y, -1, 0, 1);
}

///////////////////////////////////////////////////////////
// computes the mean curvature motion with the scheme    //
// given in the lecture notes                            //
///////////////////////////////////////////////////////////
cv::Mat compute_mean_curvature (const cv::Mat& phi, const cv::Mat& w, float eps)
{
    //cv::Matx31f kernel_x    (-0.5,  0.,   0.5); 
    cv::Vec3f kernel_x      (-0.5,  0.,   0.5); 
    //cv::Matx31f kernel_xx   (0.25, -0.5, 0.25); 
    cv::Vec3f kernel_xx     (0.25, -0.5, 0.25); 
    cv::Matx33f kernel_xy   (0.25,  0.,  -0.25,
                             0.,    0.,   0.,
                            -0.25,  0.,   0.25);

    cv::Mat phi_x       (phi.size(), phi.type()),
            phi_xx      (phi.size(), phi.type()),
            phi_y       (phi.size(), phi.type()),
            phi_yy      (phi.size(), phi.type()),
            phi_xy      (phi.size(), phi.type());

    cv::filter2D(phi, phi_x, -1, kernel_x);
    cv::filter2D(phi, phi_y, -1, kernel_x.t());        
    cv::filter2D(phi, phi_xx, -1, kernel_xx);
    cv::filter2D(phi, phi_yy, -1, kernel_xx.t());       
    cv::filter2D(phi, phi_xy, -1, kernel_xy);

    cv::Mat phi_x2, phi_y2;
    cv::pow(phi_x, 2, phi_x2);
    cv::pow(phi_y, 2, phi_y2);

    cv::Mat nominator   = phi_xx.mul( phi_y2) - 2 * phi_x.mul( phi_y.mul(phi_xy) ) + phi_yy.mul( phi_x2 );
    cv::Mat denominator = phi_x2 + phi_y2 + eps; 

    return w.mul(nominator / denominator);
}

///////////////////////////////////////////////////////////////
// computes the mean curvature motion with the scheme        //
// given in the lecture notes but with cv::Sobel derivatives //
///////////////////////////////////////////////////////////////
cv::Mat compute_mean_curvature_sobel (const cv::Mat& phi, const cv::Mat& w, float eps)
{
    cv::Mat phi_x       (phi.size(), phi.type()),
            phi_xx      (phi.size(), phi.type()),
            phi_y       (phi.size(), phi.type()),
            phi_yy      (phi.size(), phi.type()),
            phi_xy      (phi.size(), phi.type());

    cv::Sobel(phi, phi_x, -1, 1, 0);
    cv::Sobel(phi, phi_y, -1, 0, 1);
    cv::Sobel(phi, phi_xx, -1, 2, 0);
    cv::Sobel(phi, phi_yy, -1, 0, 2);
    cv::Sobel(phi, phi_xy, -1, 1, 1);

    cv::Mat phi_x2, phi_y2;
    cv::pow(phi_x, 2, phi_x2);
    cv::pow(phi_y, 2, phi_y2);

    cv::Mat nominator   = phi_xx.mul( phi_y2 ) - 2 * phi_x.mul( phi_y.mul(phi_xy) ) + phi_yy.mul( phi_x2 );
    cv::Mat denominator = phi_x2 + phi_y2 + eps; 

    return w.mul(nominator / denominator);
}

////////////////////////////////////////////////////////////////////////////////////////////
// calculates the mean curvature motion without the scheme provided in the lecture notes, //
// instead calculates with the help of the divergence of (grad_phi/absolute_grad_phi)     //
////////////////////////////////////////////////////////////////////////////////////////////
cv::Mat compute_mean_curvature_div (const cv::Mat& phi, const cv::Mat& w, float eps)
{
    cv::Mat phi_x, phi_y;
    cv::Mat abs_grad_phi;

    cv::Sobel(phi, phi_x, -1, 1, 0);
    cv::Sobel(phi, phi_y, -1, 0, 1);
    cv::sqrt(phi_x.mul(phi_x) + phi_y.mul(phi_y), abs_grad_phi);

    // normalize the gradients and get rid of singularities
    cv::Mat norm_phi_x = phi_x / (abs_grad_phi + eps);
    cv::Mat norm_phi_y = phi_y / (abs_grad_phi + eps);

    cv::Mat div_grad_x, div_grad_y;

    cv::Sobel(norm_phi_x, div_grad_x, -1, 1, 0);
    cv::Sobel(norm_phi_y, div_grad_y, -1, 0, 1);

    // calculate the curvature 
    cv::Mat K = div_grad_x + div_grad_y;

    return w.mul( abs_grad_phi.mul(K) );
}

///////////////////////////////////////////////////////////
// computes the front propaagtion term upwind scheme     //
///////////////////////////////////////////////////////////
cv::Mat  compute_front_propagation   (const cv::Mat& phi, const cv::Mat& grad_w_x, const cv::Mat& grad_w_y)
{
    cv::Matx31f kernel_plus (0.,   -0.5,  0.5);
    cv::Matx31f kernel_minus(-0.5,  0.5,  0.);

    cv::Mat phi_x_plus  (phi.size(), phi.type()),
            phi_x_minus (phi.size(), phi.type()),
            phi_y_plus  (phi.size(), phi.type()),
            phi_y_minus (phi.size(), phi.type());

    cv::filter2D(phi, phi_x_plus,   -1, kernel_plus);
    cv::filter2D(phi, phi_x_minus,  -1, kernel_minus);
    cv::filter2D(phi, phi_y_plus,   -1, kernel_plus.t());
    cv::filter2D(phi, phi_y_minus,  -1, kernel_minus.t());

    return      phi_x_minus.mul(cv::max(grad_w_x, 0.)) + phi_x_plus.mul(cv::min(grad_w_x, 0.))
            +   phi_y_minus.mul(cv::max(grad_w_y, 0.)) + phi_y_plus.mul(cv::min(grad_w_y, 0.));
}

///////////////////////////////////////////////////////////
// computes the front propaagtion term no upwind,        //
// regular sobel derivatives                             //
///////////////////////////////////////////////////////////
cv::Mat  compute_front_propagation_sobel (const cv::Mat& phi, const cv::Mat& grad_w_x, const cv::Mat& grad_w_y)
{
    cv::Mat phi_x, phi_y;

    cv::Sobel(phi, phi_x, -1, 1, 0);
    cv::Sobel(phi, phi_y, -1, 0, 1);

    return grad_w_x.mul(phi_x) + grad_w_y.mul(phi_y);
}

void renormalize    (cv::Mat& phi)
{
    cv::Mat a = (phi != 0) / 255;
    cv::distanceTransform(a, phi, CV_DIST_L2, 3);
}

///////////////////////////////////////////////////////////
// runs the level-set geodesic active contours algorithm //
///////////////////////////////////////////////////////////
void levelSetContours( const cv::Mat& img, const cv::Point2f center, const float radius, cv::Mat& phi )
{
    phi.create( img.size(), CV_32FC1 );
    //////////////////////////////
    // signed distance map **phi**
    //////////////////////////////
    // initialize as a cone around the
    // center with phi(x,y)=0 at the radius
    for (int y=0; y<phi.rows; y++)   
    {   
        const  float disty2 = pow( y-center.y, 2 );
        for (int x=0; x<phi.cols; x++)       
            phi.at<float>(y,x) = disty2 + pow( x-center.x, 2 );   
    }
                              
    cv::sqrt( phi, phi );

    // positive values inside
    phi = (radius - phi);
    cv::Mat contour = computeContour( phi, 0.0f);

    ///////////////////////////////////////////////////////////
    std::cout << "Press any key to continue...\n" << std::endl;
    ///////////////////////////////////////////////////////////
    showGray(    phi, "phi", 1 );
    showContour( img, contour,  0 );
    /////////////////////////////

    // Perform optimization of the initialized level-set function with geodesic active contours as described in the exercise sheet and the slides.
    // Iterate until
    // - the contour does not change between 2 consequitive iterations, or
    // - until a maximum number of iterations is reached
    cv::Mat w = compute_w(img);
    cv::Mat grad_w_x    (phi.size(), phi.type()),
            grad_w_y    (phi.size(), phi.type());
    
    //compute_grad_w(w, grad_w_x, grad_w_y);
    compute_grad_w_sobel(w, grad_w_x, grad_w_y);

    double  max_w;
    cv::minMaxLoc(w, NULL, &max_w);

    float   tau = cv::min(0.5, 1./(4*max_w));
    float   eps = 1e-4;
    int     iterations = 500;
    int     k = 0;
    double  maxVal = 1.;

    while(k < iterations )//&& maxVal != 0.)
    {
        // all these methods were implemented and tested, because of ambiguity in the lecture notes
        // none provide the correct results
        cv::Mat mean_curvature          = compute_mean_curvature        (phi, w, eps);
        cv::Mat mean_curvature_sobel    = compute_mean_curvature_sobel  (phi, w, eps);
        cv::Mat mean_curvature_div      = compute_mean_curvature_div    (phi, w, eps);
        cv::Mat front_propagation       = compute_front_propagation     (phi, grad_w_x, grad_w_y);
        cv::Mat front_propagation_sobel = compute_front_propagation_sobel(phi, grad_w_x, grad_w_y);

        phi += tau * (mean_curvature_sobel + front_propagation);
        cv::Mat new_contour = computeContour( phi, 0.0f);
        
        cv::Mat diff;
        cv::absdiff(contour, new_contour, diff);
        cv::minMaxLoc(diff, NULL, &maxVal);
        showContour( img, new_contour, 0 );

        if(k % 100 == 0)
        {
            //renormalize(phi);
            std::cout << "Iteration no " << k << "\n";
            showGray(    phi, "phi", 1 );
            showContour( img, new_contour, 0 );
            std::cout << "Diff between the current and previous contours: " << maxVal << "\n";            
        }
        contour = new_contour;
        ++k;

        // At each step visualize the current result
        // using **showGray() and showContour()** as in the example above and when necessary **std::cout**
        // In the end, after the last visualization, use **cv::destroyAllWindows()**        
    }
    cv::destroyAllWindows();
}


////////////////////////////
// show a grayscale image //
////////////////////////////
void showGray( const cv::Mat& img, const std::string title, const int t )
{
    CV_Assert( img.channels() == 1 );

    double               minVal,  maxVal;
    cv::minMaxLoc( img, &minVal, &maxVal );

    cv::Mat            temp;
    img.convertTo(     temp, CV_32F, 1./(maxVal-minVal), -minVal/(maxVal-minVal));
    cv::imshow( title, temp);
    cv::waitKey(t);
}


//////////////////////////////////////////////
// compute the pixels where phi(x,y)==level //
//////////////////////////////////////////////
cv::Mat computeContour( const cv::Mat& phi, const float level )
{
    CV_Assert( phi.type() == CV_32FC1 );

    cv::Mat segmented_NORMAL( phi.size(), phi.type() );
    cv::Mat segmented_ERODED( phi.size(), phi.type() );

    cv::threshold( phi, segmented_NORMAL, level, 1.0, cv::THRESH_BINARY );
    cv::erode(          segmented_NORMAL, segmented_ERODED, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size2i(3,3)) );

    return ( segmented_NORMAL != segmented_ERODED );
}


///////////////////////////
// draw contour on image //
///////////////////////////
void showContour( const cv::Mat& img, const cv::Mat& contour, const int t )
{
    CV_Assert( img.cols == contour.cols   &&
               img.rows == contour.rows   &&
               img.type()     == CV_32FC3 &&
               contour.type() == CV_8UC1  );

    cv::Mat temp( img.size(), img.type() );

    const cv::Vec3f color( 0, 0, 1 ); // BGR

    for     (int y=0; y<img.rows; y++)
        for (int x=0; x<img.cols; x++)
            temp.at<cv::Vec3f>(y,x) = contour.at<uchar>(y,x)!=255 ? img.at<cv::Vec3f>(y,x) : color;

    cv::imshow("contour", temp);
    cv::waitKey(t);
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

