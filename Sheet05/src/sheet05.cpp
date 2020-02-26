#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>

#define PI 3.14159
std::string PATH_Image   = "./images/gnome.png";
cv::Rect bb_Image(92,65,105,296);

void part1__1(const cv::Mat& img, const cv::Mat& mask_fg, const cv::Mat& mask_bg);
void part1__2(const cv::Mat& img, const cv::Mat& mask_fg, const cv::Mat& mask_bg);

////////////////////////////////////
// class declaration for task 1_1 //
////////////////////////////////////

class GMM_opencv{
private:
    int num_clus;
    cv::Mat_<double> samples;               // add more variables if necessary
    cv::EM Em_object;
public:
    GMM_opencv();
    ~GMM_opencv();
    void init(const int nmix, const cv::Mat& img, const cv::Mat& mask);
    void learnGMM();
    cv::Mat return_posterior(const cv::Mat& img);
};


GMM_opencv::GMM_opencv(): num_clus(0), samples(cv::Mat_<double>()) {}

GMM_opencv::~GMM_opencv() {}
    
void GMM_opencv::init(const int nmix, const cv::Mat& img, const cv::Mat& mask)
{
    std::cout << "Opencv GMM init function\n";
    std::flush(std::cout);

    num_clus = nmix;

    int rows = img.rows;
    int cols = img.cols;

    cv::Mat img_new, img_1C;
    img.copyTo(img_new, mask);

    img_new.convertTo(img_1C, CV_64FC1, (1./255.));
    samples = img_1C.reshape(1, rows*cols);

    Em_object = cv::EM(num_clus);//, cv::EM::COV_MAT_DIAGONAL, cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 50, 0.1));
}

void GMM_opencv::learnGMM()
{
    std::cout << "Opencv GMM learnGMM function\n";
    std::flush(std::cout);
    
    bool t = Em_object.train(samples);
    assert(t);
}

cv::Mat GMM_opencv::return_posterior(const cv::Mat& img)
{
    std::cout << "Opencv GMM return_posterior function\n";
    std::flush(std::cout);
    
    int rows = img.rows;
    int cols = img.cols;
    
    cv::Mat img_1C;
    img.convertTo(img_1C, CV_64FC1, 1. / 255.);

    cv::Mat temp(img_1C.size(), CV_64FC1);
    cv::Mat post_prob;
    
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            cv::Vec3d sample = img_1C.at<cv::Vec3d>(i,j);
            cv::Vec2d result = Em_object.predict(sample, post_prob);
            temp.at<double>(i,j) = result[0]; //post_prob.at<double>((int)result[1]);
        }
    }    
    return temp;
}

////////////////////////////////////
// class declaration for task 1_2 //
////////////////////////////////////

class GMM_custom{
private:
    int num_clus;
    cv::Mat wt;                         // cache for E step + final model
    cv::Mat mu;
    std::vector<cv::Mat> cov;
    cv::Mat samples;                    // training pixel samples
    cv::Mat_<double> posterior;         // posterior probability for M step
    int maxIter;

    double performEM();                   // iteratively called by learnGMM()
    double calculate_loglikelihood(cv::Vec3d sample);
public:
    GMM_custom();
    ~GMM_custom();
    void init(const int nmix, const cv::Mat& img, const cv::Mat& mask); // call this once per image
    void learnGMM();    // call this to learn GMM
    cv::Mat return_posterior(const cv::Mat& img);     // call this to generate probability map
};

GMM_custom::GMM_custom(): num_clus(0), maxIter(0) {}

GMM_custom::~GMM_custom()
{}

void GMM_custom::init(const int nmix, const cv::Mat& img, const cv::Mat& mask) // call this once per image
{
    std::cout << "GMM_custom init function\n";
    std::flush(std::cout);

    // initialization of the parameters    
    int rows = img.rows;
    int cols = img.cols; 
    int num_samples = rows*cols;

    maxIter = 5;
    num_clus = nmix;
    
    wt = cv::Mat(1, num_clus, CV_64FC1);
    cov.resize(num_clus);
    posterior = cv::Mat(img.size(), CV_64FC1);

    // initialization of the samples
    cv::Mat img_new, img_masked;
    img.convertTo(img_new, CV_32F, (1./255.));
    img_new.copyTo(img_masked, mask);
    samples = img_masked.reshape(1, num_samples);

    // initialize the GMM with k-means
    cv::Mat labels, centers;

    cv::kmeans(samples, num_clus, labels, 
        cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 10, 0.1), 
        10, cv::KMEANS_PP_CENTERS, centers);
    
    centers.convertTo(mu, CV_64FC1);
    cv::Mat temp;
    samples.convertTo(temp, CV_64FC1);
    samples = temp;

    // compute the weights and the covariance
    for(int i = 0; i < num_clus; ++i)
    {
        // cluster the data samples according to kmeans result
        cv::Mat clusterSamples;
        for(int sampleIndex = 0; sampleIndex < num_samples; sampleIndex++)
        {
            if(labels.at<int>(sampleIndex) == i)
                clusterSamples.push_back(samples.row(sampleIndex));
        }
        // calculate the convariance matrix using the opencv function
        cv::calcCovarMatrix(clusterSamples, cov[i], mu.row(i),
            CV_COVAR_NORMAL + CV_COVAR_ROWS + CV_COVAR_USE_AVG + CV_COVAR_SCALE, CV_64FC1);

        // initialize weights by the size of each cluster
        wt.at<double>(i) = (double)clusterSamples.rows / (double)num_samples;
    }
}

void GMM_custom::learnGMM()    // call this to learn GMM
{
    std::cout << "GMM_custom learnGMM function\n";
    std::flush(std::cout);

    int iter = 0;
    double log_likelihood, previous_loglikelihood = 0.;
    do 
    {
        log_likelihood = performEM();
        ++iter;
    } while (iter < maxIter); //abs(log_likelihood - previous_loglikelihood) < 1e-10) ;
}

cv::Mat GMM_custom::return_posterior(const cv::Mat& img)     // call this to generate probability map
{
    std::cout << "GMM_custom return_posterior function\n";
    std::flush(std::cout);
    
    int rows = img.rows;
    int cols = img.cols;
    
    cv::Mat img_1C;
    img.convertTo(img_1C, CV_64FC1, 1. / 255.);

    cv::Mat temp(img_1C.size(), CV_64FC1);
    cv::Mat post_prob;
    
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            cv::Vec3d sample = img_1C.at<cv::Vec3d>(i,j);
            temp.at<double>(i,j) = calculate_loglikelihood(sample); 
        }
    }    
    return temp;
}

double GMM_custom::performEM()
{   
    std::cout << "\tGMM_custom performEM function\n";
    std::flush(std::cout);

    std::vector<cv::Mat> R(num_clus);                   // the responsibility of the ith Gaussian for the jth data point
    cv::Mat sum_R(samples.rows, 1, CV_64FC1, 0.);       // the sum of each R[i] mat over all clusters
    double log_likelihood = 0.;                         // sum of all the log likelihoods over all clusters
    
    // E-Step
    for (int i = 0; i < num_clus; ++i)
    {
        cv::Mat inv_cov;
        cv::invert(cov[i], inv_cov, cv::DECOMP_SVD);

        double det = sqrt(cv::determinant(cov[i]));

        R[i] = cv::Mat(samples.rows, 1, CV_64FC1);

        for (int j = 0; j < samples.rows; ++j)
        {
            cv::Mat shift = cv::Mat(samples.at<cv::Vec3d>(j) - mu.at<cv::Vec3d>(i));

            cv::Mat prod = (shift.t() * inv_cov) * shift;
            double norm = exp(-prod.at<double>(0,0));
            
            R[i].at<double>(j) = wt.at<double>(i) * norm / (pow(2*PI, 1.5)*sqrt(det));
        }
        sum_R += R[i];
    }

    for (int i = 0; i < num_clus; ++i)
        R[i] /= sum_R;


    // M-Step
    double sum_Ri = 0.;
    for (int i = 0; i < num_clus; ++i)
    {
        double sum_Rj = cv::sum(R[i])[0];
        sum_Ri += sum_Rj;

        cv::Vec3b sum_Rx = cv::Vec3d(0.);

        for (int j = 0; j < samples.rows; ++j)
        {
            sum_Rx += R[i].at<double>(j) * samples.at<cv::Vec3d>(j);

            cv::Mat shift = cv::Mat(samples.at<cv::Vec3d>(j) - mu.at<cv::Vec3d>(i));
            cov[i] += R[i].at<double>(j) * shift * shift.t();
        }

        wt.at<double>(i) = sum_Rj;
        mu.at<cv::Vec3d>(i) = sum_Rx/sum_Rj;
        cov[i] /= sum_Rj;
    }
    wt = wt / sum_Ri;

    for (int j = 0; j < samples.rows; ++j)
    {
        cv::Vec3d sample = samples.at<cv::Vec3d>(j);
        log_likelihood += calculate_loglikelihood(sample);
    }    
    return log_likelihood;
}

double GMM_custom::calculate_loglikelihood (cv::Vec3d sample)
{
    double log_likelihood = 0.;

    for (int i = 0; i < num_clus; ++i)
    {
        double det = cv::determinant(cov[i]);
        
        cv::Mat inv_cov;
        cv::invert(cov[i], inv_cov, cv::DECOMP_SVD);

        cv::Mat shift = cv::Mat(sample - mu.at<cv::Vec3d>(i));
        cv::Mat prod = (shift.t() * inv_cov) * shift;

        log_likelihood += log(wt.at<double>(i)) - 0.5 * ( 3 * log(2*PI) - log(det) - prod.at<double>(0));
    }
    return log_likelihood;
}

////////////////////////////////////
// 2_* and 3 are theoretical work //
////////////////////////////////////

int main()
{

    // Uncomment the part of the exercise that you wish to implement.
    // For the final submission all implemented parts should be uncommented.
    cv::Mat img = cv::imread(PATH_Image);
    assert
    (img.rows*img.cols>0);
    
    cv::Mat mask_fg(img.rows,img.cols,CV_8U);
    mask_fg.setTo(0); 
    mask_fg(bb_Image).setTo(255);
    
    cv::Mat mask_bg(img.rows,img.cols,CV_8U); 
    mask_bg.setTo(255); 
    mask_bg(bb_Image).setTo(0);
    
    cv::Mat show=img.clone();
    
    cv::rectangle(show,bb_Image,cv::Scalar(0,0,255),1);
    cv::imshow("Image",show);
    cv::imshow("mask_fg",mask_fg);
    cv::imshow("mask_bg",mask_bg);
    cv::waitKey(0);

    part1__1(img,mask_fg,mask_bg);
    part1__2(img,mask_fg,mask_bg);

    std::cout <<                                                                                                   std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////    END    /////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout <<                                                                                                   std::endl;

}


void part1__1(const cv::Mat& img, const cv::Mat& mask_fg, const cv::Mat& mask_bg)
{
    std::cout <<                                                                                                   std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////    Part 1__1  /////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout <<                                                                                                   std::endl;

    
    int nmix = 10;

    GMM_opencv gmm_fg;
    gmm_fg.init(nmix,img,mask_fg);
    gmm_fg.learnGMM();
    cv::Mat fg = gmm_fg.return_posterior(img);

    GMM_opencv gmm_bg;
    gmm_bg.init(nmix,img,mask_bg);
    gmm_bg.learnGMM();
    cv::Mat bg=gmm_bg.return_posterior(img);

    cv::Mat show=bg+fg;
    cv::divide(bg,show,show);
    show.convertTo(show,CV_8U,255);
    cv::imshow("gmm_opencv",show);
    cv::waitKey(0);

    cv::destroyAllWindows();
}


void part1__2(const cv::Mat& img, const cv::Mat& mask_fg, const cv::Mat& mask_bg)
{
    std::cout <<                                                                                                   std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////    Part 1__2 //////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout <<                                                                                                   std::endl;

    int nmix = 10;

    GMM_custom gmm_fg;
    gmm_fg.init(nmix,img,mask_fg);
    gmm_fg.learnGMM();
    cv::Mat fg=gmm_fg.return_posterior(img);

    GMM_custom gmm_bg;
    gmm_bg.init(nmix,img,mask_bg);
    gmm_bg.learnGMM();
    cv::Mat bg=gmm_bg.return_posterior(img);

    cv::Mat show=bg+fg;
    cv::divide(bg,show,show);
    show.convertTo(show,CV_8U,255);
    cv::imshow("gmm_custom",show);
    cv::waitKey(0);

    cv::destroyAllWindows();
}





