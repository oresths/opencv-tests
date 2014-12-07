//============================================================================
// Name        : opencv-tests.cpp
// Author      : ore
// Version     :
// Copyright   : cp
// Description : Hello World in C++, Ansi-style
//============================================================================

//#include "opencv2/core/core.hpp"
//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/opencv.hpp"
#include "iostream"

using namespace cv;
using namespace std;

int main()
{
//	unsigned char m[5][5] = {{1,2,3,4,5}, {6,7,8,9,10}, {11,12,13,14,15}, {16,17,18,19,20}, {21,22,23,24,25}};
//	Mat ky = (Mat_<signed char>(1,3) << -2, 0, 2);
	Mat ky = (Mat_<float>(1,3) << -1, 0, 1);
	Mat kx = (Mat_<unsigned char>(1,3) << 3, 10, 3);
//
//	Mat dst;
//
//	sepFilter2D(Mat(5, 5, CV_8U, m), dst, CV_16S, kx, ky, Point(-1, -1), 0, BORDER_DEFAULT);

    Mat src1;
//    src1 = imread("/home/odroid/Pictures/xar.jpg", IMREAD_COLOR);
    src1 = imread("/home/odroid/Pictures/people.jpg", IMREAD_COLOR);

    Mat grey;
    cvtColor(src1, grey, COLOR_BGR2GRAY);

    Mat sobelx;

//    Size ksize; ksize.width=3, ksize.height = 3;
//    GaussianBlur(grey, sobelx, ksize, 1, 0, BORDER_DEFAULT);
//    getGaussianKernel()

    Mat fgrey;

    double exec_time = (double)getTickCount();

    ksize.width=5, ksize.height = 5;
//    GaussianBlur(grey, fgrey, ksize, 1, 0, BORDER_DEFAULT);
//    bilateralFilter(grey, fgrey, 5, 50, 50, BORDER_DEFAULT);

//    Sobel(grey, sobelx, CV_16S, 1, 0, -1);//x Scharr
//    Sobel(grey, sobelx, CV_16S, 0, 1, -1);//y Scharr kernel_row=[3, 10, 3]
//    sepFilter2D(fgrey, sobelx, CV_16S, kx, ky, Point(-1, -1), 0, BORDER_DEFAULT);
    Canny(grey, sobelx, 10, 30);

    exec_time = ((double)getTickCount() - exec_time)*1000./getTickFrequency();
	cout << "exec_time = " << exec_time << " ms";


    double minVal, maxVal;
    minMaxLoc(sobelx, &minVal, &maxVal); //find minimum and maximum intensities
//    cout << "minVal : " << minVal << endl << "maxVal : " << maxVal << endl;

    Mat draw;
    sobelx.convertTo(draw, CV_8U, 255.0/(maxVal - minVal), -minVal * 255.0/(maxVal - minVal));

    imwrite("/home/odroid/Pictures/edges.jpg", draw);

    return 0;
}
