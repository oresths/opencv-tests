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
//	Mat kx = (Mat_<signed char>(1,3) << -1, 0, 1);
//	Mat ky = (Mat_<signed char>(1,3) << 1, 2, 1);

	Mat kx = (Mat_<float>(1,5) << 0.0708, 0.2445, 0.3694, 0.2445, 0.0708);
	Mat ky = (Mat_<float>(1,5) << 0.0708, 0.2445, 0.3694, 0.2445, 0.0708);

//	Mat dst;
//
//	sepFilter2D(Mat(5, 5, CV_8U, m), dst, CV_16S, kx, ky, Point(-1, -1), 0, BORDER_DEFAULT);

    Mat src1;
//    src1 = imread("/home/odroid/Pictures/xar.jpg", IMREAD_COLOR);
    src1 = imread("/home/odroid/Pictures/people.jpg", IMREAD_COLOR);

    Mat grey;
    cvtColor(src1, grey, COLOR_BGR2GRAY);

    Mat sobelx;

    Mat fgrey;

//    GaussianBlur(grey, fgrey, Size(5,5), 1.1, 0);
//    blur(grey, fgrey, Size(20,20));
//    getGaussianKernel()

    double exec_time = (double)getTickCount();

//    GaussianBlur(grey, fgrey, Size(5,5), 33, 0);
//    bilateralFilter(grey, fgrey, 5, 50, 50);

//    Sobel(grey, sobelx, CV_16S, 1, 0, -1);//x Scharr
//    Sobel(grey, sobelx, CV_16S, 0, 1, -1);//y Scharr kernel_row=[3, 10, 3]
    sepFilter2D(grey, sobelx, CV_8U, kx, ky, Point(-1, -1), 0, BORDER_DEFAULT);
//    Canny(fgrey, sobelx, 10, 30);

    exec_time = ((double)getTickCount() - exec_time)*1000./getTickFrequency();
	cout << "exec_time = " << exec_time << " ms" << endl;


    double minVal, maxVal;
    minMaxLoc(sobelx, &minVal, &maxVal); //find minimum and maximum intensities
//    cout << "minVal : " << minVal << endl << "maxVal : " << maxVal << endl;

    Mat draw;
    sobelx.convertTo(draw, CV_8U, 255.0/(maxVal - minVal), -minVal * 255.0/(maxVal - minVal));

    imwrite("/home/odroid/Pictures/edges.jpg", draw);
    imwrite("/home/odroid/Pictures/blur.jpg", fgrey);

    return 0;
}
