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
    Mat src1;
    src1 = imread("/home/odroid/Pictures/lenna.jpg", IMREAD_COLOR);
//    namedWindow( "Original image", CV_WINDOW_AUTOSIZE );
//    imshow( "Original image", src1 );

    Mat grey;
    cvtColor(src1, grey, COLOR_BGR2GRAY);

    Mat sobelx;
    Sobel(grey, sobelx, CV_32F, 1, 0);

    double minVal, maxVal;
    minMaxLoc(sobelx, &minVal, &maxVal); //find minimum and maximum intensities
    cout << "minVal : " << minVal << endl << "maxVal : " << maxVal << endl;

    Mat draw;
    sobelx.convertTo(draw, CV_8U, 255.0/(maxVal - minVal), -minVal * 255.0/(maxVal - minVal));

//    namedWindow("image", CV_WINDOW_AUTOSIZE);
//    imshow("image", draw);

//    waitKey(0);
    return 0;
}
