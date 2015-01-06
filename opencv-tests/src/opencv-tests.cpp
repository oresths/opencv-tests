/*
//============================================================================
// Name        : opencv-tests.cpp
// Author      : ore
// Version     :
// Copyright   : cp
// Description : Hello World in C++, Ansi-style
//============================================================================

#include "opencv2/core/core.hpp"
//#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "opencv2/opencv_modules.hpp"
//#include "opencv2/opencv.hpp"
#include "iostream"

using namespace cv;
using namespace std;

int main(int argc, char **argv) {
    int loops;
    if (argc == 1) {
        loops = 1;
    } else if (argc == 2) {
        if (sscanf(argv[1], "%i", &loops) != 1) {
            printf("error - not an integer");
        }
    } else
        cout << "Wrong number of arguments" << endl;

//	unsigned char m[5][5] = {{1,2,3,4,5}, {6,7,8,9,10}, {11,12,13,14,15}, {16,17,18,19,20}, {21,22,23,24,25}};

//	//symmrowsmall8u32s - symmcolumnsmall32s16s, kx<->ky, integer kernel
//	Mat ky = (Mat_<signed char>(1,3) << -2, 0, 2);
//	Mat kx = (Mat_<signed char>(1,3) << 3, 12, 3);

//    Mat ky = (Mat_<signed char>(1, 5) << -5, -2, 0, 2, 5);
//    Mat kx = (Mat_<signed char>(1, 5) << 1, 3, 10, 3, 1);

//symmrowsmall8u32s if dst=8U
	Mat kx = (Mat_<float>(1,5) << 0.0708, 0.2445, 0.3694, 0.2445, 0.0708);
	Mat ky = (Mat_<float>(1,5) << 0.0708, 0.2445, 0.3694, 0.2445, 0.0708);

//if (dst==16S && 1<<bits && accept non-integer) symmrowsmall8u32s
//	Mat kx = (Mat_<float>(1,5) << 0.1, 0.2408, 0.3184, 0.2408, 0.1);
//	Mat ky = (Mat_<float>(1,5) << -0.9432, -1.1528, 0, 1.1528, 0.9432);

//symmrowsmall8u32s if dst=8U and replace KERNEL_SMOOTH+KERNEL_SYMMETRICAL -> KERNEL_ASYMMETRICAL
//	Mat kx = (Mat_<float>(1,5) << -0.9432, -1.1528, 0, 1.1528, 0.9432);
//	Mat ky = (Mat_<float>(1,5) << -0.9432, -1.1528, 0, 1.1528, 0.9432);

//	Mat dst;
//
//	sepFilter2D(Mat(5, 5, CV_8U, m), dst, CV_16S, kx, ky, Point(-1, -1), 0, BORDER_DEFAULT);

    Mat src1;
//    src1 = imread("/home/odroid/Pictures/xar.jpg", IMREAD_COLOR);
    src1 = imread("/home/odroid/Pictures/people.jpg", IMREAD_COLOR);

    Mat grey;
    cvtColor(src1, grey, COLOR_BGR2GRAY);

    //preallocate matrix and give values to every element to trigger linux page fault mechanism
    //before filter operations, otherwise page faults during columnfilter would add about 28ms
    //to column filter time
    Mat sobelx(grey.rows, grey.cols, CV_16S, Scalar(0));

    Mat fgrey;

    double exec_time = (double) getTickCount();
    for (int i = 0; i < loops; ++i) {
            GaussianBlur(grey, fgrey, Size(5,5), 1.1, 0);
//            bilateralFilter(grey, fgrey, 5, 50, 50);
        //    blur(grey, fgrey, Size(5,5));

//            grey.convertTo(grey, CV_32F);

        //    Sobel(grey, sobelx, CV_16S, 1, 0, -1);//x Scharr
        //    Sobel(grey, sobelx, CV_16S, 0, 1, -1);//y Scharr kernel_row=[3, 10, 3]
//        sepFilter2D(grey, sobelx, CV_16S, kx, ky, Point(-1, -1), 0, BORDER_DEFAULT);
        //    sepFilter2D(grey, sobelx, CV_16S, ky, kx, Point(-1, -1), 0, BORDER_DEFAULT);

            exec_time = (double) getTickCount();
//            Canny(src1, sobelx, 100, 150, 3);
            Canny(fgrey, sobelx, 100, 150, 3);
            exec_time = ((double) getTickCount() - exec_time) * 1000. / getTickFrequency();
            cout << "Canny exec_time = " << exec_time << " ms" << endl;
    }
//	exec_time = ((double)getTickCount() - exec_time)*1000./getTickFrequency()/loops;
//	cout << "average exec_time = " << exec_time << " ms" << endl;

    double minVal, maxVal;
    minMaxLoc(sobelx, &minVal, &maxVal); //find minimum and maximum intensities
//    cout << "minVal : " << minVal << endl << "maxVal : " << maxVal << endl;

    Mat draw;
    sobelx.convertTo(draw, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));

    Mat current, previous, test;
    //save and reload jpeg to avoid difference caused by compression
    imwrite("/home/odroid/Pictures/edges0.jpg", draw);
    current = imread("/home/odroid/Pictures/edges0.jpg", IMREAD_UNCHANGED);
//    previous = imread("/home/odroid/Pictures/edges.jpg", IMREAD_UNCHANGED);
    previous = imread("/home/odroid/Pictures/dedges100_150.jpg", IMREAD_UNCHANGED);
    if (previous.rows == current.rows && previous.cols == current.cols) {
//        test = abs( previous - current ) > 1;
        test = previous != current;
        int cnz =  cv::countNonZero(test);
        bool equal = cnz == 0;
        cout << "1 = " << equal << endl;
    }

    imwrite("/home/odroid/Pictures/edges.jpg", draw);
    imwrite("/home/odroid/Pictures/blur.jpg", fgrey);

    return 0;
}
*/

#include "opencv2/core/core.hpp"
//#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "opencv2/opencv_modules.hpp"
//#include "opencv2/opencv.hpp"
#include "iostream"

using namespace cv;
using namespace std;

int main(int argc, char **argv) {
    int loops;
    if (argc == 1) {
        loops = 1;
    } else if (argc == 2) {
        if (sscanf(argv[1], "%i", &loops) != 1) {
            printf("error - not an integer");
        }
    } else
        cout << "Wrong number of arguments" << endl;


    int count = 0;
    Mat current, previous, test;
    for (int i = 0; i < loops; ++i) {
        Mat image = imread("/home/odroid/Pictures/lenna.jpg", IMREAD_GRAYSCALE);

        if (i==0) Canny(image, previous, 0, 0);
        Canny(image, image, 0, 0);
        current = image;
        test = previous != current;
        int cnz =  cv::countNonZero(test);
        bool equal = cnz == 0;
        if (!equal)
        {
            count++;
            cout << "In loop " << i << " " << count << "th failure with " << cnz << "mismatches" << endl;
//            imwrite("/home/odroid/Pictures/isfail.jpg", image);
//            break;
        }
//        else
//        {
//            imwrite("/home/odroid/Pictures/iscorrect.jpg", image);
//            break;
//        }

        current.copyTo(previous);
    }

    return 0;
}
