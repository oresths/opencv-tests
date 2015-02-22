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
#include <iostream>
#include <fstream>
#include<errno.h>

using namespace cv;
using namespace std;

std::string DescribeIosFailure(const std::ios& stream);

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
	Mat ky = (Mat_<signed char>(1,3) << -1, 0, 1);
	Mat kx = (Mat_<signed char>(1,3) << 1, 2, 1);

//    Mat ky = (Mat_<signed char>(1, 5) << -5, -2, 0, 2, 5);
//    Mat kx = (Mat_<signed char>(1, 5) << 1, 3, 10, 3, 1);

//symmrowsmall8u32s if dst=8U
//	Mat kx = (Mat_<float>(1,5) << 0.0708, 0.2445, 0.3694, 0.2445, 0.0708);
//	Mat ky = (Mat_<float>(1,5) << 0.0708, 0.2445, 0.3694, 0.2445, 0.0708);

//if (dst==16S && 1<<bits && accept non-integer) symmrowsmall8u32s
//	Mat kx = (Mat_<float>(1,5) << 0.1, 0.2408, 0.3184, 0.2408, 0.1);
//	Mat ky = (Mat_<float>(1,5) << -0.9432, -1.1528, 0, 1.1528, 0.9432);

//symmrowsmall8u32s if dst=8U and replace KERNEL_SMOOTH+KERNEL_SYMMETRICAL -> KERNEL_ASYMMETRICAL
//	Mat kx = (Mat_<float>(1,5) << -0.9432, -1.1528, 0, 1.1528, 0.9432);
//	Mat ky = (Mat_<float>(1,5) << -0.9432, -1.1528, 0, 1.1528, 0.9432);

//	Mat dst;
//
//	sepFilter2D(Mat(5, 5, CV_8U, m), dst, CV_16S, kx, ky, Point(-1, -1), 0, BORDER_DEFAULT);


    Mat src4k, grey4k;
    src4k = imread((string(getenv("HOME")) + "/Pictures/people.jpg").c_str(), IMREAD_COLOR);
    cvtColor(src4k, grey4k, COLOR_BGR2GRAY);

    Mat src1080, grey1080;
    src1080 = imread((string(getenv("HOME")) + "/Pictures/xar.jpg").c_str(), IMREAD_COLOR);
    cvtColor(src1080, grey1080, COLOR_BGR2GRAY);

    Mat src_small, grey_small;
    src_small = imread((string(getenv("HOME")) + "/Pictures/xar.jpg").c_str(), IMREAD_COLOR);
    cvtColor(src_small, grey_small, COLOR_BGR2GRAY);

    Mat src_mess, grey_mess;
    src_mess = imread((string(getenv("HOME")) + "/Pictures/xar.jpg").c_str(), IMREAD_COLOR);
    cvtColor(src_mess, grey_mess, COLOR_BGR2GRAY);

    //preallocate matrix and give values to every element to trigger linux page fault mechanism
    //before filter operations, otherwise page faults during columnfilter would add about 28ms
    //to column filter time
    double exec_time = (double) getTickCount();
    exec_time = (double) getTickCount();
    Mat sobelx(1080, 1920, CV_8S, Scalar(0));
    exec_time = ((double) getTickCount() - exec_time) * 1000. / getTickFrequency();
    cout << "Memory alloc time = " << exec_time << " ms" << endl;

    ofstream file4k ((std::string(getenv("HOME")) + "/results4k.csv").c_str(), ios::out | ios::trunc);
    if (file4k.fail()) {
        cout << "Error: " << DescribeIosFailure(file4k) << endl;
        exit(1);
    }
    ofstream file1080 ((std::string(getenv("HOME")) + "/results1080.csv").c_str(), ios::out | ios::trunc);
    if (file1080.fail()) {
        cout << "Error: " << DescribeIosFailure(file1080) << endl;
        exit(1);
    }
    ofstream file_small ((std::string(getenv("HOME")) + "/results_small.csv").c_str(), ios::out | ios::trunc);
    if (file_small.fail()) {
        cout << "Error: " << DescribeIosFailure(file_small) << endl;
        exit(1);
    }

    GaussianBlur(grey4k, grey4k, Size(5,5), 1.1, 0);
    GaussianBlur(grey1080, grey1080, Size(5,5), 1.1, 0);
    GaussianBlur(grey_small, grey_small, Size(5,5), 1.1, 0);
    GaussianBlur(grey_mess, grey_mess, Size(5,5), 1.1, 0);

    for (int i = 0; i < loops; ++i) {

//            bilateralFilter(grey, fgrey, 5, 50, 50);
        //    blur(grey, fgrey, Size(5,5));

//            grey.convertTo(grey, CV_32F);

        //    Sobel(grey, sobelx, CV_16S, 1, 0, -1);//x Scharr
        //    Sobel(grey, sobelx, CV_16S, 0, 1, -1);//y Scharr kernel_row=[3, 10, 3]
//        sepFilter2D(grey, sobelx, CV_16S, kx, ky, Point(-1, -1), 0, BORDER_DEFAULT);
        //    sepFilter2D(grey, sobelx, CV_16S, ky, kx, Point(-1, -1), 0, BORDER_DEFAULT);

//            Canny(grey, sobelx, 100, 150, 3);

//            sobelx.release(); //to reallocate at every round

        Canny(grey4k, sobelx, 100, 150, 3, false, file4k);

        Canny(grey_mess, sobelx, 100, 150, 3); //causes sobelx reallocation and also messes the caches

        Canny(grey1080, sobelx, 100, 150, 3, false, file1080);

        Canny(grey_mess, sobelx, 100, 150, 3); //causes sobelx reallocation and also messes the caches

        Canny(grey_small, sobelx, 100, 150, 3, false, file_small);

        Canny(grey_mess, sobelx, 100, 150, 3); //causes sobelx reallocation and also messes the caches
    }
//	exec_time = ((double)getTickCount() - exec_time)*1000./getTickFrequency()/loops;
//	cout << "average exec_time = " << exec_time << " ms" << endl;

    file4k.close();
    file1080.close();
    file_small.close();

    return 0;
}

std::string DescribeIosFailure(const std::ios& stream)
{
  std::string result;

  if (stream.eof()) {
    result = "Unexpected end of file.";
  }

#ifdef WIN32
  // GetLastError() gives more details than errno.
  else if (GetLastError() != 0) {
    result = FormatSystemMessage(GetLastError());
  }
#endif

  else if (errno) {
#if defined(__unix__)
    // We use strerror_r because it's threadsafe.
    // GNU's strerror_r returns a string and may ignore buffer completely.
    char buffer[255];
    result = std::string(strerror_r(errno, buffer, sizeof(buffer)));
#else
    result = std::string(strerror(errno));
#endif

  }

  return result;
}
