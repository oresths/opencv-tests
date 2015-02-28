//============================================================================
// Name        : opencv-tests.cpp
// Author      : ore
// Version     :
// Copyright   : cp
// Description : Hello World in C++, Ansi-style
//============================================================================

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "opencv2/opencv_modules.hpp"

#include <iostream>
#include <fstream>
#include<errno.h>

#define CANNY 1 //1 for canny tests, 0 for filter tests

//filename
#define TYPE "CS" // CS (Canny Steps), CA (Canny All), F (Filters)
#define CPU "A15" // A7, A9, A15, i7
#define OPTIMIZ "P" //S (Serial), A (Auto - filters only), P (Parallel)

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

    uchar num_of_pictures = 3;
    uchar num_of_tests = 5;
    double * times = (double *) malloc(loops * num_of_tests * num_of_pictures * sizeof(double));

    Mat src4k, grey4k;
    src4k = imread((string(getenv("HOME")) + "/Pictures/people.jpg").c_str(), IMREAD_COLOR);
    cvtColor(src4k, grey4k, COLOR_BGR2GRAY);

    Mat src1080, grey1080;
    src1080 = imread((string(getenv("HOME")) + "/Pictures/trapezia.jpg").c_str(), IMREAD_COLOR);
    cvtColor(src1080, grey1080, COLOR_BGR2GRAY);

    Mat src_small, grey_small;
    src_small = imread((string(getenv("HOME")) + "/Pictures/lenna.jpg").c_str(), IMREAD_COLOR);
    cvtColor(src_small, grey_small, COLOR_BGR2GRAY);

    Mat src_mess, grey_mess;
    src_mess = imread((string(getenv("HOME")) + "/Pictures/xar.jpg").c_str(), IMREAD_COLOR);
    cvtColor(src_mess, grey_mess, COLOR_BGR2GRAY);

    Mat canny_dst;

    //preallocate matrix and give values to every element to trigger linux page fault mechanism
    //before filter operations, otherwise page faults during columnfilter would add about 28ms
    //to column filter time
    Mat filters_dst_4k(grey4k.rows, grey4k.cols, CV_16S, Scalar(0));
    Mat filters_dst_1080(grey1080.rows, grey1080.cols, CV_16S, Scalar(0));
    Mat filters_dst_small(grey_small.rows, grey_small.cols, CV_16S, Scalar(0));
    Mat filters_dst_mess(grey_mess.rows, grey_mess.cols, CV_16S, Scalar(0));

    double exec_time;
    exec_time = (double) getTickCount();
    Mat memory_allocations(1080, 1920, CV_16S, Scalar(0)); //pixels , dst type
    exec_time = ((double) getTickCount() - exec_time) * 1000. / getTickFrequency();
    cout << "Memory alloc time = " << exec_time << " ms" << endl;

    ofstream file4k((std::string(getenv("HOME")) + "/" + TYPE + "4k" + CPU + OPTIMIZ + ".csv").c_str(),
            ios::out | ios::trunc);
    if (file4k.fail()) {
        cout << "Error: " << DescribeIosFailure(file4k) << endl;
        exit(1);
    }
    ofstream file1080((std::string(getenv("HOME")) + "/" + TYPE + "1080" + CPU + OPTIMIZ + ".csv").c_str(),
            ios::out | ios::trunc);
    if (file1080.fail()) {
        cout << "Error: " << DescribeIosFailure(file1080) << endl;
        exit(1);
    }
    ofstream file_small((std::string(getenv("HOME")) + "/" + TYPE + "s" + CPU + OPTIMIZ + ".csv").c_str(),
            ios::out | ios::trunc);
    if (file_small.fail()) {
        cout << "Error: " << DescribeIosFailure(file_small) << endl;
        exit(1);
    }

    ofstream file_null("/dev/null");
    if (file_null.fail()) {
        cout << "Error: " << DescribeIosFailure(file_null) << endl;
        exit(1);
    }


    Mat canny_src_4k, canny_src_1080, canny_src_small, canny_src_mess;
    GaussianBlur(grey4k, canny_src_4k, Size(5, 5), 1.1, 0);
    GaussianBlur(grey1080, canny_src_1080, Size(5, 5), 1.1, 0);
    GaussianBlur(grey_small, canny_src_small, Size(5, 5), 1.1, 0);
    GaussianBlur(grey_mess, canny_src_mess, Size(5, 5), 1.1, 0);

    uchar resolut_index;
    int index;
    for (int i = 0; i < loops; ++i) {
#if !CANNY
        resolut_index = 0;

        exec_time = (double) getTickCount();
        Sobel(grey4k, filters_dst_4k, CV_16S, 1, 0);   //x Sobel
        exec_time = ((double) getTickCount() - exec_time) * 1000. / getTickFrequency();
        index = i * num_of_tests * num_of_pictures + resolut_index * num_of_tests + 0;
        times[index] = exec_time;
        Canny(canny_src_mess, filters_dst_mess, 100, 150, 3, false, file_null); //messes the caches

        exec_time = (double) getTickCount();
        Sobel(grey4k, filters_dst_4k, CV_16S, 0, 1);   //y Sobel
        exec_time = ((double) getTickCount() - exec_time) * 1000. / getTickFrequency();
        index = i * num_of_tests * num_of_pictures + resolut_index * num_of_tests + 1;
        times[index] = exec_time;
        Canny(canny_src_mess, filters_dst_mess, 100, 150, 3, false, file_null); //messes the caches

        exec_time = (double) getTickCount();
        Scharr(grey4k, filters_dst_4k, CV_16S, 1, 0);  //x Scharr
        exec_time = ((double) getTickCount() - exec_time) * 1000. / getTickFrequency();
        index = i * num_of_tests * num_of_pictures + resolut_index * num_of_tests + 2;
        times[index] = exec_time;
        Canny(canny_src_mess, filters_dst_mess, 100, 150, 3, false, file_null); //messes the caches

        exec_time = (double) getTickCount();
        Scharr(grey4k, filters_dst_4k, CV_16S, 0, 1);  //y Scharr
        exec_time = ((double) getTickCount() - exec_time) * 1000. / getTickFrequency();
        index = i * num_of_tests * num_of_pictures + resolut_index * num_of_tests + 3;
        times[index] = exec_time;
        Canny(canny_src_mess, filters_dst_mess, 100, 150, 3, false, file_null); //messes the caches

        exec_time = (double) getTickCount();
        GaussianBlur(grey4k, grey4k, Size(5, 5), 1.1, 0);
        exec_time = ((double) getTickCount() - exec_time) * 1000. / getTickFrequency();
        index = i * num_of_tests * num_of_pictures + resolut_index * num_of_tests + 4;
        times[index] = exec_time;
        cvtColor(src4k, grey4k, COLOR_BGR2GRAY); //restores Gaussian src
        Canny(canny_src_mess, filters_dst_mess, 100, 150, 3, false, file_null); //messes the caches

//-----------------------------------------------------------------------------------------------------------
        resolut_index = 1;

        exec_time = (double) getTickCount();
        Sobel(grey1080, filters_dst_1080, CV_16S, 1, 0);   //x Sobel
        exec_time = ((double) getTickCount() - exec_time) * 1000. / getTickFrequency();
        index = i * num_of_tests * num_of_pictures + resolut_index * num_of_tests + 0;
        times[index] = exec_time;
        Canny(canny_src_mess, filters_dst_mess, 100, 150, 3, false, file_null); //messes the caches

        exec_time = (double) getTickCount();
        Sobel(grey1080, filters_dst_1080, CV_16S, 0, 1);   //y Sobel
        exec_time = ((double) getTickCount() - exec_time) * 1000. / getTickFrequency();
        index = i * num_of_tests * num_of_pictures + resolut_index * num_of_tests + 1;
        times[index] = exec_time;
        Canny(canny_src_mess, filters_dst_mess, 100, 150, 3, false, file_null); //messes the caches

        exec_time = (double) getTickCount();
        Scharr(grey1080, filters_dst_1080, CV_16S, 1, 0);  //x Scharr
        exec_time = ((double) getTickCount() - exec_time) * 1000. / getTickFrequency();
        index = i * num_of_tests * num_of_pictures + resolut_index * num_of_tests + 2;
        times[index] = exec_time;
        Canny(canny_src_mess, filters_dst_mess, 100, 150, 3, false, file_null); //messes the caches

        exec_time = (double) getTickCount();
        Scharr(grey1080, filters_dst_1080, CV_16S, 0, 1);  //y Scharr
        exec_time = ((double) getTickCount() - exec_time) * 1000. / getTickFrequency();
        index = i * num_of_tests * num_of_pictures + resolut_index * num_of_tests + 3;
        times[index] = exec_time;
        Canny(canny_src_mess, filters_dst_mess, 100, 150, 3, false, file_null); //messes the caches

        exec_time = (double) getTickCount();
        GaussianBlur(grey1080, grey1080, Size(5, 5), 1.1, 0);
        exec_time = ((double) getTickCount() - exec_time) * 1000. / getTickFrequency();
        index = i * num_of_tests * num_of_pictures + resolut_index * num_of_tests + 4;
        times[index] = exec_time;
        cvtColor(src1080, grey1080, COLOR_BGR2GRAY); //restores Gaussian src
        Canny(canny_src_mess, filters_dst_mess, 100, 150, 3, false, file_null); //messes the caches

//-----------------------------------------------------------------------------------------------------------
        resolut_index = 2;

        exec_time = (double) getTickCount();
        Sobel(grey_small, filters_dst_small, CV_16S, 1, 0);   //x Sobel
        exec_time = ((double) getTickCount() - exec_time) * 1000. / getTickFrequency();
        index = i * num_of_tests * num_of_pictures + resolut_index * num_of_tests + 0;
        times[index] = exec_time;
        Canny(canny_src_mess, filters_dst_mess, 100, 150, 3, false, file_null); //messes the caches

        exec_time = (double) getTickCount();
        Sobel(grey_small, filters_dst_small, CV_16S, 0, 1);   //y Sobel
        exec_time = ((double) getTickCount() - exec_time) * 1000. / getTickFrequency();
        index = i * num_of_tests * num_of_pictures + resolut_index * num_of_tests + 1;
        times[index] = exec_time;
        Canny(canny_src_mess, filters_dst_mess, 100, 150, 3, false, file_null); //messes the caches

        exec_time = (double) getTickCount();
        Scharr(grey_small, filters_dst_small, CV_16S, 1, 0);  //x Scharr
        exec_time = ((double) getTickCount() - exec_time) * 1000. / getTickFrequency();
        index = i * num_of_tests * num_of_pictures + resolut_index * num_of_tests + 2;
        times[index] = exec_time;
        Canny(canny_src_mess, filters_dst_mess, 100, 150, 3, false, file_null); //messes the caches

        exec_time = (double) getTickCount();
        Scharr(grey_small, filters_dst_small, CV_16S, 0, 1);  //y Scharr
        exec_time = ((double) getTickCount() - exec_time) * 1000. / getTickFrequency();
        index = i * num_of_tests * num_of_pictures + resolut_index * num_of_tests + 3;
        times[index] = exec_time;
        Canny(canny_src_mess, filters_dst_mess, 100, 150, 3, false, file_null); //messes the caches

        exec_time = (double) getTickCount();
        GaussianBlur(grey_small, grey_small, Size(5, 5), 1.1, 0);
        exec_time = ((double) getTickCount() - exec_time) * 1000. / getTickFrequency();
        index = i * num_of_tests * num_of_pictures + resolut_index * num_of_tests + 4;
        times[index] = exec_time;
        cvtColor(src_small, grey_small, COLOR_BGR2GRAY); //restores Gaussian src
        Canny(canny_src_mess, filters_dst_mess, 100, 150, 3, false, file_null); //messes the caches
#else
        Canny(canny_src_4k, canny_dst, 100, 150, 3, false, file4k);
        Canny(canny_src_mess, canny_dst, 100, 150, 3, false, file_null); //causes canny_dst reallocation and also messes the caches

        Canny(canny_src_1080, canny_dst, 100, 150, 3, false, file1080);
        Canny(canny_src_mess, canny_dst, 100, 150, 3, false, file_null); //causes canny_dst reallocation and also messes the caches

        Canny(canny_src_small, canny_dst, 100, 150, 3, false, file_small);
        Canny(canny_src_mess, canny_dst, 100, 150, 3, false, file_null); //causes canny_dst reallocation and also messes the caches
#endif

    }
#if !CANNY
for (int i = 0; i < loops; ++i) {
    for (int j = 0; j < num_of_pictures; ++j) {

        for (int k = 0; k < num_of_tests; ++k) {
            index = i * num_of_tests * num_of_pictures + j * num_of_tests + k;

            if (j==0)
                file4k << times[index] << ";";
            else if (j==1)
                file1080 << times[index] << ";";
            else if (j==2)
                file_small << times[index] << ";";
        }

        if (j==0)
            file4k << "\n";
        else if (j==1)
            file1080 << "\n";
        else if (j==2)
            file_small << "\n";
    }
}
#endif

    file4k.close();
    file1080.close();
    file_small.close();

    cout << "Complete" << endl;
    return 0;
}

std::string DescribeIosFailure(const std::ios& stream) {
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
