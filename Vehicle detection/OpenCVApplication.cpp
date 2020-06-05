// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"

using namespace std;

void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", CV_LOAD_IMAGE_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, CV_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		
		Mat src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height,width,CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				uchar val = src.at<uchar>(i,j);
				uchar neg = 255 - val;
				dst.at<uchar>(i,j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the “diblook style”
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		int w = (int) src.step; // no dword alignment is done !!!
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height,width,CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("input image",src);
		imshow("gray image",dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// Componentele d eculoare ale modelului HSV
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, CV_BGR2HSV);

		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				int hi = i*width * 3 + j * 3;
				int gi = i*width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;		// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src,dst,gauss;
		src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int) k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);
		imshow("input image",src);
		imshow("canny",dst);
		waitKey();
	}
}

void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}
		
	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		Canny(grayFrame,edges,40,100,3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = cvWaitKey(0);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n"); 
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];
	
	// video resolution
	Size capS = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;
		
		imshow(WIN_SRC, frame);

		c = cvWaitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115){ //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess) 
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == CV_EVENT_LBUTTONDOWN)
	{
		printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
			x, y,
			(int)(*src).at<Vec3b>(y, x)[2],
			(int)(*src).at<Vec3b>(y, x)[1],
			(int)(*src).at<Vec3b>(y, x)[0]);
	}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

/* Histogram display function - display a histogram using bars (simlilar to L3 / PI)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i<hist_cols; i++)
	if (hist[i] > max_hist)
		max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}

Mat convolution(Mat src, Mat core, float c) {
	Mat newImg(src.rows, src.cols, CV_32FC1);

	for (int i = 2; i < src.rows - 2; i++) {
		for (int j = 2; j < src.cols - 2; j++) {
			float sum = 0;
			for (int u = 0; u < 5; u++) {
				for (int v = 0; v < 5; v++) {
					sum += core.at<float>(u, v) * src.at<uchar>(i + u - 2, j + v - 2);
				}
			}


			sum /= c;
			if (sum > 255) {
				sum = 255;
			}
			else if (sum < 0) {
				sum = 0;
			}

			newImg.at<float>(i, j) = sum;
		}
	}

	return newImg;
}

Mat gaussian(Mat src) {
	float phi = 0.5;

	Mat core(5, 5, CV_32FC1);

	float c = 0;

	for (int k = 0; k < 5; k++) {
		for (int l = 0; l < 5; l++) {
			float exponent1 = (k - 2) * (k - 2);
			float exponent2 = exponent1 + (l - 2) * (l - 2);
			float exponent = -exponent2 / (2 * phi * phi);
			float val = (1 / (2 * PI * phi * phi)) * exp(exponent);
			core.at<float>(k, l) = val;
			c += val;
		}
	}

	return convolution(src, core, c);
}


int zone(float x) {
	if ((x >= 3 * PI / 8 && x < 5 * PI / 8) || (x >= 11 * PI / 8 && x < 13 * PI / 8)) {
		return 0;
	}
	if ((x >= PI / 8 && x < 3 * PI / 8) || (x >= 9 * PI / 8 && x < 11 * PI / 8)) {
		return 1;
	}
	if (x >= 0 && x < PI / 8 || x >= 15 * PI / 8 || (x >= 7 * PI / 8 && x < 9 * PI / 8)) {
		return 2;
	}
	if ((x >= 5 * PI / 8 && x < 7 * PI / 8) || (x >= 13 * PI / 8 && x < 15 * PI / 8)) {
		return 3;
	}
}

int createCopy(Mat gradOrientation, Mat gradOrientationCopy, Mat gradiendModule) {
	int zeros = 0;
	for (int i = 0; i < gradOrientation.rows; i++) {
		for (int j = 0; j < gradOrientation.cols; j++) {
			switch (zone(gradOrientation.at<float>(i, j)))
			{
			case 0:
				if (i > 0 && gradiendModule.at<float>(i, j) < gradiendModule.at<float>(i - 1, j)) {
					gradOrientationCopy.at<float>(i, j) = 0;
					zeros++;
				}
				else if (i < gradiendModule.rows - 1 && gradiendModule.at<float>(i, j) < gradiendModule.at<float>(i + 1, j)) {
					gradOrientationCopy.at<float>(i, j) = 0;
					zeros++;
				}
				break;
			case 1:
				if (i > 0 && j < gradiendModule.cols - 1 && gradiendModule.at<float>(i, j) < gradiendModule.at<float>(i - 1, j + 1)) {
					gradOrientationCopy.at<float>(i, j) = 0;
					zeros++;
				}
				else if (i < gradiendModule.rows - 1 && j > 0 && gradiendModule.at<float>(i, j) < gradiendModule.at<float>(i + 1, j - 1)) {
					gradOrientationCopy.at<float>(i, j) = 0;
					zeros++;
				}
				break;
			case 2:
				if (j > 0 && gradiendModule.at<float>(i, j) < gradiendModule.at<float>(i, j - 1)) {
					gradOrientationCopy.at<float>(i, j) = 0;
					zeros++;
				}
				else if (j < gradiendModule.cols - 1 && gradiendModule.at<float>(i, j) < gradiendModule.at<float>(i, j + 1)) {
					gradOrientationCopy.at<float>(i, j) = 0;
					zeros++;
				}
				break;
			case 3:
				if (i > 0 && j > 0 && gradiendModule.at<float>(i, j) < gradiendModule.at<float>(i - 1, j - 1)) {
					gradOrientationCopy.at<float>(i, j) = 0;
					zeros++;
				}
				else if (i < gradiendModule.rows - 1 && j < gradiendModule.cols - 1 && gradiendModule.at<float>(i, j) < gradiendModule.at<float>(i + 1, j + 1)) {
					gradOrientationCopy.at<float>(i, j) = 0;
					zeros++;
				}
				break;
			default:
				break;
			}
		}
	}
	return zeros;
}

void cannyEdgeDetection(Mat src) {

	Mat res = gaussian(src);
	short matx[3][3] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
	short maty[3][3] = { 1, 2, 1, 0, 0, 0, -1, -2, -1 };

	Mat fx(src.rows, src.cols, CV_16SC1);
	Mat fy(src.rows, src.cols, CV_16SC1);

	for (int i = 1; i < src.rows - 1; i++) {
		for (int j = 1; j < src.cols - 1; j++) {
			short sum = 0;
			short sum2 = 0;
			for (int u = 0; u < 3; u++) {
				for (int v = 0; v < 3; v++) {
					sum += matx[u][v] * src.at<uchar>(i + u - 1, j + v - 1);
				}
			}
			for (int u = 0; u < 3; u++) {
				for (int v = 0; v < 3; v++) {
					sum2 += maty[u][v] * src.at<uchar>(i + u - 1, j + v - 1);
				}
			}

			fx.at<short>(i, j) = sum;
			fy.at<short>(i, j) = sum2;
		}
	}

	Mat gradiendModule(src.rows, src.cols, CV_32FC1);
	Mat gradOrientation(src.rows, src.cols, CV_32FC1);
	Mat gradOrientationCopy(src.rows, src.cols, CV_32FC1);
	for (int i = 0; i < gradiendModule.rows; i++) {
		for (int j = 0; j < gradiendModule.cols; j++) {
			float val = sqrt(1.0 * fx.at<short>(i, j) * fx.at<short>(i, j) +
				1.0 * fy.at<short>(i, j) * fy.at<short>(i, j));
			val /= 4.0 * sqrt(2);
			gradiendModule.at<float>(i, j) = val;
			gradOrientationCopy.at<float>(i, j) = val;
		}
	}

	for (int i = 0; i < gradiendModule.rows; i++) {
		for (int j = 0; j < gradiendModule.cols; j++) {
			float val = atan2(1.0 * fy.at<short>(i, j), 1.0 * fx.at<short>(i, j));
			gradOrientation.at<float>(i, j) = val + PI;

		}
	}

	int zeros = createCopy(gradOrientation, gradOrientationCopy, gradiendModule);
	float p = 0.1;

	int nrEdgePixels = p * (src.cols * src.rows - zeros);


	int hist[256] = { 0 };

	for (int i = 1; i < gradOrientationCopy.rows - 1; i++) {
		for (int j = 1; j < gradOrientationCopy.cols - 1; j++) {
			int val = (int)round(gradOrientationCopy.at<float>(i, j));
			hist[val]++;
		}
	}

	for (int i = 0; i < gradOrientationCopy.rows; i++) {
		int val = (int)round(gradOrientationCopy.at<float>(i, 0));
		if (val == 0) {
			hist[0]++;
		}
	}
	for (int i = 0; i < gradOrientationCopy.cols; i++) {
		int val = (int)round(gradOrientationCopy.at<float>(0, i));
		if (val == 0) {
			hist[0]++;
		}
	}

	int nrNonEdgePixels = (1 - p) * ((src.cols - 2) * (src.rows - 2) - hist[0]);

	int sum = 0;
	int thresholdHigh = 0;
	for (int i = 1; i < 256; i++) {
		sum += hist[i];
		if (sum >= nrNonEdgePixels) {
			thresholdHigh = i;
			break;
		}
	}
	int thresholdLow = (2 * thresholdHigh) / 5;

	for (int i = 0; i < gradOrientationCopy.rows; i++) {
		for (int j = 0; j < gradOrientationCopy.cols; j++) {
			int col = round(gradOrientationCopy.at<float>(i, j));
			if (col <= thresholdLow) {
				src.at<uchar>(i, j) = 0;
			}
			else if (col > thresholdLow && col < thresholdHigh) {
				src.at<uchar>(i, j) = 128;
			}
			else {
				src.at<uchar>(i, j) = 255;
			}
		}
	}
}

typedef struct {
	int iStart;
	int jStart;
	int iEnd;
	int jEnd;
}ROI;

int symmetryFunction(Mat src, int m, int n, int x) {
	int k = 0;

	int f = 0;

	for (int i = n - x; i < n; i++) {
		if (i >= 1 && i + x + 1 < src.cols && n + x - k < src.cols) {
			int val1 = int(src.at<uchar>(m, i));
			int val2 = int(src.at<uchar>(m, n + x - k));
			f += abs(val1 - val2);
		}
		k++;
	}
	return f;
}

int symmetry(int istart, int jstart, int iend, int jend, Mat img) {
	
	uchar col = img.at<uchar>(img.rows - 1, img.cols / 2);

	int interval = img.cols / 5;
	int st = img.cols / 2 - interval;
	int end = img.cols / 2 + interval;

	for (int i = istart; i < iend; i++) {
		int min = INT_MAX;
		int minPoz = jstart;
		for (int j = jstart; j < jend; j++) {
			int val = symmetryFunction(img, i, j, (jend - jstart) / 2);
			if (val > 0 && val < min) {
				min = val;
				minPoz = j;
			}
		}
		img.at<uchar>(i, minPoz) = 255;
	}

	int nrSym = 0;
	for (int i = istart; i < iend; i++) {
		for (int j = jstart; j < jend; j++) {
			if (img.at<uchar>(i, j) == 255 && j > st && j < end) {
				nrSym++;
				break;
			}
		}
	}

	return nrSym;
}

Mat resizeImage(Mat src) {
	Mat dst;

	resize(src, dst, Size(), 64.0 / src.cols, 64.0 / src.rows);
	return dst;
}

Mat createImage(int iStart, int iEnd, int jStart, int jEnd, Mat src) {
	Mat dest(iEnd - iStart, jEnd - jStart, CV_8UC1);
	int m = 0;
	int n;
	for (int i = iStart; i < iEnd; i++) {
		n = 0;
		for (int j = jStart; j < jEnd; j++) {
			dest.at<uchar>(m, n) = src.at<uchar>(i, j);
			n++;
		}
		m++;
	}

	return resizeImage(dest);
}

void createCsvForImage(vector<Mat> images) {
	ofstream f("C:/Users/Ionut/Desktop/PI/PIproject/neural_network/vehiclesToDetect.csv");
	for (int k = 0; k < 4095; k++) {
		char pixel[10];
		sprintf(pixel, "pixel%d", k);

		f << pixel << ",";
	}
	f << "pixel4095" << endl;
	

	for (Mat im : images) {
		for (int i = 0; i < im.rows; i++) {
			for (int j = 0; j < im.cols; j++) {
				if (im.at<uchar>(i, j) != 0) {
					f << "1";
				}
				else {
					f << "0";
				}
				if (i < im.rows - 1 || j < im.cols - 1) {
					f << ",";
				}
			}	
		}
		f << endl;
	}
	
	f.close();
}

void markVehicle(Mat src, vector<ROI> rois) {
	for (ROI r : rois) {
		for (int i = r.iStart; i < r.iEnd; i++) {
			src.at<Vec3b>(i, r.jStart) = Vec3b(0, 0, 255);
			src.at<Vec3b>(i, r.jEnd) = Vec3b(0, 0, 255);
		}
		for (int i = r.jStart; i < r.jEnd; i++) {
			src.at<Vec3b>(r.iStart, i) = Vec3b(0, 0, 255);
			src.at<Vec3b>(r.iEnd, i) = Vec3b(0, 0, 255);
		}
	}
}

Mat globalImg;
vector<ROI> rois;

void readDetected() {
	fstream f("C:/Users/Ionut/Desktop/PI/PIproject/neural_network/vehiclesDetected.csv");
	int k = 0;
	vector<ROI> goodRois;

	string line;
	getline(f, line);

	for (int i = 0; i < line.size(); i++) {
		if (line[i] != ',') {
			if (line[i] == '1') {
				goodRois.push_back(rois[k]);
			}
			k++;
		}
	}
	
	markVehicle(globalImg, goodRois);

	imshow("result", globalImg);
	waitKey();
	f.close();
}

void detectVehicles() {
	char fname[MAX_PATH];
	openFileDlg(fname);
	Mat src = imread(fname, IMREAD_GRAYSCALE);
	
	Mat clone = src.clone();
	globalImg = imread(fname, IMREAD_COLOR);
	
	uchar tresh = 50;
	uchar treshv = 2;
	vector<Mat> images;

	Mat shadow(src.rows, src.cols, CV_8UC1);

	for (int i = 0; i < src.rows - 1; i++) {
		for (int j = 0; j < src.cols; j++) {
			shadow.at<uchar>(i, j) = src.at<uchar>(i, j);
		}
	}

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (i < src.rows - 1 && shadow.at<uchar>(i, j) <= tresh && shadow.at<uchar>(i + 1, j) - shadow.at<uchar>(i, j) >= treshv) {
				shadow.at<uchar>(i, j) = 255;
			}	
			else{
				shadow.at<uchar>(i, j) = 0;
			}
		}
	}

	imshow("shadow", shadow);
	

	int length = src.cols / 15;
	int length_max = 30;

	int i = 2;
	int j;

	vector<int> is;
	vector<int> js;

	int nrImages = 0;

	while (i < src.rows - 2) {
		int j = 0;
		int counter = 0;
		Mat cloneForSymmetry = clone.clone();
		while (j < src.cols) {
			if (shadow.at<uchar>(i, j) == 255 || shadow.at<uchar>(i + 1, j) == 255 || shadow.at<uchar>(i - 1, j) == 255 ||
				shadow.at<uchar>(i + 2, j) == 255 || shadow.at<uchar>(i - 2, j) == 255) {
				counter++;
			}
			else {
				if (counter >= length) {
					int h[256] = { 0 };

					int sides = counter / 4;

					if (i - counter >= 0 && j - counter >= 0) {


						int nrLines = 0;
						ROI roi;
						// If I can, I take a bigger ROI
						roi.iStart = (i - counter - sides >= 0) ? (i - counter - sides) : (i - counter);
						roi.jStart = (j - counter - sides >= 0) ? (j - counter - sides) : (j - counter);
						roi.iEnd = (i + sides < src.rows) ? (i + sides) : i;
						roi.jEnd = (j + sides < src.cols) ? (j + sides) : j;
						
						
						
						int nr = symmetry(roi.iStart, roi.jStart, roi.iEnd, roi.jEnd, cloneForSymmetry);
						
						int len = roi.iEnd - roi.iStart;
						
						
						if (nr >= len / 2) {
							for (int k = roi.iStart; k < i; k++) {
								src.at<uchar>(k, roi.jEnd) = 0;
								src.at<uchar>(k, roi.jStart) = 0;
							}
							for (int k = roi.jStart; k < roi.jEnd; k++) {
								src.at<uchar>(roi.iStart, k) = 0;
								src.at<uchar>(i, k) = 0;
							}


							char name[10];

							Mat aux = createImage(roi.iStart, roi.iEnd, roi.jStart, roi.jEnd, clone);

							sprintf(name, "image%d", nrImages);
							cannyEdgeDetection(aux);

							images.push_back(aux);
							rois.push_back(roi);

							sprintf(name, "image%d", nrImages * 2);
							nrImages++;
							j += counter;
						}
						else {
							for (int k = roi.iStart; k < i; k++) {
								src.at<uchar>(k, roi.jEnd) = 128;
								src.at<uchar>(k, roi.jStart) = 128;
							}
							for (int k = roi.jStart; k < roi.jEnd; k++) {
								src.at<uchar>(roi.iStart, k) = 128;
								src.at<uchar>(i, k) = 128;
							}
						}
					}
				}
				counter = 0;
			}
			j++;
		}
		i++;
	}

	createCsvForImage(images);

	imshow("img", src);
	waitKey();

	readDetected();
}


void allVehicles() {
	for (int i = 0; i < 126; i++) {
		cout << i << endl;
		char s[MAX_PATH];
		sprintf(s, "newImages/img%d.bmp", 975 + 500 + 975 + 975 + 975 + 975 + 975 + 975 + 53 + 126 + 65 + 101 + 47 + 54 + 5 + 46 + 526 + i);

		char nr[MAX_PATH];
		if (i + 1 < 10) {
			sprintf(nr, "Images/OwnCollection/OwnCollection/new3/image_000%d.jpg", i + 1);
		}
		else if (i + 1 < 100) {
			sprintf(nr, "Images/OwnCollection/OwnCollection/new3/image_00%d.jpg", i + 1);
		}
		else {
			sprintf(nr, "Images/OwnCollection/OwnCollection/new3/image_0%d.jpg", i + 1);
		}
		Mat src = imread(nr, IMREAD_GRAYSCALE);

		src = resizeImage(src);
		cannyEdgeDetection(src);
		imwrite(s, src);
	}
}

void createCsv() {
	ofstream f("vehicles.csv");

	for (int k = 0; k < 4096; k++) {
		char pixel[10];
		sprintf(pixel, "pixel%d,", k);
		f << pixel;
	}

	f << "vehicle" << endl;

	for (int k = 0; k < 8472; k++) {
		char nr[MAX_PATH];
		sprintf(nr, "training_data/img%d.bmp", k);

		Mat src = imread(nr, IMREAD_GRAYSCALE);

		if (src.rows > 0 && src.cols > 0) {

			for (int i = 0; i < src.rows; i++) {
				for (int j = 0; j < src.cols; j++) {
					if (src.at<uchar>(i, j) != 0) {
						f << "1" << ",";
					}
					else {
						f << "0" << ",";
					}
				}
			}
			if (k < 3425 || k > 7325) {
				f << "1";
			}
			else {
				f << "0";
			}
			f << endl;
		}
	}
	f.close();
}


int main()
{
	detectVehicles();
	return 0;
}