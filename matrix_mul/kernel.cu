
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>


using namespace std;
using namespace cv;

__global__ void matrix_mul(){
   ciccio
}


int main(){

	Mat img = imread("image.jpg"); //apro l'immagine e la carico nella variabile img
	if (img.data == NULL)          // controllo che siano presenti i dati dell'immagine
	{
		cerr << "Errore nell'aprire l'immagine" << endl;
		return(-1);
	}
	
	namedWindow("image");
	imshow("image", img);
	waitKey();
	/*
	Mat immagineOriginale = imread("image.jpg");
	if (immagineOriginale.data == NULL)
	{
		cerr << "Error open image" << endl;
		return(-1);
	}
	
		namedWindow("Immagine Originale");
		imshow("Immagine Originale", immagineOriginale);

		cout << "Numero di righe dell'immagine: " << immagineOriginale.rows << endl;
		cout << "Numero di colonne dell'immagine: " << immagineOriginale.cols << endl;

		int z = 0;
			for (int i = 0; i < immagineOriginale.rows; i++)
			{
				for (int j = 0; j < immagineOriginale.cols; j++)
				{
					int b = immagineOriginale.at<Vec3b>(i, j)[0];
					int g = immagineOriginale.at<Vec3b>(i, j)[1];
					int r = immagineOriginale.at<Vec3b>(i, j)[2];
					cout << "Pixel numero: " << z++ << " -> " << r << " " << g << " " << b << endl;
				}
			}

		waitKey(0);*/
	return 0;
}