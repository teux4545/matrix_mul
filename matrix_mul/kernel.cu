// librerie CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//librerie standard
#include <iostream>
#include <stdio.h>

//librerie OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;


// Controllo errori cuda
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            return 1; \
        } \
    } while (0)


// Dimensioni array multidimensionale (anche define andava bene)
const int colonne = 640;
const int righe = 480;

//Dimensioni del blocco su cui andrà a lavorare ogni thread
//#define BLKXSIZE 32
//#define BLKYSIZE 4
//#define BLKZSIZE 4

// funzione eseguita sulla GPU (calcolo parallelo)
__global__ void matrix_mul(int*** matrice, int*** mat, int*** mresult){

	//unsigned idrow = blockIdx.y*blockDim.y + threadIdx.y;
	//unsigned idcol = blockIdx.x*blockDim.x + threadIdx.x;
	
}



int main(){
	
	//apro l'immagine e la carico nella variabile img
	// la conversione in scala di grigi serve per semplificare e avere matrici in due dimensioni invece  di tre
	Mat imgOriginal = imread("image.jpg", IMREAD_GRAYSCALE); 
	
	// controllo che siano presenti i dati dell'immagine
	if (imgOriginal.data == NULL) {
		cerr << "Errore nell'aprire l'immagine" << endl;
		return(-1);
	}

	// adattamento dell' immagine alle dimensioni della matrice
		Mat imgResized;         
		double scale_x = colonne / (int)imgOriginal.cols;
		double scale_y = righe / (int)imgOriginal.rows;
		resize(imgOriginal, imgResized, Size(), scale_x, scale_y, INTER_LINEAR);

		Mat img = imgResized;

		imshow("grayscale image", img);

		// Wait for a keystroke.   
		waitKey(0);
	return 0;
}