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
            system("pause");\
            return 1; \
        } \
    } while (0)


// Dimensioni array multidimensionale (anche define andava bene)
const int righe = 480;
const int colonne = 640;

#define BLKSIZE 32


// funzione eseguita sulla GPU (calcolo parallelo)
__global__ void matrix_mul(int *a, int *b, int *c){

	// Compute each thread's global row and column index
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < righe && col < colonne) 
			c[row * colonne + col] = (a[row * colonne + col] * b[col * righe + row])/255;

	
}



int main() {

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

	// allocazione  matrice che si andrà a moltiplicare a quelle presenti nelle memorie
	int* matRandHost;
	matRandHost = (int *)malloc(righe*colonne * sizeof(int));

	// generazione valori pseudorandomici e popolamento matrice
	for (int i = 0; i < righe; i++)
		for (int j = 0; j < colonne; j++)
			matRandHost[i*colonne + j] = rand() % 256;

	// allocazione matrice di host
	int* matriceHost;
	matriceHost = (int *)malloc(righe*colonne * sizeof(int));

	// popolamento matrice con i valori dell' immagine
	for (int i = 0; i < righe; i++)
		for (int j = 0; j < colonne; j++)
			matriceHost[i*colonne + j] = (int)img.at<uchar>(i, j);

	int *matriceGPU, *matRGPU, *matResGPU;

	int *matResHost;
	matResHost = (int *)malloc(righe*colonne * sizeof(int));

	cudaMalloc((int **)&matriceGPU, (righe*colonne) * sizeof(int));
	cudaMalloc((int **)&matRGPU, (righe*colonne) * sizeof(int));
	cudaMalloc((int **)&matResGPU, (righe*colonne) * sizeof(int));

	cudaMemcpy(matriceGPU, matriceHost, (righe*colonne) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(matRGPU, matRandHost, (righe*colonne) * sizeof(int), cudaMemcpyHostToDevice);

	dim3 block(BLKSIZE, BLKSIZE);
	dim3 grid(colonne / BLKSIZE, righe / BLKSIZE);


	matrix_mul << <grid, block >> > (matriceGPU, matRGPU, matResGPU);
	cudaCheckErrors("kernel fail");
	cudaMemcpy(matResHost, matResGPU, (righe*colonne) * sizeof(int), cudaMemcpyDeviceToHost);
	cudaCheckErrors("fail to copy the data back");


	int **image = new int *[righe];
	for (int i = 0; i < righe; i++)
		image[i] = new int[colonne];
	
	for (int i = 0; i < righe; i++)
		for (int j = 0; j < colonne; j++) {
			if (matRandHost[i*colonne + j] != 0)
				image[i][j] = (matResHost[i*colonne + j]*255) / matRandHost[i*colonne + j];
			else
				image[i][j] = 0;

		}
	
		Mat imgRes = Mat(righe, colonne, CV_8UC1, image);
	    namedWindow("Immagine Originale (Scala di grigi)");
		imshow("Immagine Originale (Scala di grigi)", img); 
		namedWindow("Immagine risultante");
		imshow("Immagine risultante", imgRes);
		waitKey();

		free(matRandHost);
		free(matriceHost);
		free(image);

		cudaFree(matriceGPU);
		cudaFree(matRGPU);
		cudaFree(matResGPU);

		destroyAllWindows();
	return 0;
}