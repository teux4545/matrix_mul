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


// Dimensioni array multidimensionale (matrici)
// NB il numero di colonne della prima matrice deve coincidere assolutamente con il numero di righe della seconda matrice
// oppure, viceversa, le righe della prima dovranno coincidere con le colonne della seconda

//prima matrice (M1)
const int righeM1 = 960;
const int colonneM1 = 1280;

//seconda matrice (M2)
const int righeM2 = 1280;
const int colonneM2 = 960;

// la matrice risultante dal prodotto avrà dimesioni (colonneM1 * righeM2) o (righeM1 * colonneM2) 
// a seconda se facciamo rispettivamente M2*M1 oppure M1*M2

#define BLKSIZE 32


// funzione eseguita sulla GPU (calcolo parallelo)
__global__ void matrix_mulGPU(int *a, int *b, int *c){

	// Compute each thread's global row and column index
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	int sum = 0;
	//col<righe si riferisce alle colonne della seconda matrice che infatti nel nostro caso sono le righe
	if (row < righeM1 && col < colonneM2) {
		for (int i = 0; i < colonneM1; i++) {

			sum += (a[row * colonneM1 + i] * b[i * colonneM2 + col])/255;
		}
		c[row * colonneM2 + col] = sum;
	}
}

void matrix_mulCPU(int* a, int* b, int* c)
{
	int sum = 0;
	//righe della prima matrice
	for (int i = 0; i < righeM1; i++)
	{
		//colonne della seconda matrice che corrispondono infatti alle righe della prima matrice
		//(definizione di prodotto tra matrici)
		for (int j = 0; j < colonneM2; j++)
		{
			sum = 0;
			//colonne della prima matrice
			for (int k = 0; k < colonneM1; k++)
			{
				//colonne prima matrice -> colonne , colonne seconda matrice - > righe
				sum += (a[i * colonneM1 + k] * b[k * colonneM2 + j])/255;
			}
			c[i * colonneM2 + j] = sum;
		}
	}
}


int main() {

	//apro l'immagine e la carico nella variabile img
	// la conversione in scala di grigi serve per semplificare e avere matrici in due dimensioni invece  di tre
	puts("Acquisizione dati immagine");
	Mat imgOriginal = imread("image.jpg", IMREAD_GRAYSCALE);
	
	// controllo che siano presenti i dati dell'immagine
	if (imgOriginal.data == NULL) {
		cerr << "Errore nell'aprire l'immagine" << endl;
		return(-1);
	}

	// adattamento dell' immagine alle dimensioni della matrice

	Mat imgResized;
	double scale_x = colonneM1 / (int)imgOriginal.cols;
	double scale_y = righeM1 / (int)imgOriginal.rows;
	resize(imgOriginal, imgResized, Size(), scale_x, scale_y, INTER_LINEAR);
	Mat img = imgResized;

	puts("Acquisizione completata");

	// allocazione  matrice che si andrà a moltiplicare a quelle presenti nelle memorie
	int* matRandHost;
	matRandHost = (int *)malloc(colonneM2*righeM2 * sizeof(int));

	// generazione valori pseudorandomici e popolamento matrice
	for (int i = 0; i < righeM2; i++)
		for (int j = 0; j < colonneM2; j++)
			matRandHost[i*colonneM2 + j] = rand() % 256;

	// allocazione matrice di host
	int* matriceHost;
	matriceHost = (int *)malloc(righeM1*colonneM1 * sizeof(int));

	// popolamento matrice con i valori dell' immagine
	for (int i = 0; i < righeM1; i++)
		for (int j = 0; j < colonneM1; j++)
			matriceHost[i*colonneM1 + j] = (int)img.at<uchar>(i, j);

	int *matriceGPU, *matRGPU, *matResGPU;

	int *matResHost, *matResCPU;
	matResHost = (int *)malloc(righeM1*colonneM2 * sizeof(int));
	matResCPU = (int *)malloc(righeM1*colonneM2 * sizeof(int));

	cudaMalloc((int **)&matriceGPU, (righeM1*colonneM1) * sizeof(int));
	cudaMalloc((int **)&matRGPU, (righeM2*colonneM2) * sizeof(int));
	cudaMalloc((int **)&matResGPU, (righeM1*colonneM2) * sizeof(int));

	cudaMemcpy(matriceGPU, matriceHost, (righeM1*colonneM1) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(matRGPU, matRandHost, (righeM2*colonneM2) * sizeof(int), cudaMemcpyHostToDevice);

	dim3 block(BLKSIZE, BLKSIZE);
	dim3 grid(colonneM2/ BLKSIZE, righeM1/ BLKSIZE);

	matrix_mulGPU << <grid, block >> > (matriceGPU, matRGPU, matResGPU);
	cudaCheckErrors("kernel fail");

	matrix_mulCPU(matriceHost, matRandHost, matResCPU);
	
	cudaMemcpy(matResHost, matResGPU, (righeM1*colonneM2) * sizeof(int), cudaMemcpyDeviceToHost);
	cudaCheckErrors("fail to copy the data back");
	


		//Mat imgRes = Mat(righe, colonne, CV_8UC1, image);
	    namedWindow("Immagine Originale (Scala di grigi)");
		imshow("Immagine Originale (Scala di grigi)", img); 
		//namedWindow("Immagine risultante");
		//imshow("Immagine risultante", imgRes);
		waitKey();

		free(matRandHost);
		free(matriceHost);
		free(matResHost);
		//free(image);

		cudaFree(matriceGPU);
		cudaFree(matRGPU);
		cudaFree(matResGPU);

		destroyAllWindows();
	return 0;
}