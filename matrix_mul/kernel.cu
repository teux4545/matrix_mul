// librerie CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>

//librerie standard
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

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
            fprintf(stderr, "\n*** FAILED - ABORTING\n"); \
            system("pause");\
            return 1; \
        } \
    } while (0)

cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
	if (result != cudaSuccess) {
		fprintf(stderr, "CUDA Runtime Error: %s\n",
			cudaGetErrorString(result));
		assert(result == cudaSuccess);
	}
#endif
	return result;
}


// Dimensioni array multidimensionale (matrici)
// NB il numero di colonne della prima matrice deve coincidere assolutamente con il numero di righe della seconda matrice
// oppure, viceversa, le righe della prima dovranno coincidere con le colonne della seconda

//prima matrice (M1)
const int righeM1 = 1500;
const int colonneM1 = 1500;

//seconda matrice (M2)
const int righeM2 = 1500;
const int colonneM2 = 1500;

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
	int somma = 0;
	//righe della prima matrice
	for (int i = 0; i < righeM1; i++)
	{
		//colonne della seconda matrice che corrispondono infatti alle righe della prima matrice
		//(definizione di prodotto tra matrici)
		for (int j = 0; j < colonneM2; j++)
		{
			somma = 0;
			//colonne della prima matrice
			for (int k = 0; k < colonneM1; k++)
			{
				//colonne prima matrice -> colonne , colonne seconda matrice - > righe
				somma += (a[i * colonneM1 + k] * b[k * colonneM2 + j])/255;
			}
			c[i * colonneM2 + j] = somma;
			
		}
	}
}


int main() {

	/*       VARIANTE CON IMMAGINI
	
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
	cout << endl;
	*/
	puts("Allocazione delle variabili Host nella memoria");
	// allocazione  matrice che si andrà a moltiplicare a quelle presenti nelle memorie

	int* matRandHost;
	matRandHost = (int *)malloc((colonneM2*righeM2) * sizeof(int));

	// generazione valori pseudorandomici e popolamento matrice
	puts("Generazione valori randomici per la prima e la seconda matrice");

	for (int i = 0; i < righeM2; i++)
		for (int j = 0; j < colonneM2; j++)
			matRandHost[i*colonneM2 + j] = rand() % 256;

	// allocazione matrice di host
	int* matriceHost;
	matriceHost = (int *)malloc(righeM1*colonneM1 * sizeof(int));

	// popolamento matrice con i valori dell' immagine
	for (int i = 0; i < righeM1; i++)
		for (int j = 0; j < colonneM1; j++)
			matriceHost[i*colonneM1 + j] = rand() % 256;
			//matriceHost[i*colonneM1 + j] = (int)img.at<uchar>(i, j);

	int *matResHost, *matResCPU;
	matResHost = (int *)malloc(righeM1*colonneM2 * sizeof(int));
	matResCPU = (int *)malloc(righeM1*colonneM2 * sizeof(int));

	puts("Allocazione popolamento matrici completati");
	cout << endl;

	// Allocazione di memoria per le variabili che lavoreranno sulla GPU
	puts("Allocazione variabili nella memoria della GPU");

	int *matriceGPU, *matRGPU, *matResGPU;

	cudaMalloc((int **)&matriceGPU, (righeM1*colonneM1) * sizeof(int));
	cudaCheckErrors("Allocazione fallita");
	cudaMalloc((int **)&matRGPU, (righeM2*colonneM2) * sizeof(int));
	cudaCheckErrors("Allocazione fallita");
	cudaMalloc((int **)&matResGPU, (righeM1*colonneM2) * sizeof(int));
	cudaCheckErrors("Allocazione fallita");

	puts("Allocazione completata");
	cout << endl ;

	//Copia dei valori della prima e seconda matrice (host) nelle variabili device
	puts("Trasferimento valori delle due matrici nella GPU");

	cudaMemcpy(matriceGPU, matriceHost, (righeM1*colonneM1) * sizeof(int), cudaMemcpyHostToDevice);
	cudaCheckErrors("Copia dei dati da Host a Device fallita");
	cudaMemcpy(matRGPU, matRandHost, (righeM2*colonneM2) * sizeof(int), cudaMemcpyHostToDevice);
	cudaCheckErrors("Copia dei dati da Host a Device fallita");
	cudaMemcpy(matResGPU, matResHost, (righeM1*colonneM2) * sizeof(int), cudaMemcpyHostToDevice);
	cudaCheckErrors("Copia dei dati da Host a Device fallita");

	puts("Trasferimento completato");
	cout << endl;

	// Dimensionamento della griglia di blocchi e thread (max 1024 thread per blocco)
	puts("Costruzione griglia di calcolo per la GPU");

	dim3 threads(BLKSIZE, BLKSIZE);
	dim3 blocks((colonneM2 + BLKSIZE - 1)/ BLKSIZE, (righeM1 * BLKSIZE - 1 )/ BLKSIZE);

	cout << endl;

	// Esecuzione funzione sulla GPU
	puts("Avvio calcolo sulla GPU");

	matrix_mulGPU << <blocks, threads >> > (matriceGPU, matRGPU, matResGPU);
	cudaCheckErrors("Esecuzione del kernel Fallita");
	cudaDeviceSynchronize();
	puts("Calcolo sulla GPU eseguito");
	cout << endl;

	//Trasferimento dei valori della matrice risultante dalla compilazione sulla GPU alla variabile Host
	puts("Trasferimento valori della GPU alla matrice del Host del risultato");
	cudaMemcpyAsync(matResHost, matResGPU, (righeM1*colonneM2) * sizeof(int), cudaMemcpyDeviceToHost);
	cudaCheckErrors("Copia dei dati fallita\n");
	puts("Trasferimento completato");
	cout << endl;

	// Esecuzione funzione sulla CPU
	puts("Avvio calcolo sulla CPU");
	matrix_mulCPU(matriceHost, matRandHost, matResCPU);
	puts("Calcolo sulla CPU eseguito");
	cout << endl << endl;

		free(matRandHost);
		free(matriceHost);
		free(matResHost);

		cudaFree(matriceGPU);
		cudaFree(matRGPU);
		cudaFree(matResGPU);

		system("pause");
	return 0;
}