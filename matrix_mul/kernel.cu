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
const int canali = 3;

//Dimensioni del blocco su cui andrà a lavorare ogni thread
#define BLKXSIZE 32
#define BLKYSIZE 4
#define BLKZSIZE 4

// funzione eseguita sulla GPU (calcolo parallelo)
__global__ void matrix_mul(int matrice[][colonne][righe]){

	unsigned idrow = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned idcol = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned idchan = blockIdx.z*blockDim.z + threadIdx.z;

}


int main(){
	
	//apro l'immagine e la carico nella variabile img
	Mat img = imread("image.jpg"); 
	
	// controllo che siano presenti i dati dell'immagine
	if (img.data == NULL) {
		cerr << "Errore nell'aprire l'immagine" << endl;
		return(-1);
	}

	// adattamento dell' immagine alle dimensioni della matrice
		Mat imgResized;         
		double scale_x = colonne / (int)img.cols;
		double scale_y = righe / (int)img.rows;
		resize(img, imgResized, Size(), scale_x, scale_y, INTER_LINEAR);

	/* posso creare un array tridimensionale per la CPU e per la GPU(matrice x*y*z)
	 variante per allocare memoria --> int*** matrix3dCPU = new int**[righe];
		dove dovrò iterare per creare memoria sia per righe che per colonne
		
		il codice sarà il seguente: 
		/*  alloco dinamicamente memoria per ogni elemento 
		     che corrisponde al valore di un canale in ogni punto x,y della matrice
		for (int i = 0; i < righe; i++)
		{
			matrix3dCPU[i] = new int*[colonne];
			for (int j = 0; j < colonne; j++) {
				matrix3dCPU[i][j] = new int[canali];
			}
		}*/

		// creazione array tridimensionali e relativa allocazione nelle rispettive memorie
		int*** matrix3dCPU, matrix3dGPU;
		if((matrix3dCPU = (int ***)malloc((righe*colonne*canali) * sizeof(int)))==0)
		{
			printf("Allocazione fallita \n"); 
			return -1;
		}
		cudaMalloc((int ***) &matrix3dGPU, (righe*colonne*canali) * sizeof(int));
		cudaCheckErrors("Failed to allocate device buffer");

		/* creazione matrice che si andrà a moltiplicare alle nostre matrici con i dati dell'immagine
           -> allocazione memoria per la nuova matrice con il metodo illustrato precedentemente  */
		int*** mat = new int**[righe];

		for (int i = 0; i < righe; i++)
		{
			mat[i] = new int*[colonne];
			for (int j = 0; j < colonne; j++)
				mat[i][j] = new int[canali];
		}
		// Popolamento con valori casuali nella matrice allocata
		for (int i = 0; i < righe; i++)
			for (int j = 0; j < colonne; j++)
				for (int k = 0; k < canali; k++) {
					mat[i][j][k] = rand() % 256;
				}
		/*  qui definisco le dimensioni della griglia e dei blocchi  
		       (più thread formano un blocco e più blocchi formano una griglia)  */
		const dim3 blockSize(BLKXSIZE, BLKYSIZE, BLKZSIZE);
		const dim3 gridSize(((colonne + BLKXSIZE - 1) / BLKXSIZE), 
			((righe + BLKYSIZE - 1) / BLKYSIZE), 
			((canali + BLKZSIZE - 1) / BLKZSIZE));

		/*  caricamento dei valori dei canali B, G ed R nella matrice che lavorerà nella CPU ...
		       (questa operazione potrebbe essere superflua,
			     potrei lavorare direttamante con l'oggetto 'img' già inizializzato ) */
		for (int i = 0; i < righe; i++)
		{
			for (int j = 0; j < colonne; j++)
			{
				for (int k = 0; k < canali; k++) {
					matrix3dCPU[i][j][k] = imgResized.at<Vec3b>(i, j)[k];
				}
			}
		}

		// ...e questo per la GPU
		cudaMemcpy(matrix3dGPU, matrix3dCPU, bytes, cudaMemcpyHostToDevice);
		
		

		// deallocazioe della memoria
		for (int i = 0; i < righe; i++)
		{
			for (int j = 0; j < colonne; j++) {
				delete[] mat[i][j];
			}
			delete[] mat[i];
		}
		delete[] mat;

		/*
		Allocate 2D array on device

float **device2DArray;

   float *h_temp[5];

// Create 2D Array

  cudaMalloc((void **)&device2DArray, 5*sizeof(float *));

  for(int i=0; i<5; i++)

  {

	cudaMalloc( (void **)&h_temp[i], 3*sizeof(float));

  }

  cudaMemcpy(device2DArray, h_temp, 5*sizeof(float *), cudaMemcpyHostToDevice);

// Do not destroy the contents of h_temp
 so, we dont need to copy the pointers from the device again and again. We will hold a copy of the row pointers in h_temp


//Copy host** array onto device

float **cpuArray = someValid2DHostarray;

for(int i=0; i<5; i++)

{

   cudaMemcpy(h_temp[i], cpuArray[i], 3*sizeof(float), cudaMemcpyHostToDevice);

}
Now, device2DArray pointer in GPU is a true 2D array created on GPU with data…

The device2DArray can be passed to a kernel as “float **device2DArray” and the kernel can access it like “device2DArray[i][j]”.

-----------------------------------------------------------------------------


Once done, to copy out the data back to host:

float **cpuArray = someValid2DHostarray;

for(int i=0; i<5; i++)

{

   cudaMemcpy(cpuArray[i], h_temp[i], 3*sizeof(float), cudaMemcpyDeviceToHost);

}
Now cpuArray[i][j] will have same contents as device2DArray[i][j].
		*/
	
	return 0;
}