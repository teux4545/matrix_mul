//librerie standard
#include <iostream>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// librerie CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>

/*//librerie OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>*/

using namespace std;
//using namespace cv;


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


/* Dimensioni array multidimensionale (matrici)
      N.B.: il numero di colonne della prima matrice deve coincidere assolutamente con il numero di righe della seconda matrice  */

// prima matrice (M1)
const int righeM1 = 1440;
const int colonneM1 = 2560;

// seconda matrice (M2)
const int righeM2 = 2560;
const int colonneM2 = 1440;

// dimensioni delle matrici M1, M2 e matrice dei risultati
size_t dimM1 = (righeM1*colonneM1) * sizeof(int); 
size_t dimM2 = (righeM2*colonneM2) * sizeof(int);
size_t dimRes = (righeM1*colonneM2) * sizeof(int);

/*  La matrice risultante dal prodotto avrà dimesioni righeM1 * colonneM2 (righe della prima e colonne della seonda)

     -> IL PRODOTTO TRA MATRICI NON E' COMMUTATIVO  */

// Dimensioni del blocco (x,y) impostate uguali in modo che formino blocchi quadrati esattamente di 1024 threads (limite imposto dall'hardware)
#define BLKSIZE 32


// Funzione eseguita sulla GPU (calcolo parallelo)
// viene utilizzata la memoria globale (qui le variabili infatti possono essere allocate dinamicamente con le APIs cudamalloc e cudamallocHost)
__global__ void matrix_mulGPU(int *a, int *b, int *c) {

	// Inizializzo le coordinate dei thread all'interno della griglia (col e row identificano un singolo thread specifico)
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	// Somma impostata a zero, aumenterà man mano che, come avviene nel prodotto tra matrici, si sommano gli elementi moltiplicati

	int somma = 0;

	/*  viene posta una condizione di controllo affinché vengano chiamati solo i thread effettivamente predisposti per le operazioni di moltiplicazione
	    infatti la griglia creata è impostata sempre leggermente più grande delle dimensioni del prodotto righe*colonne  */

	if (row < righeM1 && col < colonneM2) {

		/*  l'iterazione parte: per la matrice M1 si percorrono le righe della griglia (thread posti orizzontalmente sull'asse x, cioè le colonneM1)
		    per la matrice M2 invece accade la stessa cosa ma si procede scorrendo lungo i thread posti verticalmente (asse y, cioè le righeM2)
		    row e col mantengono "fissato" il calcolo sulle righe e colonne corrispondenti  */

		for (int i = 0; i < colonneM1; i++) {

			// la somma accumula i prodotti che man mano vanno aggiungendosi, scorrendo infatti lungo le dimensioni x e y

			somma += a[row * colonneM1 + i] * b[i * colonneM2 + col];
			// la durata computazionale del processo è data proprio da quest'ultima stringa che dipende direttamente dalle dimensioni delle matrici in esame: il calcolo effettuato è (M1*M2)

		}
		// alla fine di ogni iterazione vengono popolati ordinatamente in modo crescente gli elementi nell'array del risultato

		c[row * colonneM2 + col] = somma;
	}
}

__global__ void matrix_mulGPUShared(int *a, int *b, int *c) {

	__shared__ int sA[BLKSIZE][BLKSIZE];   // sA usa un blocco da 32*32 = 1024 thread
	__shared__ int sB[BLKSIZE][BLKSIZE];   // sB usa un blocco da 32*32 = 1024 thread

	int row = blockDim.y * blockIdx.y + threadIdx.y;
	int col = blockDim.x * blockIdx.x + threadIdx.x;

	int somma = 0;
	sA[threadIdx.y][threadIdx.x] = 0;
	sB[threadIdx.y][threadIdx.x] = 0;

	for (int ph = 0; ph < (((colonneM1 - 1) / BLKSIZE) + 1); ph++) {
		if ((row < righeM1) && (threadIdx.x + (ph * BLKSIZE)) < colonneM1) {
			sA[threadIdx.y][threadIdx.x] = a[(row * colonneM1) + threadIdx.x + (ph * BLKSIZE)];
		}
		else {
			sA[threadIdx.y][threadIdx.x] = 0.0;
		}
		if (col < colonneM2 && (threadIdx.y + ph * BLKSIZE) < righeM2) {
			sB[threadIdx.y][threadIdx.x] = b[(threadIdx.y + ph * BLKSIZE) * colonneM2 + col];
		}
		else {
			sB[threadIdx.y][threadIdx.x] = 0.0;
		}
		__syncthreads(); // errore di Intellisense, non comporta problemi durante l'esecuzione

		for (int j = 0; j < BLKSIZE; ++j) {
			somma += sA[threadIdx.y][j] * sB[j][threadIdx.x];
		}
	}
	if (row < righeM1 && col < colonneM2) {
		c[row * colonneM2 + col] = somma;
	}
}


/*  Funzione eseguita sulla CPU (calcolo sequenziale)
    A differenza delle operazioni per cui il calcolo avviene simultaneamente per ogni fila ThreadY/ThreadX con l'unica analogia che riguarda lo scorrimento dei valori lungo le fasce,
	la CPU, operando in modo sequenziale, deve scorrerere un elemento per volta e moltiplicarlo per il giusto elemento dell'altra matrice  */

void matrix_mulCPU(int* a, int* b, int* c) {

	for (int i = 0; i < righeM1; i++) {
		for (int j = 0; j < colonneM2; j++) {
			//ogni volta avviene un reset della somma
			 int somma = 0;

			for (int k = 0; k < colonneM1; k++) {
				somma += a[i * colonneM1 + k] * b[k * colonneM2 + j];
			}

			c[i * colonneM2 + j] = somma;
		}
	}

	return;
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

	Si va poi a popolare la matrice host con i dati dei pixel dell'immagine iterando il comando matriceHost[i*colonneM1 + j]=(int)img.at<uchar>(i,j)
	I valori per un canale solo (Greyscale) vanno da 0 a 255

	*/

	cudaFree(0);

	// Creazione Cuda event, servirà per calcolare la durata delle operazioni che riguardano la GPU
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	float elapsed1 = 0; // time in ms
	float elapsed2 = 0; // time in ms
	float elapsed3 = 0; // time in ms

	// Clock per calcolare la durata della funzione eseguita sulla CPU
	clock_t inizio, fine;
	float tempo;

	puts(" ____Operazioni di moltiplicazione matriciale a conftonto CPU vs GPU____");
	cout << endl;

	// Restituisce alcuni parametri della scheda NVidia in uso
	cudaDeviceProp prop;
	size_t free1, free2, total;

	cudaGetDeviceProperties(&prop, 0);
		cudaCheckErrors("Errore acquisizione dati");
	cudaMemGetInfo(&free1, &total);
		cudaCheckErrors("Errore acquisizione dati");

	printf(" - Device: %s\n", prop.name);
	cout << " - GPU -> memory: free = " << free1 / (1024 * 1024) << " MegaBytes, total = " << total / (1024*1024*1024) << " GigaBytes" << endl;
	cout << endl;

	// Condizione di controllo  per confrontare le dimensioni delle matrici e non generare eccezioni durante l'elaborazione
	if (righeM2 != colonneM1) {

		cout << "Le colonne della prima matrice non corrispondono alle righe della seconda" << endl;
		cout << "ESECUZIONE ARRESTATA, ATTENZIONE IMMETTERE VALORI UGUALI PER LE DUE DIMENSIONI" << endl << endl;

		system("pause");
		return -1;
	}


	puts(" -- Allocazione delle variabili Host nella memoria --");
	// Allocazione matrice che si andrà a moltiplicare a quelle presenti nelle memorie

	int* matRandHost;
	cudaMallocHost((void **)&matRandHost, dimM2);

	// Generazione valori randomici e popolamento matrice
	puts(" - Generazione valori randomici per la prima e la seconda matrice");

	for (int i = 0; i < righeM2; i++)
		for (int j = 0; j < colonneM2; j++)
			matRandHost[i*colonneM2 + j] = rand() % 256;

	// Allocazione matrice di host
	int* matriceHost;
	cudaMallocHost((void **)&matriceHost, dimM1);

	// Popolamento matrice con valori randomici
	for (int i = 0; i < righeM1; i++)
		for (int j = 0; j < colonneM1; j++)
			matriceHost[i*colonneM1 + j] = rand() % 256;
	//matriceHost[i*colonneM1 + j] = (int)img.at<uchar>(i, j);

	// Matrici dei risultati e relativa allocazione
	int *matResHost, *matResHostSH, *matResCPU;

	cudaMallocHost((void **)&matResHost, dimRes);
	cudaMallocHost((void **)&matResHostSH, dimRes);
	matResCPU = (int *)malloc(dimRes);

	puts(" - Allocazione e popolamento matrici completati");
		cout << endl;


	// Allocazione di memoria per le variabili che lavoreranno sulla GPU
	puts(" -- Allocazione variabili nella memoria della GPU --");

	int *matriceGPU, *matRGPU, *matResGPU, *matResGPUSH;

	cudaMalloc((void **)&matriceGPU, dimM1);
		cudaCheckErrors("Allocazione fallita");
	cudaMalloc((void **)&matRGPU, dimM2);
		cudaCheckErrors("Allocazione fallita");
	cudaMalloc((void **)&matResGPU, dimRes);
		cudaCheckErrors("Allocazione fallita");
	cudaMalloc((void **)&matResGPUSH, dimRes);
		cudaCheckErrors("Allocazione fallita");

	puts(" - Allocazione completata");
		cout << endl;


	// Copia dei valori della prima e seconda matrice (host) nelle variabili device
	puts(" -- Trasferimento valori delle due matrici nella GPU --");

	cudaEventRecord(start);

	cudaMemcpy(matriceGPU, matriceHost, dimM1, cudaMemcpyHostToDevice);
		cudaCheckErrors("Copia dei dati da Host a Device fallita");
	cudaMemcpy(matRGPU, matRandHost, dimM2, cudaMemcpyHostToDevice);
		cudaCheckErrors("Copia dei dati da Host a Device fallita");
	cudaMemcpy(matResGPU, matResHost, dimRes, cudaMemcpyHostToDevice);
		cudaCheckErrors("Copia dei dati da Host a Device fallita");

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed1, start, stop);

	puts(" - Trasferimento completato");
		cout << " - Tempo trascorso: " << elapsed1 << " ms" << endl;
	cudaMemGetInfo(&free2, &total);
	cout << " - GPU -> memory: memoria occupata dalle matrici = " << (free1 - free2) / (1024*1024) << " MegaBytes" << endl;
		cout << endl;


	// Dimensionamento della griglia di blocchi e thread (max 1024 thread per blocco)
	puts(" -- Costruzione griglia di calcolo per la GPU --");

	dim3 block(BLKSIZE, BLKSIZE); // 32 * 32 = 1024 (colonne,righe)
	dim3 grid((int)ceil((colonneM2 + BLKSIZE - 1) / BLKSIZE), (int)ceil((righeM1 + BLKSIZE - 1) / BLKSIZE)); //trovo il valore intero più grande per costruire la griglia di dimensioni adeguate (colonne,righe)

	cout << endl;


	// Esecuzione funzione sulla GPU
	puts(" -- Avvio calcolo sulla GPU --");

	cudaEventRecord(start);

	matrix_mulGPU << <grid, block >> > (matriceGPU, matRGPU, matResGPU);
		cudaCheckErrors("Esecuzione del kernel Fallita");

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed2, start, stop);


	// Condizione del WDDM TDR DELAY
	if ((elapsed2 / 1000) == 0 || (elapsed2 / 1000) > 5) {
		cout << endl << "Sono passati piu' di 5 secondi, e' intervenuto il  TIMEOUT DETECTION & RECOVERY (WDDM TDR DELAY)" << endl;
		cout << "PROGRAMMA ARRESTATO, IMMETTERE MATRICI DI DIMENSIONI MINORI" << endl << endl;
		system("pause");
			return -1;
	}

	puts(" - Calcolo sulla GPU completato");
	cout << " - Tempo trascorso: " << elapsed2 / 1000 << " s" << endl;
	cout << endl;


	// Esecuzione funzione sulla GPU usando la shared memory
	puts(" -- Avvio calcolo sulla GPU utilizzando la Shared Memory --");

	cudaEventRecord(start);

	matrix_mulGPUShared << <grid, block >> > (matriceGPU, matRGPU, matResGPUSH);
	cudaCheckErrors("Esecuzione del kernel Fallita");

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed2, start, stop);


	// Condizione del WDDM TDR DELAY
	if ((elapsed2 / 1000) == 0 || (elapsed2 / 1000) > 5) {
		cout << endl << "Sono passati piu' di 5 secondi, e' intervenuto il  TIMEOUT DETECTION & RECOVERY (WDDM TDR DELAY)" << endl;
		cout << "PROGRAMMA ARRESTATO, IMMETTERE MATRICI DI DIMENSIONI MINORI" << endl << endl;
		system("pause");
		return -1;
	}

	puts(" - Calcolo sulla GPU completato");
	cout << " - Tempo trascorso: " << elapsed2 / 1000 << " s" << endl;
	cout << endl;


	// Trasferimento dei valori della matrice risultante dalla compilazione sulla GPU alla variabile Host
	puts(" -- Trasferimento valori della GPU alla matrice del Host del risultato --");

	cudaEventRecord(start);

	cudaMemcpyAsync(matResHost, matResGPU, dimRes, cudaMemcpyDeviceToHost);
		cudaCheckErrors("Trasferimento fallito\n");

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed3, start, stop);

	puts(" - Trasferimento completato");
	cout << " - Tempo trascorso: " << elapsed3 << " ms" << endl;
	printf(" - Larghezza di banda utilizzata (Device2H) per trasferire la matrice dei risultati (GB/s): %f\n", dimRes * 1e-6 / elapsed3);
		cout << endl;


		// Trasferimento dei valori della matrice risultante dalla compilazione sulla GPU alla variabile Host (SHARED MEMORY)
		puts(" -- Trasferimento valori della GPU alla matrice del Host del risultato (SM) --");

		cudaEventRecord(start);

		cudaMemcpyAsync(matResHostSH, matResGPUSH, dimRes, cudaMemcpyDeviceToHost);
		cudaCheckErrors("Trasferimento fallito\n");

		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsed3, start, stop);

		puts(" - Trasferimento completato");
		cout << " - Tempo trascorso: " << elapsed3 << " ms" << endl;
		printf(" - Larghezza di banda utilizzata (Device2H) per trasferire la matrice dei risultati (GB/s): %f\n", dimRes * 1e-6 / elapsed3);
		cout << endl;


	// Esecuzione funzione sulla CPU
	puts(" -- Avvio calcolo sulla CPU --");

	inizio = clock();

	matrix_mulCPU(matriceHost, matRandHost, matResCPU);

	fine = clock();
	tempo = ((float)(fine - inizio)) / CLOCKS_PER_SEC;

	puts(" - Calcolo sulla CPU eseguito");
	cout << " - Tempo trascorso: " << tempo << " s" << endl;
		cout << endl;


	// Funzione di confronto degli elementi nelle matrici ottenute dalla CPU e dalla GPU
	puts(" -- Controllo dei risultati --");
	bool esito = true;

	for (int i = 0; i < righeM1; i++) {
		if (esito != false) {
			for (int j = 0; j < colonneM2; j++) {
				if (matResCPU[i*colonneM2 + j] != matResHost[i*colonneM2 + j]) {
					cout << " --> ERRORE" << endl << endl;
					esito = false;
					break;
				}
				else if (matResCPU[i*colonneM2 + j] != matResHostSH[i*colonneM2 + j]) {
					cout << " --> ERRORE" << endl << endl;
					esito = false;
					break;
				}
				else if (matResHostSH[i*colonneM2 + j] != matResHost[i*colonneM2 + j]) {
					cout << " --> ERRORE" << endl << endl;
					esito = false;
					break;
				}
			}
		}
		else
			break;
	}

	if (esito)
		puts(" - Esito: completato senza aver riscontrato errori");
	else
		cout << " - Esito: ATTENZIONE SONO STATI RILEVATI VALORI DISCORDANTI";


	cudaFree(matriceGPU);
	cudaFree(matRGPU);
	cudaFree(matResGPU);
	cudaFree(matResGPUSH);

	cudaFreeHost(matRandHost);
	cudaFreeHost(matriceHost);
	cudaFreeHost(matResHost);
	cudaFreeHost(matResHostSH);

	cout << endl << endl;
	system("pause");

		return 0;
}