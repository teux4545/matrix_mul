//librerie standard
#include <iostream>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <time.h>
#include <math.h>
#include "TextTable.h"

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
#define BLK 16

// Funzione eseguita sulla GPU (calcolo parallelo)
// viene utilizzata la memoria globale (qui le variabili infatti possono essere allocate dinamicamente con le APIs cudamalloc e cudamallocHost)
__global__ void matrix_mulGPU(int *a, int *b, int *c) {

	// Inizializzo le coordinate dei thread all'interno della griglia (col e row identificano un singolo thread specifico)
	// Vengono mappate le posizioni degli elementi delle metrici facendole corrispondere alle posizioni dei thread ogni elemente verra elaborato sul thread per lui predisposto
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	// Somma impostata a zero, aumenterà man mano che, come avviene nel prodotto tra matrici, si sommano gli elementi moltiplicati

	int somma = 0;

	/*  viene posta una condizione di controllo affinché vengano presi solo gli elementi opportuni negli array delle matrici
	    (la griglia creata è impostata sempre leggermente più grande delle dimensioni del prodotto righe*colonne)  */

	if (row < righeM1 && col < colonneM2) {

		/*  l'iterazione parte ed è svolta simultaneamente per ogni gruppo (riga/colonna -> y/x) di thread: 
		    per la matrice M1 si percorrono le righe della griglia (thread posti orizzontalmente sull'asse x, cioè le colonneM1)
		    per la matrice M2 invece accade la stessa cosa ma si procede scorrendo lungo i thread posti verticalmente (asse y, cioè le righeM2)
		    row e col mantengono "fissato" il calcolo sulle righe e colonne corrispondenti  */

		for (int i = 0; i < colonneM1; i++) {

			// la somma accumula i prodotti che man mano vanno aggiungendosi, scorrendo infatti lungo le dimensioni x e y

			somma += a[row * colonneM1 + i] * b[i * colonneM2 + col];
			// la durata computazionale del processo è data proprio da quest'ultima stringa che dipende direttamente dalle dimensioni delle matrici in esame: il calcolo effettuato è (M1*M2)
			__syncthreads();
		}
		/* alla fine di ogni iterazione vengono popolati in modo gli elementi nell'array del risultato.
		   Ad esempio:
		   l' elemento [0,0] della nuova matrice sarà il risultato della somma di tutti i prodotti tra gli elementi della riga 0 della prima matrice 
		   e egli elemnti della colonna 0 della seconda matrice  */

		c[row * colonneM2 + col] = somma;
	}
}

__global__ void matrix_mulGPUShared(int *a, int *b, int *c) {

	// queste due matrici lavoreranno sulla shared memory (sotto-matrici)
	// più i blocchi sono piccoli più lavora veloce la funzione
	__shared__ int sA[BLK][BLK];   // sA usa un blocco da 32*32 = 1024 thread
	__shared__ int sB[BLK][BLK];   // sB usa un blocco da 32*32 = 1024 thread

	// coordinate dei thread
	int row = blockDim.y * blockIdx.y + threadIdx.y;
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	
	int somma = 0;
	// inizializzo gli elementi delle sottomatrici a zero
	sA[threadIdx.y][threadIdx.x] = 0;
	sB[threadIdx.y][threadIdx.x] = 0;
		
	for (int i = 0; i < (((colonneM1 - 1) / BLK) + 1); i++) {
		if ((row < righeM1) && (threadIdx.x + (i * BLK)) < colonneM1) {
			sA[threadIdx.y][threadIdx.x] = a[(row * colonneM1) + threadIdx.x + (i * BLK)];
		}
		else {
			sA[threadIdx.y][threadIdx.x] = 0;
		}

		if (col < colonneM2 && (threadIdx.y + i * BLK) < righeM2) {
			sB[threadIdx.y][threadIdx.x] = b[(threadIdx.y + i * BLK) * colonneM2 + col];
		}
		else {
			sB[threadIdx.y][threadIdx.x] = 0;
		}
		
		__syncthreads(); // errore di Intellisense, non comporta problemi durante l'esecuzione

		for (int j = 0; j < BLK; ++j) {
			somma += sA[threadIdx.y][j] * sB[j][threadIdx.x];
			__syncthreads();
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
	float elapsedsh = 0; // time in ms
	float elapsed3 = 0; // time in ms

	// Clock per calcolare la durata della funzione eseguita sulla CPU
	clock_t inizio, fine;
	float tempo;

	puts(" ____OPERAZIONI DI MOLTIPLICAZIONE MATRICIALE A CONFRONTO, CPU vs GPU____");
		cout << endl << endl;

	// Restituisce alcuni parametri della scheda NVidia in uso
	cudaDeviceProp prop;
	size_t free1, free2, total;

	cudaGetDeviceProperties(&prop, 0);
		cudaCheckErrors("Errore acquisizione dati");
	cudaMemGetInfo(&free1, &total);
		cudaCheckErrors("Errore acquisizione dati");

	printf(" - Device: %s\n", prop.name);
	cout << " - GPU -> memory: free = " << free1 / (1024 * 1024) << " MegaBytes, total = " << total / (1024*1024*1024) << " GigaBytes" << endl;
		cout << endl << endl;

	// Condizione di controllo  per confrontare le dimensioni delle matrici e non generare eccezioni durante l'elaborazione
	if (righeM2 != colonneM1) {

		cout << "Le colonne della prima matrice non corrispondono alle righe della seconda" << endl;
		cout << "ESECUZIONE ARRESTATA, ATTENZIONE IMMETTERE VALORI UGUALI PER LE DUE DIMENSIONI" << endl << endl;

		system("pause");
		return -1;
	}

	cout << fixed << setprecision(3);


	puts(" - Allocazione delle variabili Host (RAM) -");
	// Allocazione matrice che si andrà a moltiplicare a quelle presenti nelle memorie

	int* matRandHost;
	cudaMallocHost((void **)&matRandHost, dimM2);

	// Generazione valori randomici e popolamento matrice
	puts("    Generazione valori randomici per la prima e la seconda matrice");

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

	puts("    Allocazione e popolamento matrici completati");
		cout << endl << endl;


	// Allocazione di memoria per le variabili che lavoreranno sulla GPU
	puts(" - Allocazione variabili nella memoria della GPU -");

	int *matriceGPU, *matRGPU, *matResGPU, *matResGPUSH;

	cudaMalloc((void **)&matriceGPU, dimM1);
		cudaCheckErrors("Allocazione fallita");
	cudaMalloc((void **)&matRGPU, dimM2);
		cudaCheckErrors("Allocazione fallita");
	cudaMalloc((void **)&matResGPU, dimRes);
		cudaCheckErrors("Allocazione fallita");
	cudaMalloc((void **)&matResGPUSH, dimRes);
		cudaCheckErrors("Allocazione fallita");

	puts("    Allocazione completata");
		cout << endl << endl;


	// Copia dei valori della prima e seconda matrice (host) nelle variabili device
	puts(" - Trasferimento valori delle due matrici nella GPU -");

	cudaEventRecord(start);

	cudaMemcpy(matriceGPU, matriceHost, dimM1, cudaMemcpyHostToDevice);
		cudaCheckErrors("Copia dei dati da Host a Device fallita");
	cudaMemcpy(matRGPU, matRandHost, dimM2, cudaMemcpyHostToDevice);
		cudaCheckErrors("Copia dei dati da Host a Device fallita");

	// questi due cudaMemcpy possono essere omesse perché non viene copiato nulla visto che le matrici dei risultati sono vuote
	cudaMemcpy(matResGPU, matResHost, dimRes, cudaMemcpyHostToDevice);
		cudaCheckErrors("Copia dei dati da Host a Device fallita");
	cudaMemcpy(matResGPUSH, matResHost, dimRes, cudaMemcpyHostToDevice);
		cudaCheckErrors("Copia dei dati da Host a Device fallita");

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed1, start, stop);

	puts("    Trasferimento completato");
		cout << "    Tempo trascorso: " << elapsed1 << " ms" << endl;
	cudaMemGetInfo(&free2, &total);
		cout << "    GPU -> memory: memoria occupata dalle matrici = " << (free1 - free2) / (1024*1024) << " MegaBytes" << endl;
		printf("    Larghezza di banda utilizzata (Host2D) durante il caricamento delle matrici (GB/s): %f\n", ((dimRes * 2) + dimM1 + dimM2) * 1e-6 / elapsed1);
		cout << endl << endl;


	// Dimensionamento della griglia di blocchi e thread (max 1024 thread per blocco)
	puts(" - Costruzione griglia di calcolo per la GPU -");

	dim3 block(BLKSIZE, BLKSIZE); // 32 * 32 = 1024 (colonne,righe)
	dim3 grid((int)ceil((colonneM2 + BLKSIZE - 1) / BLKSIZE), (int)ceil((righeM1 + BLKSIZE - 1) / BLKSIZE)); //trovo il valore intero più grande per costruire la griglia di dimensioni adeguate (colonne,righe)

		cout << endl;


	// Esecuzione funzione sulla GPU
	puts(" - Avvio calcolo sulla GPU -");

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

	puts("    Calcolo sulla GPU completato");
	cout << "    Tempo trascorso: " << (elapsed2 / 1000) << " s" << endl;
	cout << endl << endl;


	// Dimensionamento della griglia di blocchi e thread (max 1024 thread per blocco)
	puts(" - Costruzione griglia di calcolo per la GPU (SH) -");

	dim3 blocky(BLK, BLK); 
	dim3 gridy((int)ceil((colonneM2 + BLK - 1) / BLK), (int)ceil((righeM1 + BLK - 1) / BLK)); 

	cout << endl;


	// Esecuzione funzione sulla GPU usando la shared memory
	puts(" - Avvio calcolo sulla GPU utilizzando la Shared Memory -");
	
	cudaEventRecord(start);

	matrix_mulGPUShared << <gridy, blocky >> > (matriceGPU, matRGPU, matResGPUSH);
	cudaCheckErrors("Esecuzione del kernel Fallita");

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedsh, start, stop);


	// Condizione del WDDM TDR DELAY
	if ((elapsedsh / 1000) == 0 || (elapsedsh / 1000) > 5) {
		cout << endl << "Sono passati piu' di 5 secondi, e' intervenuto il  TIMEOUT DETECTION & RECOVERY (WDDM TDR DELAY)" << endl;
		cout << "PROGRAMMA ARRESTATO, IMMETTERE MATRICI DI DIMENSIONI MINORI" << endl << endl;
		system("pause");
		return -1;
	}

	puts("    Calcolo sulla GPU completato");
		cout << "    Tempo trascorso: " << (elapsedsh / 1000) << " s" << endl;
		cout << endl << endl;


	// Trasferimento dei valori della matrice risultante dalla compilazione sulla GPU alla variabile Host
	puts(" - Trasferimento valori della GPU alla matrice del Host del risultato (GM & SM) -");

	cudaEventRecord(start);

	/* Non viene usato cudaMemcpyAsync (che assicura di avere tutti i dati prima di procedere alla loro elaborazione) per due motivi:

	   1. abbiamo cudaEventSynchronize che assicura, dopo essere stato chiamato in un blocco cudaEventRecord, 
	   che tutte le operazioni del device siano concluse prima di procedere con l'istruzione successiva

	   2. c'è tempo di recuperarli durante la lenta esecuzione del calcolo sulla CPU (potrebbe non essere valido per matrici molto piccole) */

	cudaMemcpy(matResHost, matResGPU, dimRes, cudaMemcpyDeviceToHost);
		cudaCheckErrors("Trasferimento fallito\n");
	cudaMemcpy(matResHostSH, matResGPUSH, dimRes, cudaMemcpyDeviceToHost);
		cudaCheckErrors("Trasferimento fallito\n");

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed3, start, stop);

	puts("    Trasferimento completato");
		cout << "    Tempo trascorso: " << elapsed3 << " ms" << endl;
		printf("    Larghezza di banda utilizzata (Device2H) durante il trasferimento delle matrici dei risultati (GB/s): %f\n", dimRes * 2 * 1e-6 / elapsed3);
		cout << endl << endl;


	// Esecuzione funzione sulla CPU
	puts(" - Avvio calcolo sulla CPU -");

	inizio = clock();

	matrix_mulCPU(matriceHost, matRandHost, matResCPU);

	fine = clock();
	tempo = ((float)(fine - inizio)) / CLOCKS_PER_SEC;

	puts("    Calcolo sulla CPU eseguito");
	cout << "    Tempo trascorso: " << tempo << " s" << endl;
		cout << endl << endl;


	// Funzione di confronto degli elementi nelle matrici ottenute dalla CPU e dalla GPU
	puts(" - Controllo dei risultati -");
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
		puts("    Esito: COMPLETATO SENZA AVER INDIVIDUATO INCONGRUENZE");
	else
		cout << "    Esito: ATTENZIONE SONO STATI RILEVATI VALORI DISCORDANTI";


	// visualizzazione dei tempi come tabella 
	cout << endl;

	string elap2 = to_string(elapsed2/1000);
	string e2 = elap2.substr(0,6);
	string elapsh = to_string(elapsedsh/1000);
	string esh = elapsh.substr(0, 6);
	string tem = to_string(tempo);
	string te = tem.substr(0, 6);

	TextTable t('-', '|', '+');

	t.add(" GPU (using Global Memory) [s] ");
	t.add(" GPU (using Shared Memory) [s] ");
	t.add(" CPU [s] ");
	t.endOfRow();

	t.add(e2);
	t.add(esh);
	t.add(te);
	t.endOfRow();

	t.setAlignment(2, TextTable::Alignment::RIGHT);
    cout << t;


	// Deallocazione di tutte le variabili nelle memorie

	cudaFree(matriceGPU);
	cudaFree(matRGPU);
	cudaFree(matResGPU);
	cudaFree(matResGPUSH);

	cudaFreeHost(matRandHost);
	cudaFreeHost(matriceHost);
	cudaFreeHost(matResHost);
	cudaFreeHost(matResHostSH);


	cout << endl << endl << endl;
	system("pause");
		return 0;
}