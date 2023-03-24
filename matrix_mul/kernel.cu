//librerie standard
#include <iostream>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <time.h>
#include <math.h>
#include "matrixClass.h"
#include "Funzioni_GPU.h"
#include "Funzioni_CPU.h"
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
            fprintf(stderr, "Fatal error: %s(%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
               fprintf(stderr, "\n *** FAILED - ABORTING\n"); \
            system("pause");\
            exit(1); \
        } \
    } while (0)


/*  La matrice risultante dal prodotto avrà dimesioni righeM1 * colonneM2 (righe della prima e colonne della seonda)

     -> IL PRODOTTO TRA MATRICI NON E' COMMUTATIVO  */


//Creazione oggetto Matrice con caratteristiche prese dal file "matrixClass.h"
	Matrice mat;
	Blocco B;

//Griglia GPU (GM)
	 dim3 block(B.BLKSIZE, B.BLKSIZE);   // 32 * 32 = 1024 (colonne,righe)
	 dim3 grid((int)ceil((mat.colonneM2 + B.BLKSIZE - 1) / B.BLKSIZE), (int)ceil((mat.righeM1 + B.BLKSIZE - 1) / B.BLKSIZE));   //trovo il valore intero più grande per costruire la griglia di dimensioni adeguate (colonne,righe)

//Griglia GPU (SM)
	 dim3 blocksh(B.BLOCK, B.BLOCK); // 16 * 16 = 256 dimesione del blocco uguale alla dimensione delle matrici situate nella shared memory (più piccoli di quelli usati nella global memory garantiscono performance migliori)
	 dim3 gridsh((int)ceil((mat.colonneM2 + B.BLOCK - 1) / B.BLOCK), (int)ceil((mat.righeM1 + B.BLOCK - 1) / B.BLOCK));


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

	cudaFree(0); // inizializzo il device (può essere omesso)


	// Creazione Cuda event, servirà per calcolare la durata delle operazioni che riguardano la GPU
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	float elapsed1 = 0; // time in ms
	float elapsed2 = 0; // time in ms
	float elapsedsh = 0; // time in ms
	float elapsed3 = 0; // time in ms

	// Clock per calcolare la durata della funzione eseguita sulla CPU
	clock_t inizio, fine, chkGPUs, chkGPUe;
	float tempo;
	float tempogm;
	float temposm;

	puts(" ____OPERAZIONI DI MOLTIPLICAZIONE MATRICIALE A CONFRONTO, CPU vs GPU____");
	cout << endl;

	// Restituisce alcuni parametri della scheda NVidia in uso
	cudaDeviceProp prop;
	size_t free1, free2, total;

	cudaGetDeviceProperties(&prop, 0);
		cudaCheckErrors("Errore acquisizione dati\n");
	cudaMemGetInfo(&free1, &total);
		cudaCheckErrors("Errore acquisizione dati\n");

	printf(" - Device: %s\n", prop.name);
	cout << " - GPU -> memory: free = " << free1 / (1024 * 1024) << " MegaBytes, total = " << total / (1024*1024*1024) << " GigaBytes" << endl;
		cout << endl << endl;

	// Condizione di controllo  per confrontare le dimensioni delle matrici e non generare eccezioni durante l'elaborazione
	if (mat.righeM2 != mat.colonneM1) {

		cout << "Le colonne della prima matrice non corrispondono alle righe della seconda" << endl;
		cout << "ESECUZIONE ARRESTATA, ATTENZIONE IMMETTERE VALORI UGUALI PER LE DUE DIMENSIONI" << endl << endl;

		system("pause");
		return -1;
	}

	cout << fixed << setprecision(3);


	/*----------------------------------------------------------------------------------------------------------------------------------------------------*/


	puts(" - Allocazione delle variabili Host (RAM) -");
	// Allocazione matrice che si andrà a moltiplicare a quelle presenti nelle memorie

	int* matRandHost;
	cudaMallocHost((void **)&matRandHost, mat.dimM2);
		cudaCheckErrors("Allocazione fallita\n");

	// Allocazione matrice di host

	int* matriceHost;
	cudaMallocHost((void **)&matriceHost, mat.dimM1);
		cudaCheckErrors("Allocazione fallita\n");

	// Matrici dei risultati e relativa allocazione

	int *matResHost, *matResHostSH, *matResCPU;

	cudaMallocHost((void **)&matResHost, mat.dimRes);
		cudaCheckErrors("Allocazione fallita\n");
	cudaMallocHost((void **)&matResHostSH, mat.dimRes);
		cudaCheckErrors("Allocazione fallita\n");
	matResCPU = (int *)malloc(mat.dimRes);
		if (matResCPU == NULL) {
			fprintf(stderr, "Fatal: malloc ha fallito nell'allocare %zu bytes.\n", mat.dimRes);
			system("pause");
			exit(1);
		}

	puts("    Allocazione matrici completata");
		cout << endl << endl;


	/*----------------------------------------------------------------------------------------------------------------------------------------------------*/


	// Generazione valori randomici e popolamento matrici
	puts(" - Generazione e popolamento con valori randomici per la prima e la seconda matrice");

	//Rand
	for (int i = 0; i < mat.righeM2; i++)
		for (int j = 0; j < mat.colonneM2; j++)
			matRandHost[i*mat.colonneM2 + j] = rand() % 256;

	//Host
	for (int i = 0; i < mat.righeM1; i++)
		for (int j = 0; j < mat.colonneM1; j++)
			matriceHost[i*mat.colonneM1 + j] = rand() % 256;
	//matriceHost[i*colonneM1 + j] = (int)img.at<uchar>(i, j);

	cout << endl << endl;


	/*----------------------------------------------------------------------------------------------------------------------------------------------------*/


	// Allocazione di memoria per le variabili che lavoreranno sulla GPU
	puts(" - Allocazione variabili nella memoria della GPU -");

	int *matriceGPU, *matRGPU, *matResGPU, *matResGPUSH;

	cudaMalloc((void **)&matriceGPU, mat.dimM1);
		cudaCheckErrors("Allocazione fallita\n");
	cudaMalloc((void **)&matRGPU, mat.dimM2);
		cudaCheckErrors("Allocazione fallita\n");
	cudaMalloc((void **)&matResGPU, mat.dimRes);
		cudaCheckErrors("Allocazione fallita\n");
	cudaMalloc((void **)&matResGPUSH,mat.dimRes);
		cudaCheckErrors("Allocazione fallita\n");

	puts("    Allocazione completata");
		cout << endl << endl;


	/*----------------------------------------------------------------------------------------------------------------------------------------------------*/


	// Copia dei valori della prima e seconda matrice (host) nelle variabili device
	puts(" - Trasferimento valori delle due matrici nella GPU -");

	cudaEventRecord(start);
		cudaCheckErrors("Errore, impossibile avviare cudaEventRecord\n");

	cudaMemcpy(matriceGPU, matriceHost, mat.dimM1, cudaMemcpyHostToDevice);
		cudaCheckErrors("Copia dei dati da Host a Device fallita\n");
	cudaMemcpy(matRGPU, matRandHost, mat.dimM2, cudaMemcpyHostToDevice);
		cudaCheckErrors("Copia dei dati da Host a Device fallita\n");

	// questi due cudaMemcpy possono essere omesse perché non viene copiato nulla visto che le matrici dei risultati sono vuote
	cudaMemcpy(matResGPU, matResHost, mat.dimRes, cudaMemcpyHostToDevice);
		cudaCheckErrors("Copia dei dati da Host a Device fallita\n");
	cudaMemcpy(matResGPUSH, matResHost, mat.dimRes, cudaMemcpyHostToDevice);
		cudaCheckErrors("Copia dei dati da Host a Device fallita\n");

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed1, start, stop);
		cudaCheckErrors("Errore di timing durante copia dei dati\n");

	puts("    Trasferimento completato");
		cout << "    Tempo trascorso: " << elapsed1 << " ms" << endl;

	cudaMemGetInfo(&free2, &total);
		cudaCheckErrors("Errore acquisizione dati\n");

		cout << "    GPU -> memory: memoria occupata dalle matrici = " << (free1 - free2) / (1024*1024) << " MegaBytes" << endl;
		printf("    Larghezza di banda utilizzata (Host2D) durante il caricamento delle matrici (GB/s): %f\n", ((mat.dimRes * 2) + mat.dimM1 + mat.dimM2) * 1e-6 / elapsed1);
		cout << endl << endl;


	/*----------------------------------------------------------------------------------------------------------------------------------------------------*/


	// Esecuzione funzione sulla GPU
	puts(" - Avvio calcolo sulla GPU -");

	cudaEventRecord(start); chkGPUs = clock();
		cudaCheckErrors("Errore, impossibile avviare cudaEventRecord\n");
	
	matrix_mulGPU << <grid, block >> > (matriceGPU, matRGPU, matResGPU);
	// per intercettare e gestire errori del kernel (asincroni), la procedura di controllo prevede due istruzioni, la prima è la seguente che identifica errori sincroni
	// la seconda si tratta di lanciare nuovamente la funzione di intercettazione delle eccezioni per errori asincroni
	cudaCheckErrors("Errore sincrono del kernel, parametri della chiamata non validi\n");

	cudaEventRecord(stop);
	cudaEventSynchronize(stop); chkGPUe = clock();

	// Condizione del WDDM TDR DELAY
	tempogm = ((int)(chkGPUe - chkGPUs)) / CLOCKS_PER_SEC;
	if (tempogm >= 5)
		cout << endl << "Sono passati piu' di 5 secondi, e' intervenuto il  TIMEOUT DETECTION & RECOVERY (WDDM TDR DELAY)" << endl << endl;

	cudaCheckErrors("Il kernel ha riscontrato un errore durante l'esecuzione o e' stata interrotta la sua elaborazione\n"); // (seconda funzione dopo un'azione di sync) gestione degli errori cuda del kernel diversa da una gestione normale

	cudaEventElapsedTime(&elapsed2, start, stop);
		cudaCheckErrors("Errore di timing della funzione eseguita sulla GPU (GM)\n");

	puts("    Calcolo sulla GPU completato");
	cout << "    Tempo trascorso: " << (elapsed2 / 1000) << " s"<< endl;
	cout << endl << endl;


	/*----------------------------------------------------------------------------------------------------------------------------------------------------*/


	// Esecuzione funzione sulla GPU usando la shared memory
	puts(" - Avvio calcolo sulla GPU utilizzando la Shared Memory -");

	if (BLK != B.BLOCK) {
		cout << endl << "ATTENZIONE: PROGRAMMA ARRESTATO,\nLE DIMENSIONI USATE PER LE MATRICI NELLA SHARED MEMORY NON COINCIDONO CON QUELLE IMPOSTATE NELLA CLASSE" << endl << endl;
		system("pause");
		exit(1);
	}

	cudaEventRecord(start); chkGPUs = clock();
		cudaCheckErrors("Errore, impossibile avviare cudaEventRecord\n");

	matrix_mulGPUShared << <gridsh, blocksh >> > (matriceGPU, matRGPU, matResGPUSH);
		cudaCheckErrors("Errore sincrono del kernel, parametri della chiamata non validi\n");

	cudaEventRecord(stop);
	cudaEventSynchronize(stop); chkGPUe = clock();

	// Condizione del WDDM TDR DELAY
	temposm = ((int)(chkGPUe - chkGPUs)) / CLOCKS_PER_SEC;
	if (temposm >= 5)
		cout << endl << "Sono passati piu' di 5 secondi, e' intervenuto il  TIMEOUT DETECTION & RECOVERY (WDDM TDR DELAY)" << endl << endl;

	cudaCheckErrors("Il kernel ha riscontrato un errore durante l'esecuzione o e' stata interrotta la sua elaborazione\n");
	
	cudaEventElapsedTime(&elapsedsh, start, stop);
		cudaCheckErrors("Errore di timing della funzione eseguita sulla GPU (SM)\n");

	puts("    Calcolo sulla GPU completato");
		cout << "    Tempo trascorso: " << (elapsedsh / 1000) << " s" << endl;
		cout << endl << endl;


	/*----------------------------------------------------------------------------------------------------------------------------------------------------*/


	// Trasferimento dei valori della matrice risultante dalla compilazione sulla GPU alla variabile Host
	puts(" - Trasferimento valori della GPU alla matrice del Host del risultato (GM & SM) -");

	cudaEventRecord(start);

	/* Non viene usato cudaMemcpyAsync (che assicura di avere tutti i dati prima di procedere alla loro elaborazione) per due motivi:

	   1. abbiamo cudaEventSynchronize che assicura, dopo essere stato chiamato in un blocco cudaEventRecord, 
	   che tutte le operazioni del device siano concluse prima di procedere con l'istruzione successiva

	   2. c'è tempo di recuperarli durante la lenta esecuzione del calcolo sulla CPU (potrebbe non essere valido per matrici molto piccole) */

	cudaMemcpy(matResHost, matResGPU, mat.dimRes, cudaMemcpyDeviceToHost);
		cudaCheckErrors("Trasferimento fallito\n");
	cudaMemcpy(matResHostSH, matResGPUSH, mat.dimRes, cudaMemcpyDeviceToHost);
		cudaCheckErrors("Trasferimento fallito\n");

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed3, start, stop);
		cudaCheckErrors("Errore di timing durante copia dei dati\n");

	puts("    Trasferimento completato");
		cout << "    Tempo trascorso: " << elapsed3 << " ms" << endl;
		printf("    Larghezza di banda utilizzata (Device2H) durante il trasferimento delle matrici dei risultati (GB/s): %f\n", mat.dimRes * 2 * 1e-6 / elapsed3);
		cout << endl << endl;


	/*----------------------------------------------------------------------------------------------------------------------------------------------------*/


	// Esecuzione funzione sulla CPU
	puts(" - Avvio calcolo sulla CPU -");

	inizio = clock();

	matrix_mulCPU(matriceHost, matRandHost, matResCPU);

	fine = clock();
	tempo = ((float)(fine - inizio)) / CLOCKS_PER_SEC;

	puts("    Calcolo sulla CPU eseguito");
	cout << "    Tempo trascorso: " << tempo << " s" << endl;
		cout << endl << endl;


	/*----------------------------------------------------------------------------------------------------------------------------------------------------*/


	// Controllo della correttezza dei risultati
	puts(" - Controllo dei risultati -");

	if (checkRes(matResCPU, matResHost, matResHostSH))
		puts("    Esito: COMPLETATO SENZA AVER INDIVIDUATO INCONGRUENZE");
	else
		cout << "--> ERRORE"<< endl <<"    Esito: ATTENZIONE SONO STATI RILEVATI VALORI DISCORDANTI";


	/*----------------------------------------------------------------------------------------------------------------------------------------------------*/


	// visualizzazione dei tempi come tabella 
	cout << endl << endl;

	string elap2 = to_string(elapsed2/1000);
	string e2 = elap2.substr(0, 7);
	string elapsh = to_string(elapsedsh/1000);
	string esh = elapsh.substr(0, 7);
	string tem = to_string(tempo);
	string te = tem.substr(0, 8);

	TextTable t('-', '|', '+');

	t.add(" GPU (using Global Memory) [s] ");
	t.add(" GPU (using Shared Memory) [s] ");
	t.add(" CPU [s] ");
	t.endOfRow();

	t.add(e2);
	t.add(esh);
	t.add(te);
	t.endOfRow();

	t.setAlignment(3, TextTable::Alignment::RIGHT);
    cout << t;


	/*----------------------------------------------------------------------------------------------------------------------------------------------------*/


	// Deallocazione di tutte le variabili nelle memorie

	cudaFreeHost(matRandHost);
		cudaCheckErrors("Errore, impossibile deallocare la memoria\n");
	cudaFreeHost(matriceHost);
		cudaCheckErrors("Errore, impossibile deallocare la memoria\n");
	cudaFreeHost(matResHost);
		cudaCheckErrors("Errore, impossibile deallocare la memoria\n");
	cudaFreeHost(matResHostSH);
		cudaCheckErrors("Errore, impossibile deallocare la memoria\n");

	cudaFree(matriceGPU);
		cudaCheckErrors("Errore, impossibile deallocare la memoria\n");
	cudaFree(matRGPU);
		cudaCheckErrors("Errore, impossibile deallocare la memoria\n");
	cudaFree(matResGPU);
		cudaCheckErrors("Errore, impossibile deallocare la memoria\n");
	cudaFree(matResGPUSH);
		cudaCheckErrors("Errore, impossibile deallocare la memoria\n");

	free(matResCPU);



	cout << endl << endl << endl;
	system("pause");
		return 0;
}