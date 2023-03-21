#pragma once

#include <iostream>
#include <stdio.h>
#include "matrixClass.h"

// librerie CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>

#ifndef blocco
#define BLK 16
#endif // !blocco


// Funzione eseguita sulla GPU (calcolo parallelo)
// viene utilizzata la memoria globale (qui le variabili infatti possono essere allocate dinamicamente con le APIs cudamalloc e cudamallocHost)
__global__ void matrix_mulGPU(int *a, int *b, int *c) {

	Matrice mat;

	// Inizializzo le coordinate dei thread all'interno della griglia (col e row identificano un singolo thread specifico)
	// Vengono mappate le posizioni degli elementi delle metrici facendole corrispondere alle posizioni dei thread ogni elemente verra elaborato sul thread per lui predisposto
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	// Somma impostata a zero, aumenterà man mano che, come avviene nel prodotto tra matrici, si sommano gli elementi moltiplicati

	int somma = 0;

	/*  viene posta una condizione di controllo affinché vengano presi solo gli elementi opportuni negli array delle matrici
	    (la griglia creata è impostata sempre leggermente più grande delle dimensioni del prodotto righe*colonne)  */

	if (row < mat.righeM1 && col < mat.colonneM2) {

		/*  l'iterazione parte ed è svolta simultaneamente per ogni gruppo (riga/colonna -> y/x) di thread: 
		    per la matrice M1 si percorrono le righe della griglia (thread posti orizzontalmente sull'asse x, cioè le colonneM1)
		    per la matrice M2 invece accade la stessa cosa ma si procede scorrendo lungo i thread posti verticalmente (asse y, cioè le righeM2)
		    row e col mantengono "fissato" il calcolo sulle righe e colonne corrispondenti  */

		for (int i = 0; i < mat.colonneM1; i++) {

			// la somma accumula i prodotti che man mano vanno aggiungendosi, scorrendo infatti lungo le dimensioni x e y

			somma += a[row * mat.colonneM1 + i] * b[i * mat.colonneM2 + col];
			// la durata computazionale del processo è data proprio da quest'ultima stringa che dipende direttamente dalle dimensioni delle matrici in esame: il calcolo effettuato è (M1*M2)

			// si sincronizzano tutti i thread così che tutte le operazioni (calcolo dei risultati) finiscano nello stesso momento
			__syncthreads(); // errore di Intellisense, non comporta problemi durante l'esecuzione 
		}
		/* alla fine di ogni iterazione vengono popolati in modo gli elementi nell'array del risultato.
		   Ad esempio:
		   l' elemento [0,0] della nuova matrice sarà il risultato della somma di tutti i prodotti tra gli elementi della riga 0 della prima matrice 
		   e egli elemnti della colonna 0 della seconda matrice  */

		c[row * mat.colonneM2 + col] = somma;
	}
}

__global__ void matrix_mulGPUShared(int *a, int *b, int *c) {

	Matrice mat;

	// queste due matrici vengono caricate sulla shared memory e lì lavoreranno(sotto-matrici)
	// più i blocchi sono piccoli più lavora veloce la funzione
	__shared__ int sA[BLK][BLK];   // sA e sB usano blocchi da 16*16 = 256 thread
	__shared__ int sB[BLK][BLK];

	// coordinate dei thread
	int row = blockDim.y * blockIdx.y + threadIdx.y;
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	
	int somma = 0;

	// inizializzo gli elementi delle sottomatrici a zero
	sA[threadIdx.y][threadIdx.x] = 0;
	sB[threadIdx.y][threadIdx.x] = 0;
		
	// l'iterazione prima assegna in modo opportuno gli elementi ai blocchi e non disponendoli solo in righe e colonne di thread.
	// Con la memoria globale si prende come punto di riferimento la griglia, qui vengono sfruttati i blocchi
	// si fa in modo che i dati vengano caricati in blocchi creati appositamente a partire dalle dimensioni delle matrici di partenza adattandoli
	for (int i = 0; i < (((mat.colonneM1 - 1) / BLK) + 1); i++) {
		if ((row < mat.righeM1) && (threadIdx.x + (i * BLK)) < mat.colonneM1) {
			sA[threadIdx.y][threadIdx.x] = a[(row * mat.colonneM1) + threadIdx.x + (i * BLK)];
		}
		else {
			sA[threadIdx.y][threadIdx.x] = 0;
		}

		if (col < mat.colonneM2 && (threadIdx.y + i * BLK) < mat.righeM2) {
			sB[threadIdx.y][threadIdx.x] = b[(threadIdx.y + i * BLK) * mat.colonneM2 + col];
		}
		else {
			sB[threadIdx.y][threadIdx.x] = 0;
		}
		
		// tutti i blocchi dovranno aver finito nello stesso istante
		__syncthreads(); // errore di Intellisense, non comporta problemi durante l'esecuzione

		for (int j = 0; j < BLK; ++j) {
			somma += sA[threadIdx.y][j] * sB[j][threadIdx.x];
			__syncthreads();
		}
	}
	if (row < mat.righeM1 && col < mat.colonneM2) {
		c[row * mat.colonneM2 + col] = somma;
	}
}