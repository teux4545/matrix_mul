#pragma once

#include <iostream>
#include "matrixClass.h"

/*  Funzione eseguita sulla CPU (calcolo sequenziale)
A differenza delle operazioni per cui il calcolo avviene simultaneamente per ogni fila ThreadY/ThreadX con l'unica analogia che riguarda lo scorrimento dei valori lungo le fasce,
la CPU, operando in modo sequenziale, deve scorrerere un elemento per volta e moltiplicarlo per il giusto elemento dell'altra matrice  */

void matrix_mulCPU(int* a, int* b, int* c) {

	Matrice mat;

	for (int i = 0; i < mat.righeM1; i++) {
		for (int j = 0; j < mat.colonneM2; j++) {
			//ogni volta avviene un reset della somma
			int somma = 0;

			for (int k = 0; k < mat.colonneM1; k++) {
				somma += a[i * mat.colonneM1 + k] * b[k * mat.colonneM2 + j];
			}

			c[i * mat.colonneM2 + j] = somma;
		}
	}

	return;
}


// Funzione di confronto degli elementi nelle matrici ottenute dalla CPU e dalla GPU
// Potrebbe essere usato un singolo ciclo con i<(righeM1*colonneM2)

bool checkRes(int *matResCPU, int *matResHost, int *matResHostSH) {

	Matrice mat;
	bool esito = true;

	for (int i = 0; i < mat.righeM1; i++) {
		if (esito != false) {
			for (int j = 0; j < mat.colonneM2; j++) {
				if (matResCPU[i*mat.colonneM2 + j] != matResHost[i*mat.colonneM2 + j]) {
					esito = false;
					break;
				}
				else if (matResCPU[i*mat.colonneM2 + j] != matResHostSH[i*mat.colonneM2 + j]) {
					esito = false;
					break;
				}
				else if (matResHostSH[i*mat.colonneM2 + j] != matResHost[i*mat.colonneM2 + j]) {
					esito = false;
					break;
				}
			}
		}
		else
			break;
	}

	return esito;
}