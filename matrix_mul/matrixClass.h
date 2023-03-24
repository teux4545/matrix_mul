#pragma once

#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class Matrice {

	public:

		/* Dimensioni array multidimensionale (matrici)
		N.B.: il numero di colonne della prima matrice deve coincidere assolutamente con il numero di righe della seconda matrice  */

		// prima matrice (M1)
		const int righeM1 = 1440;
		const int colonneM1 = 1920;

		// seconda matrice (M2)
		const int righeM2 = 1920;
		const int colonneM2 = 1440;

		// dimensioni in bytes delle matrici M1, M2 e matrice dei risultati
		size_t dimM1 = (righeM1*colonneM1) * sizeof(int);
		size_t dimM2 = (righeM2*colonneM2) * sizeof(int);
		size_t dimRes = (righeM1*colonneM2) * sizeof(int);

};

class Blocco {

	public:

	// Dimensioni del blocco (x,y) impostate uguali in modo che formino blocchi quadrati esattamente di 1024 threads (limite imposto dall'hardware)
	const int BLKSIZE = 32;
	const int BLOCK = 16;

};

