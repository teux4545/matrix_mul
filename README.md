# Moltiplicazione tra matrici usando CPU e GPU

## Introduzione
Il seguente progetto ha come scopo quello di effettuare un'operazione matriciale quale prodotto di due matrici e mostrare infine le differenti tempistiche di elaborazione dei due dispositivi di calcolo presenti al giorno d'oggi in ogni computer.

Inizialmente il lavoro sarà affidato alla GPU (Graphic Processing Unit, processore grafico), sfruttando due diversi approcci, in un secondo momento invece la computazione verterà sulla CPU (Central Processing Unit, processore centrale).

Si tratta dunque di mettere in risalto i benefici di un calcolo eseguito in parallelo (GPU) piuttosto di uno eseguito in modo sequenziale (CPU) nel caso in esame.

## Specifiche tecniche (Hardware utilizzato)
<ul>
<li><b>CPU:</b> Intel i5-3337U (https://www.intel.it/content/www/it/it/products/sku/72055/intel-core-i53337u-processor-3m-cache-up-to-2-70-ghz/specifications.html)</li>

<li><b>GPU:</b> (sito non ufficiale - https://www.notebookcheck.it/NVIDIA-GeForce-710M.106543.0.html#:~:text=La%20NVIDIA%20GeForce%20710M%20%C3%A8,ha%20un%20clock%20notevolmente%20superiore.)</li>
</ul>

<b>Architettura 3rd genrazione intel (Ivy Bridge) (CPU):</b> 
<img src="https://github.com/teux4545/matrix_mul/blob/master/Ivy_Bridge_Architecture.png" width="642" height="443"></img><br>

<b>Architettura Fermi (GPU):</b>                            
<img src="https://github.com/teux4545/matrix_mul/blob/master/Fermi_Architecture.png" width="642" height="531"></img><br>

## Inizializzazione dei dati e gestione della memoria
È stata creato un file '_matrixClass_' contenente la classe Matrice, che specifica le caratteristiche di due matrici e una matrice risultato:
<ul>
    <li>Righe M1 & M2 e Mres</li>
    <li>Colonne M1 & M2 e Mres</li>
    <li>Dimesioni delle tre matrici</li>
</ul>
Ho scelto poi di inizializzare tutti i valori direttamente nelle classe

Il file ne contiene anche un'altra che inizializza la dimensione dei blocchi di thread.

Nel file '_kernel.cu_', la funzione <b>main</b> esegue tutte le allocazioni di memoria previste usando i tre metodi:
<ul>
    <li><b>cudaMallocHost: </b>alloca uno spazio di memoria nell'Host <i>(attenzione si riduce la dimensione di paging)</i>, questa memoria è page-locked, accessibile <b>direttamente</b> dal device (tempi più rapidi di accesso rispetto malloc)</li>
    <li><b>cudaMalloc: </b>alloca memoria sul Device</li>
    <li><b>malloc: </b>alloca memoria sull'Host</li>
</ul>

Precisazione doverosa è definire:
<ul>
    <li><b>Host: </b>riferito a CPU e alla sua memoria</li>
    <li><b>Device: </b>riferito a GPU e alla sua memoria</li>
</ul>

Tutti i metodi sono corredati di un controllo per eventuali eccezioni che si possono generare, trattate più avanti.

## Come lavora il kernel eseguito sulla GPU

Per operare sulla GPU è necessario innanzitutto predisporre una rappresentazione astratta dei thread che andranno effettivamente ad eseguire le operazioni di calcolo.
Essenzialmente i raggruppamenti avvengono su due livelli:
* <b>grid: </b> griglia ordinata di blocchi
* <b>block: </b>insieme ordinato di thread (per questa configurazione hardware il numero massimo di thread per blocco è 1024)

Il lancio del kernel, cioè la funzione eseguita sulla GPU, avviene come per una normale funzione in linguaggio C,<br>
seguita però poi da tre parentesi angolate '<b><<< ... , ... >>></b>' (caso base) che indicano la configurazione di thread utilizzata per quel kernel, infine troviamo gli argomenti passati alla funzione
```c++
matrix_mulGPU << <grid, block >> > (matriceGPU, matRGPU, matResGPU);
```
Esecuzione sul kernel:<br>
<img src="https://github.com/teux4545/matrix_mul/blob/master/kernel-execution-on-gpu.png" width="625" height="438"></img><br><br>
Le due funzioni di seguito come quella eseguita sulla CPU sono state prese e integrate da progetti github già disponibili, link in fondo al readme 

### Funzione eseguita utilizzando la Global Memory

```c++
__global__ void matrix_mulGPU(int *a, int *b, int *c) {

	Matrice mat;

	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int somma = 0;

	if (row < mat.righeM1 && col < mat.colonneM2) {
		for (int i = 0; i < mat.colonneM1; i++) {
			somma += a[row * mat.colonneM1 + i] * b[i * mat.colonneM2 + col];
			
			__syncthreads(); 
		}
		c[row * mat.colonneM2 + col] = somma;
	}
}
```
Ogni kernel CUDA inizia con la dichiarazione '<b>__global__</b>'.<br>
Si inizializzano poi le coordinate unidimensionali x,y di tutti i thread così che, dal loro indice sapremo che i dati nelle matrici associati a quelle specifiche posizioni verrano elaborati da quegli specifici thread.<br><br>
Tutti i thread lavorano all'unisono, quasi contemporaneamente, il lavoro dunque sarà distribuito. Nel 'for' si prende uan riga e si iterano lungo le colonne (viceversa per la seconda), ogni thread si occuperà di lavorare con il dato della matrice in cui si identifica, è buona norma mettere una barriera di sincronizzazione per le operazioni svolte, di questo si occuperà '<b>__syncthreads()</b>' che permetterà a tutti i thread di terminare il lavoro nello stesso istante.
<br><br>NB: gli array bidimensionali (matrici) sono stati direttamente creati/convertiti a una dimensione, pur mantenendo un'organizzazione righe/colonne.<br>
Ad esempio: per la <i>matriceHost</i>, che poi sarà <i>matriceGPU</i> (M1), i primi x elementi corrispondono alla riga 0 con x=colonneM1, i successivi x elementi saranno della riga 1, dunque fin qui si avranno 2*x elementi e così via fino ad ottenere n*x elementi (con n=righeM1), infatti la dimensione dell'array sarà righeM1*colonneM1, dimensione della matrice 1. Stessa cosa per M2. 

### Funzione eseguita utilizzando la Shared Memory

```c++
__global__ void matrix_mulGPUShared(int *a, int *b, int *c) {

	Matrice mat;

	__shared__ int sA[BLK][BLK];   // sA e sB usano blocchi da 16*16 = 256 thread
	__shared__ int sB[BLK][BLK];

	int row = blockDim.y * blockIdx.y + threadIdx.y;
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	
	int somma = 0;
	sA[threadIdx.y][threadIdx.x] = 0;
	sB[threadIdx.y][threadIdx.x] = 0;
		
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
		
		__syncthreads(); 

		for (int j = 0; j < BLK; ++j) {
			somma += sA[threadIdx.y][j] * sB[j][threadIdx.x];
			__syncthreads();
		}
	}
	if (row < mat.righeM1 && col < mat.colonneM2) {
		c[row * mat.colonneM2 + col] = somma;
	}
}
```
"...La memoria condivisa viene allocata per blocco di thread, quindi tutti i thread del blocco hanno accesso alla stessa memoria condivisa. I thread possono accedere ai dati nella memoria condivisa caricati dalla memoria globale da altri thread all'interno dello stesso blocco di thread. ..." - developer.nvidia.com<br>
<br>
In esecuzione viene evidenziato come, utilizzando questo metodo, i tempi di calcolo sono notevolmente ridotti essendo le variabili '<b>__shared__</b>', caricate su una memoria 'on chip' (cache) più vicina all'unità di calcolo, infatti è buona norma non caricare questa memoria con troppi dati altrimenti si perderebbe in prestazioni.
<br>Vedere link in fondo (https://leimao.github.io/downloads/...) dove è spiegata anche l'utilità di syncthreads() che preveiene un hazard dei dati.
		
## Calcolo sulla CPU

```c++
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
```
La seguente funzione ha caratteristiche sequenziali, in questo caso è un solo chip, la CPU precisamente, a svolgere tutte le operazioni di calcolo.<br>
Lo svantaggio della sequenzialità è che si è costretti ad accedere ad un singolo elemento per volta ed elaborarlo, mentre sulla GPU i chip di calcolo accedono al singolo elemento ma la loro computazione e limitata ad esso.<br>(link in fondo con dominio ecatue.gitlab.io)
	
## Durata delle operazioni

Per il calcolo della durata delle operazioni è stata usata la combinazione <b>cudaEventRecord()</b> e <b>cudaEventSynchronize()</b> per i kernel, mentre <b>clock()</b> per operazioni sulla CPU

## Controllo dei risultati

```c++
bool checkRes(int *matResCPU, int *matResHost, int *matResHostSH) {

	Matrice mat;
	bool esito = true;

	for (int i = 0; i < mat.righeM1; i++) {
		if (esito) {
			for (int j = 0; j < mat.colonneM2; j++) {
				if (matResCPU[i*mat.colonneM2 + j] != matResHost[i*mat.colonneM2 + j] || 
					matResCPU[i*mat.colonneM2 + j] != matResHostSH[i*mat.colonneM2 + j] || 
					matResHostSH[i*mat.colonneM2 + j] != matResHost[i*mat.colonneM2 + j]) {
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
```

Viene effettuato uno scorrimento lungo tutti gli elementi delle tre matrici dei risultati, quando l'if presente all'interno del 'for' annidato trova un valore qualsiasi diverso, confrontando i tre array, viene settata la variabile 'esito' a <i>false</i>, e si interrompe il ciclo.<br> Uscendo e proseguendo l'iterazione con il for più esterno, l'istruzione di controllo <i>if(esito)</i> a questo punto non vale più e si passa al caso <i>else</i> che interrompe anche il for esterno.

## Gestione delle eccezioni

```c++
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
```
Gli errori durante la programmazione CUDA possono essere di due tipi, sincroni e asincroni.<br>
In tutto il progetto troviamo chiamate alla macro <b>cudaCheckErrors</b> che riportano errori a runtime sincroni, cioè che nel momento stesso in cui si presentano la macro interviene per gestirli e li riporta (errore non-sticky, cioè non permanente, dunque recuperabile: le successive chiamate delle API CUDA si comportano normalmente).<br><br>
Se dovesse presentarsi un errore asincrono, (anche sticky, cioè non recuperabile) cioè distante temporalmente dalla chiamata del kernel, errore come un accesso ad un indirizzo di memoria non valido, quindi solo durante l'esecuzione del kernel, la macro, che controlla gli errori sincroni, non registrerebbe l'errore perché al momento del lancio del kernel (operazione sincrona), non avrebbe riscontrato nulla di anomalo.<br><br>
In questo particolare occasione è buona norma chiamare un'altra volta la macro subito dopo un evento di sincronizzazione ad esempio <b>cudaDeviceSynchronize</b> o come in questo caso <b>cudaEventSynchronize</b>.<br<br>

Nel progetto si trovano anche altre condizioni di controllo come:
* controllo delle dimesioni delle matrici
* controllo sul TDR (Time Detection and Recovery delay)
   - questo in particolare è un'impostazione di sistema che interrompe il lavoro che sta svolgendo la GPU e la resetta a parametri di default invalidando tutta l'elaborazione
* dimensione delle matrici definite nella shared memory ('Funzioni_GPU.h') è diversa da quella impostata per i blocchi nella shared memory nel file 'matrixClass'

## Fonti e pagine web utili al progetto

- https://docs.nvidia.com/cuda/cuda-runtime-api/
- https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/
- https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/#:~:text=Shared%20memory%20is%20a%20powerful,mechanism%20for%20threads%20to%20cooperate.
- http://gpu.di.unimi.it/slides/lezione2.pdf
- https://ecatue.gitlab.io/gpu2018/pages/Cookbook/matrix_multiplication_cuda.html
- https://leimao.github.io/blog/Proper-CUDA-Error-Checking/
- https://leimao.github.io/downloads/blog/2022-07-04-CUDA-Shared-Memory-Capacity/02-CUDA-Shared-Memory.pdf
<br><br>Funzioni:
- (funzione global memory e su cpu) https://github.com/fbasatemur/CUDA-Matrix/tree/master/
- (funzione shared memory) https://gist.github.com/raytroop/120e2d175d95f82edbee436374293420
- (per il controllo degli errori) https://stackoverflow.com/questions/12924155/sending-3d-array-to-cuda-kernel/12925014#12925014

## Autore

- <b>Ciucciovè Leonardo</b>
