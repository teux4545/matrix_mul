# Moltiplicazione tra matrici usando CPU e GPU

## Introduzione
Il seguente progetto ha come scopo quello di effettuare un operazione matriciale, quale prodotto di due matrici e mostrare infine le differenti tempistiche di elaborazione dei due dispositivi di calcolo presenti al giorno d'oggi in ogni computer.

Inizialmente il lavoro sarà affidato alla GPU (Graphic Processing Unit, processore grafico), sfruttando due diversi approcci, in un secondo momento invece la computazione verterà sulla CPU (Central Processing Unit, processore centrale).

Si tratta dunque di mettere in risalto i benefici di un calcolo eseguito in parallelo (GPU) piuttosto di uno eseguito in modo sequenziale (CPU) nel caso in esame.

## Specifiche tecniche (Hardware utilizzato)
<ul>
<li><b>CPU:</b> Intel i5-3337U (https://www.intel.it/content/www/it/it/products/sku/72055/intel-core-i53337u-processor-3m-cache-up-to-2-70-ghz/specifications.html)</li>

<li><b>GPU:</b> (sito non ufficiale - https://www.notebookcheck.it/NVIDIA-GeForce-710M.106543.0.html#:~:text=La%20NVIDIA%20GeForce%20710M%20%C3%A8,ha%20un%20clock%20notevolmente%20superiore.)</li>
</ul>

<b>Architettura 3rd genrazione intel (Ivy Bridge) (CPU):</b> 
<img src="https://github.com/teux4545/matrix_mul/blob/master/Ivy_Bridge_Architecture.webp"></img><br>

<b>Architettura Fermi (GPU):</b>
<img src="https://github.com/teux4545/matrix_mul/blob/master/Fermi_Architecture.png"></img><br>

## Inizializzazione dei dati e gestione della memoria
È stata creato un file '_matrixClass_' contenente la classe Matrice, che specifica le caratteristiche di due matrici e una matrice risultato:
<ul>
    <li>Righe M1 & M2 e Mres</li>
    <li>Colonne M1 & M2 e Mres</li>
    <li>Dimesioni delle tre matrici</li>
</ul>
Si è scelto poi di inizializzare tutti i valori direttamente nelle classe

Il file ne contiene anche un'altra che inizializza la dimensione dei blocchi di thread.

Nel file '_kernel.cu_', la funzione <b>main</b> dunque, esegue tutte le allocazioni di memoria previste usando i tre metodi:
<ul>
    <li><b>cudaMallocHost: </b>alloca uno spazio di memoria in bytes nell'Host ***(attenzione si riduce la dimensione di paging)***, questa memoria è page-locked, accessibile <b>direttamente</b> dal device (tempi più rapidi di accesso rispetto malloc)</li>
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

Le operazioni, ricapitolate, effettuate dunque per compilare ed eseguire il programma che andrà a sfruttare la GPU sono le seguenti:
1.  Setup dei dati su host (Allocazione e successiva inizializzazione)
2.  Alloca memoria per i dati sulla GPU  
3.  Alloca memoria per output su host 
4.  Alloca memoria per output su GPU 
5.  Copia i dati da host a GPU
6.  <b>Lancia il kernel su GPU</b>
7.  Copia output da GPU a host 
8.  Cancella le memorie

L'attenzione ora ricade sul sesto punto.
Affinché la
## Calcolo sulla CPU

## Durata delle operazioni

## Controllo dei risultati

## Gestione delle eccezioni
```c++
// Controllo errori cuda
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true){}

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s(%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
               fprintf(stderr, "\n *** FAILED - ABORTING\n"); \
            system("pause");\
            return 1; \
        } \
    } while (0)
```
## Fonti e pagine web utili al progetto
Fonti documentazione:  https://docs.nvidia.com/cuda/cuda-runtime-api/
## Autore
- <b>Ciucciovè Leonardo</b>
