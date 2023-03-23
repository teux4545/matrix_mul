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
È stata creato un file 'matrixClass' contenente la classe Matrice, che specifica le caratteristiche di due matrici e una matrice risultato:
<ul>
    <li>Righe M1 & M2 e Mres</li>
    <li>Colonne M1 & M2 e Mres</li>
    <li>Dimesioni delle tre matrici</li>
</ul>
Si è scelto poi di inizializzare tutti i valori direttamente nelle classe

Il file ne contiene anche un'altra che inizializza la dimensione dei blocchi di thread.

Nel file 'kernel.cu', la funzione <i>main</i> dunque esegue tutte le allocazioni di memoria previste usando i tre metodi:
<ul>
    <li><b>cudaMallocHost: </b>alloca uno spazio di memoria in bytes nell'Host (attenzione si riduce la dimensione di paging), questa memoria è page-locked, accessibile <u>direttamente</u> dal device (tempi più rapidi di accesso rispetto malloc)</li>
    <li><b>cudaMalloc: </b>alloca memoria sul Device</li>
    <li><b>malloc: </b>alloca memoria sull'Host</li>
</ul>

Precisazione doverosa è definire:
<ul>
    <li>Host: riferito a CPU e alla sua memoria</li>
    <li>Device: riferito a GPU e alla sua memoria</li>
</ul>

Tutti i metodi sono corredati di un controllo per eventuali eccezioni che si possono generare, trattate più avanti.

## Come lavora il kernel eseguito sulla GPU

## Calcolo sulla CPU

## Durata delle operazioni

## Controllo dei risultati

## Gestione delle eccezioni

## Fonti e pagine web utili al progetto
Fonti documentazione:  https://docs.nvidia.com/cuda/cuda-runtime-api/
## Autore
- <b>Ciucciovè Leonardo</b>
