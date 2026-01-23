\## Kompilacja



Projekt wykorzystuje system CMake oraz działa na systemie Windows. W terminalu należy wykonać:



```bash

cmake -S . -B build

cmake --build . --config Release

```



\## Uruchomienie



Program przyjmuje argumenty z linii poleceń określające tryb renderowania oraz plik sceny.



\*\*Składnia:\*\*



```bash

./CSGRayCast \[cpu|gpu] \[plik\_sceny]

```



\*\*Przykłady:\*\*



```bash

./CSGRayCast gpu helix\_complex.txt

./CSGRayCast cpu industrial\_complex.txt



```



\## Format Plików



Sceny definiowane są w plikach tekstowych `.txt`. Parser przetwarza plik linia po linii w sposób rekurencyjny (Pre-order traversal). Wcięcia (indentacja) stosowane są dla czytelności kodu, ale logiczna struktura wynika z kolejności definicji.



Każda linia rozpoczyna się od nazwy typu węzła (`union`, `intersection`, `difference` lub nazwa prymitywu (kształtu)), po której następują parametry oddzielone spacjami.



\### 1. Operacje Logiczne (Węzły)



Operacje CSG przyjmują \*\*dwa argumenty\*\* (lewe i prawe dziecko), które są definiowane w kolejnych liniach po operacji.



\* `union` - Suma zbiorów.

\* `intersection` - Część wspólna.

\* `difference` - Różnica |(odejmuje drugi kształt od pierwszego).



\### 2. Prymitywy (Liście)



Każdy prymityw definiowany jest przez parametry geometryczne, a następnie \*\*6 parametrów materiału\*\*.



\*\*Wspólne parametry materiału (na końcu każdej linii):\*\*

`r g b diff spec shin`



\* `r, g, b`: Składowe koloru (0.0 - 1.0).

\* `diff`: Współczynnik rozproszenia (diffuse).

\* `spec`: Współczynnik odbicia lustrzanego (specular).

\* `shin`: Połyskliwość (shininess exponent).



\*\*Typy prymitywów i ich parametry geometryczne:\*\*



\* \*\*Kula (`sphere`):\*\*

```text

sphere x y z radius \[materiał]



```





\* `x, y, z`: Środek kuli.

\* `radius`: Promień.





\* \*\*Prostopadłościan (`cuboid`):\*\*

```text

cuboid x y z w h d \[materiał]



```





\* `x, y, z`: Współrzędne minimalnego narożnika (min\_pt).

\* `w, h, d`: Szerokość, wysokość, głębokość (wymiary wzdłuż osi X, Y, Z).





\* \*\*Walec (`cylinder`):\*\*

```text

cylinder x y z radius height \[materiał]



```





\* `x, y, z`: Środek podstawy dolnej.

\* `radius`: Promień podstawy.

\* `height`: Wysokość (wzdłuż osi Y).





\* \*\*Stożek (`cone`):\*\*

```text

cone x y z radius height \[materiał]



```





\* `x, y, z`: Środek podstawy dolnej.

\* `radius`: Promień podstawy.

\* `height`: Wysokość (wzdłuż osi Y, wierzchołek znajduje się w `y + height`).







\### Przykład definicji sceny



```text

difference

&nbsp; sphere 0.0 0.0 0.0 1.4 1.0 0.2 0.2 0.8 0.6 64

&nbsp; cuboid -1.1 -1.1 -1.1 2.2 2.2 2.2 0.2 0.2 1.0 0.8 0.5 32



```



\*Powyższy kod definiuje kulę, od której odejmowany jest prostopadłościan (sześcian).\*





\## Sterowanie



Aplikacja umożliwia interaktywną zmianę pozycji kamery oraz źródła światła:



\* \*\*Strzałki (Góra/Dół/Lewo/Prawo):\*\* Obrót kamery wokół punktu skupienia (orbitowanie).

\* \*\*Klawisze W/S/A/D:\*\* Ruch źródła światła.

\* \*\*Zamknięcie okna:\*\* Zakończenie programu.



\## Dodatkowe Kształty

Program obsługuje 4 kształty:



* Sfera
* Prostopadłościan
* Cylinder
* Stożek



---



\## Opis Techniczny i Zasada Działania



\### Algorytm Ray Tracingu CSG



1\. Każdy prymityw (np. Kula, Sześcian) zwraca listę odcinków (`Spans`), w których promień przebywa "wewnątrz" bryły. Odcinek składa się z czasu wejścia () i wyjścia () oraz wektorów normalnych w tych punktach.





2\. Operacje logiczne łączą te odcinki:

\* \*\*Union ():\*\* Scalanie nakładających się odcinków.

\* \*\*Intersection ():\*\* Znalezienie części wspólnej odcinków.

\* \*\*Difference ():\*\* Wycięcie odcinków obiektu  z odcinków obiektu .







\### Reprezentacja Drzewa CSG (Flattening)



Drzewa binarne oparte na wskaźnikach są nieefektywne na GPU. Projekt wykorzystuje strukturę `FlatCSGTree`, która spłaszcza drzewo do tablic liniowych (SoA - Structure of Arrays):



\* Węzły drzewa są przechowywane w tablicy `nodes`.

\* Relacje rodzic-dziecko są reprezentowane przez indeksy w tablicach `left\_indexes` i `right\_indexes`.

\* Dane geometryczne (promienie, wymiary) są oddzielone od topologii i skompaktowane tylko dla węzłów liści (prymitywów), co oszczędza pamięć.







Drzewo jest przetwarzane w porządku \*\*Post-Order\*\* (od dołu do góry), co pozwala na ewaluację przy użyciu stosu.



\### Implementacja GPU (CUDA)



\#### Zarządzanie Pamięcią (Stack-less dynamic allocation)



Największym wyzwaniem w CSG na GPU jest nieznana z góry liczba przedziałów, które wygeneruje promień. Dynamiczna alokacja (`malloc`/`new`) wewnątrz kernela jest bardzo wolna.

Rozwiązanie zastosowane w projekcie:



1\. \*\*Analiza wstępna (Host):\*\* Przed uruchomieniem renderowania, funkcja `computeTotalSpanUsage` symuluje przejście przez drzewo na CPU, przy każdym nodzie szacując ilość Span z góry, obliczając maksymalny potrzebny rozmiar stosu i bufora odcinków (`max\_pool\_size`, `max\_stack\_depth`).



2\. \*\*Globalny Bufor (Device):\*\* Alokowany jest jeden duży blok pamięci globalnej, podzielony dla każdego piksela/wątku.



3\. \*\*Kernel:\*\* Każdy wątek otrzymuje wskaźnik do własnego fragmentu pamięci SoA: `StridedSpan`, `StridedStack`. Działając bez blokad i bez dynamicznej alokacji.







\#### Pamięć Współdzielona (Shared Memory)



Aby przyspieszyć dostęp do struktury drzewa, cała topologia (`nodes`, indeksy) oraz dane prymitywów są kopiowane do \*\*Shared Memory\*\* na początku działania bloku wątków. Dzięki temu wszystkie wątki w bloku mają błyskawiczny dostęp do definicji sceny, co drastycznie redukuje opóźnienia pamięci globalnej.



\### Parsowanie Plików



Sceny są wczytywane z plików tekstowych. Parser rekurencyjnie buduje strukturę drzewa.



\## Struktura Plików



* `main.cu`: Punkt wejścia, pętla główna, obsługa SDL, uruchamianie kerneli.





* `tracer.cu`: Logika renderowania, implementacja operacji CSG (Union, Intersection, Difference) na odcinkach, kernel CUDA.





* `shape.h`: Definicje analityczne prymitywów (Sphere, Cuboid, Cylinder, Cone) i obliczanie przecięć promieni.





* `csg.h` / `loadfile.cpp`: Definicje struktur danych drzewa i parser plików sceny.





* `rayCast.h`: Podstawowe struktury matematyczne (Wektor, Promień, Kamera, Kolor).
