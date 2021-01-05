# jsa
for the collection of Static Features: clang-3.8 is required
$ clang-3.8 -Xclang -load -Xclang ./libStaticFeatureExtPass.so program-code.c 

Note: Bechmarks used- dhrystone,whetstone,polybench,mibench

for the collection of Dynamic Feaures: Odroid-XU4 is required
$ ./counter-collection

for the measurement of power dissipation: SmartPower2 is required
$ ./power-collection

Note: In order to synchornize execution of benchmarks and power measurement, both the above scripts must be run parallelly. 

for the traning of ML-model: python installation is required
$ python traning-testing.py

for the cross-validation of ML-model: 
$ python cross-validation.py

