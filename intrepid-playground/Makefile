all: IntrepidContractionKokkos ContractFieldFieldScalar ContractDataDataScalar

IntrepidContractionKokkos: IntrepidContractionKokkos.cu
	/opt/cuda/bin/nvcc -ccbin=g++ -m64 -O3 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -I/usr/local/cuda-6.5/include -I/clinic/2014/sandia14/trilinos-install-shadowfax//include -o IntrepidContractionKokkos IntrepidContractionKokkos.cu -Xcompiler -fopenmp -L/clinic/2014/sandia14/trilinos-install-shadowfax/lib -lkokkoscore -lhwloc -lgomp -lcudart -lcublas

ContractFieldFieldScalar: ContractFieldFieldScalar.cu
	/opt/cuda/bin/nvcc -ccbin=g++ -m64 -O3 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -I/usr/local/cuda-6.5/include -I/clinic/2014/sandia14/trilinos-install//include -o ContractFieldFieldScalar ContractFieldFieldScalar.cu -Xcompiler -fopenmp -L/clinic/2014/sandia14/trilinos-install/lib -lkokkoscore -lhwloc -lgomp -lcudart -lcublas -lteuchoscore

ContractDataDataScalar: ContractDataDataScalar.cu
	/opt/cuda/bin/nvcc -ccbin=g++ -m64 -O3 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -I/usr/local/cuda-6.5/include -I/clinic/2014/sandia14/trilinos-install//include -o ContractDataDataScalar ContractDataDataScalar.cu -Xcompiler -fopenmp -L/clinic/2014/sandia14/trilinos-install/lib -lkokkoscore -lhwloc -lgomp -lcudart -lcublas -lteuchoscore

clean:
	rm -f IntrepidContractionKokkos
	rm -f IntrepidContraction
	rm -f IntrepidContraction3x3
	rm -f ContractDataDataScalar
	rm -f ContractFieldFieldScalar
