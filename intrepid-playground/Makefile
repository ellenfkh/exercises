all: ContractDataFieldScalar ContractFieldFieldScalar ContractFieldFieldTensor ContractDataDataTensor

ContractDataDataTensor: ContractDataDataTensor.cu
	/opt/cuda/bin/nvcc -ccbin=g++ -m64 -O3 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -I/usr/local/cuda-6.5/include -I/clinic/2014/sandia14/trilinos-install//include -o ContractDataDataTensor ContractDataDataTensor.cu -Xcompiler -fopenmp -L/clinic/2014/sandia14/trilinos-install/lib -lkokkoscore -lhwloc -lgomp -lcudart -lcublas -lteuchoscore

ContractDataFieldScalar: ContractDataFieldScalar.cu
	/usr/local/cuda/bin/nvcc -ccbin=g++ -m64 -O3 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -I/usr/local/cuda-6.5/include -I/clinic/2014/sandia14/trilinos-install//include -o ContractDataFieldScalar ContractDataFieldScalar.cu -Xcompiler -fopenmp -L/clinic/2014/sandia14/trilinos-install/lib -lkokkoscore -lhwloc -lgomp -lcudart -lcublas

ContractFieldFieldScalar: ContractFieldFieldScalar.cu
	/usr/local/cuda/bin/nvcc -ccbin=g++ -m64 -O3 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -I/usr/local/cuda-6.5/include -I/clinic/2014/sandia14/trilinos-install//include -o ContractFieldFieldScalar ContractFieldFieldScalar.cu -Xcompiler -fopenmp -L/clinic/2014/sandia14/trilinos-install/lib -lkokkoscore -lhwloc -lgomp -lcudart -lcublas

ContractFieldFieldTensor: ContractFieldFieldTensor.cu
	/usr/local/cuda/bin/nvcc -ccbin=g++ -m64 -O3 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -I/usr/local/cuda-6.5/include -I/clinic/2014/sandia14/trilinos-install//include -o ContractFieldFieldTensor ContractFieldFieldTensor.cu -Xcompiler -fopenmp -L/clinic/2014/sandia14/trilinos-install/lib -lkokkoscore -lhwloc -lgomp -lcudart -lcublas -lteuchoscore

ContractDataDataScalar: ContractDataDataScalar.cu
	/usr/local/cuda/bin/nvcc -ccbin=g++ -m64 -O3 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -I/usr/local/cuda-6.5/include -I/clinic/2014/sandia14/trilinos-install//include -o ContractDataDataScalar ContractDataDataScalar.cu -Xcompiler -fopenmp -L/clinic/2014/sandia14/trilinos-install/lib -lkokkoscore -lhwloc -lgomp -lcudart -lcublas -lteuchoscore

clean:
	rm -f ContractFieldFieldScalar
	rm -f ContractDataDataScalar
	rm -f ContractFieldFieldTensor
	rm -f ContractDataFieldScalar
