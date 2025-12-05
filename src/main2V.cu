#include <iostream>
#include <fstream>
#include <vector>
#include <stdio.h>
#include <curand_kernel.h>

#include "opti_pwlnn.cu"



void printArray(int* a, int len){
	for(int i = 0; i < len; i++){
		std::cout << a[i] << std::endl;
	}
}


int main(void) {

	// les couches
	int dCPU[Sizedld];
	dCPU[0] = 2; // d_0 input layer  
	dCPU[1] = 4; // d_1 
	dCPU[2] = 4; // d_2
	dCPU[3] = 4; // d_3
	dCPU[4] = 1; // d_4 output layer

	// size of the weight and bias vectors
	//
	//	  1 biais    |
	//				 |	1
	//   1(1 poids)  |
	//				 |	2
	//   
	//   				3
	// 	 2(1 poids)
	// 					4

	//			3*4 + 5*4 + 5*4 + 5*1 = 57

	// (nb_weights+nb_biases)[i] * (nb_neurones)[i+1]
	// nb_connections_layers+bias = nn_i * nn_i+1 + nn_i+1
	
	int sizeWB = (dCPU[0] + 1) * dCPU[1] +
				 (dCPU[1] + 1) * dCPU[2] +
				 (dCPU[2] + 1) * dCPU[3] +
				 (dCPU[3] + 1) * dCPU[4];

	int* MinMax;
	cudaMallocManaged(&MinMax, 3*sizeof(int));
	MinMax[0] =  5000000;
	MinMax[1] = -5000000;
	MinMax[2] =  0;

	int nop = 16 * 16 * 16; // Number of subpolytopes 4096 default

	// TODO why the times two TODO didnt understand here 
	int deltaSize = (2 * dCPU[0] + dCPU[1] + dCPU[2] + dCPU[3] + dCPU[4]);
	int sizeB     = (2 * dCPU[0] + dCPU[1] + dCPU[2] + dCPU[3] + dCPU[4]);
	int sizeCB    = (2 * dCPU[0] + dCPU[1] + dCPU[2] + dCPU[3] + dCPU[4]) * (1 + dCPU[0]);

	for (int k = 1; k < 1 + dCPU[0]; k++) {
		sizeCB += (deltaSize - k) * (1 + dCPU[0] - k);
		sizeB  += (deltaSize - k);
	}

	dCPU[MaxDepth]     = sizeCB - deltaSize * (1 + dCPU[0]); // Values should be stored at the end Algo (4.4) deltaC
	dCPU[MaxDepth + 1] = sizeCB; // The total size needed for each configuration s DeltaC
	dCPU[MaxDepth + 3] = sizeB - deltaSize; // Values should be stored at the end Algo (4.4) deltaB
	dCPU[MaxDepth + 4] = sizeB; // The total size needed for each configuration s DeltaB

	// maximum number of vertices by subpolytope: d0 combinations among m-d0
	int NbVertices = 1000;

	dCPU[MaxDepth + 2] = NbVertices; // The total size needed for each configuration s

	printArray(dCPU+MaxDepth, 5);
	
	cudaMemcpyToSymbol(dld, dCPU, Sizedld * sizeof(int), 0, cudaMemcpyHostToDevice);
	
	// C is C matrix and beta, levels: list of levels, Ver contains the list of vertices,
	// R contains the isometry matrices, q contains the translation values
	float* C, * Ccpu, * levels, * levelscpu, * Ver, * Vercpu, * R, *q;
	int* num, * numcpu; // contains the true number of vertices and levels

	testCUDA(cudaMalloc(&C, sizeCB * nop * sizeof(float)));
	Ccpu = (float*)malloc(  sizeCB * nop * sizeof(float));

	testCUDA(cudaMalloc(&levels, 2 * NbVertices * nop * sizeof(float))); // twice the size to be able to have a sorted list
	levelscpu = (float*)malloc(      NbVertices * nop * sizeof(float));
	testCUDA(cudaMalloc(&Ver,    NbVertices * nop * dCPU[0] * sizeof(float)));
	Vercpu = (float*)malloc(     NbVertices * nop * dCPU[0] * sizeof(float));

	// TODO how those these work
	int siV  = (3 + 2); // binding index for volume V
	int siVD = (2);     // binding index for volume V TODO what are the differnces between them
	int siR  = (4);     // binding index for volume R
	int siRD = 0;       // binding index for volume R

	// TODO instead of doing 4 layers do 3 then add it 
	testCUDA(cudaMalloc(&R, siR * nop * sizeof(float)));
	testCUDA(cudaMalloc(&num, 2 * nop * sizeof(int)));
	numcpu = (int*)malloc(    2 * nop * sizeof(float));

	testCUDA(cudaMalloc(&q, nop * dCPU[0] * sizeof(float)));
	
	float* WBGPU; // Coefficents of matrices and bias vectors

	testCUDA(cudaMalloc(&WBGPU, sizeWB * sizeof(float)));

	std::string path = "weights.txt";  // path of the file with one value per row

	// Reading the file
	std::ifstream file(path);
	if (!file.is_open()) {
		std::cerr << "Error: impossible to open " << path << std::endl;
		return 1;
	}

	std::vector<float> WB_data;
	WB_data.reserve(sizeWB);

	float value;
	while (file >> value) {
		WB_data.push_back(value);
	}
	file.close();

	if (WB_data.size() != sizeWB) {
		std::cerr << "Error: " << WB_data.size() << " read values " << sizeWB << std::endl;
		return 1;
	}

	std::cout << "read done of weights: " << WB_data.size() << " floats." << std::endl;

	testCUDA(cudaMemcpy(WBGPU, WB_data.data(), sizeWB * sizeof(float), cudaMemcpyHostToDevice));

	int low = 0;  // the starting index
	int up = nop; // the ending index
	int nbN = 16; // Number of neurones TODO  we should have 15 no ? in the 3d we should have 16
	
	testCUDA(cudaMemset(C, 0, nop * sizeCB * sizeof(float)));

	float timer;
	cudaEvent_t start, stop;			// GPU timer instructions
	cudaEventCreate(&start);			// GPU timer instructions
	cudaEventCreate(&stop);				// GPU timer instructions
	cudaEventRecord(start, 0);			// GPU timer instructions

	size_t currentLimit;
	cudaDeviceGetLimit(&currentLimit, cudaLimitStackSize);
	printf("Current CUDA stack size: %zu bytes\n", currentLimit);

	size_t NcurrentLimit =  64 * currentLimit;
	cudaDeviceSetLimit(cudaLimitStackSize, NcurrentLimit);
	cudaDeviceGetLimit(&currentLimit, cudaLimitStackSize);
	printf("Current CUDA stack size: %zu bytes\n", currentLimit);

	testCUDA(cudaMemset(levels , 0, 2 * NbVertices * nop * sizeof(float)));
	testCUDA(cudaMemset(Ver, 0, 	NbVertices * nop * dCPU[0] * sizeof(float)));
	testCUDA(cudaMemset(num, 0, 2 * 			 nop * sizeof(float)));

	//        TODO     TODO      why + 1         
	Part_k <<<16 * 8, 16 * 2 * (dCPU[0] + 1), (sizeWB + 16 * 2 * max(nbN * 4, 15 * dCPU[0])) * sizeof(float) >>>
	// levL_k <<<16 * 8, 16 * 2 * (dCPU[0] + 1), (sizeWB + 16 * 2 * max(nbN * 4, 15 * dCPU[0])) * sizeof(float) >>>
	// onlyPartition_k <<<16 * 8, 16 * 2 * (dCPU[0] + 1), (sizeWB + 16 * 2 * max(nbN * 4, 15 * dCPU[0])) * sizeof(float) >>>
			(WBGPU, C, levels, 2 * dCPU[0], sizeWB, 
				low, up, R, q, Ver, num, nbN, 4, 
				siV, siVD, siR, siRD, MinMax);
		
	cudaDeviceSynchronize();
	testCUDA(cudaMemcpy(Ccpu  , C  , nop * sizeCB * sizeof(float)              , cudaMemcpyDeviceToHost));
	testCUDA(cudaMemcpy(levelscpu , levels , NbVertices * nop * sizeof(float)          , cudaMemcpyDeviceToHost));
	testCUDA(cudaMemcpy(Vercpu, Ver, NbVertices * nop * dCPU[0] * sizeof(float), cudaMemcpyDeviceToHost));
	testCUDA(cudaMemcpy(numcpu, num, 2 * nop * sizeof(float)                   , cudaMemcpyDeviceToHost));


	printf("The computed Minimum level %f\n", (float)0.0000001f * MinMax[0]);
	printf("The computed Maximum level %f\n", (float)0.0000001f * MinMax[1]);
	printf("Number of levels %i\n", MinMax[2]);
	printf("With notation (index of pol, number of vertices, number of levels), the non-empty polytopes are:\n");
	int count = 0;
	for (int k = 0; k < nop; k++) {
		if (numcpu[2 * k] > 0) {
			printf("(%d, %d, %d), ", k, numcpu[2 * k], numcpu[2 * k + 1]);
			count++;
		}
	}
	printf("\n");
	printf("The number of non-empty polytopes: %d\n", count);

	cudaEventRecord(stop, 0);			// GPU timer instructions
	cudaEventSynchronize(stop);			// GPU timer instructions
	cudaEventElapsedTime(&timer,			// GPU timer instructions
		start, stop);					// GPU timer instructions
	cudaEventDestroy(start);			// GPU timer instructions
	cudaEventDestroy(stop);				// GPU timer instructions

	printf("Execution time %f ms\n", timer);

	testCUDA(cudaFree(MinMax));
	testCUDA(cudaFree(C));
	free(Ccpu);
	testCUDA(cudaFree(levels));
	free(levelscpu);
	testCUDA(cudaFree(Ver));
	free(Vercpu);
	testCUDA(cudaFree(R));
	testCUDA(cudaFree(q));
	testCUDA(cudaFree(num));
	free(numcpu);
	testCUDA(cudaFree(WBGPU));

	return 0;
}