#include <iostream>
#include <fstream>
#include <vector>
#include <stdio.h>
#include <curand_kernel.h>

#include "opti_pwlnn.cu"


int main(void) {

	int dCPU[Sizedld];
	dCPU[0] = 3; // d_0
	dCPU[1] = 4; // d_1
	dCPU[2] = 4; // d_2
	dCPU[3] = 4; // d_3
	dCPU[4] = 1; // d_4

	int sizeWB = (dCPU[0] + 1) * dCPU[1] +
				 (dCPU[1] + 1) * dCPU[2] +
				 (dCPU[2] + 1) * dCPU[3] +
				 (dCPU[3] + 1) * dCPU[4];

	int* MinMax;
	cudaMallocManaged(&MinMax, 3*sizeof(int));
	MinMax[0] =  5000000;
	MinMax[1] = -5000000;
	MinMax[2] =  0;

	int nop = 16 * 16 * 16; // Number of subpolytopes

	int deltaSize = (2 * dCPU[0] + dCPU[1] + dCPU[2] + dCPU[3] + dCPU[4]);
	int sizeB     = (2 * dCPU[0] + dCPU[1] + dCPU[2] + dCPU[3] + dCPU[4]);
	int sizeCB    = (2 * dCPU[0] + dCPU[1] + dCPU[2] + dCPU[3] + dCPU[4]) * (1 + dCPU[0]);

	for (int k = 1; k < 1 + dCPU[0]; k++) {
		sizeCB += (deltaSize - k) * (1 + dCPU[0] - k);
		sizeB  += (deltaSize - k);
	}

	dCPU[MaxDepth]     = sizeCB - deltaSize * (1 + dCPU[0]); // Values should be stored at the end Algo (4.4) deltaC
	dCPU[MaxDepth + 1] = sizeCB;  // The total size needed for each configuration s DeltaC
	dCPU[MaxDepth + 3] = sizeB - deltaSize; // Values should be stored at the end Algo (4.4) deltaB
	dCPU[MaxDepth + 4] = sizeB; // The total size needed for each configuration s DeltaB

	// maximum number of vertices by subpolytope: d0 combinations among m-d0
	int NbVertices = 1000;

	dCPU[MaxDepth + 2] = NbVertices; // The total size needed for each configuration s
	
	cudaMemcpyToSymbol(dld, dCPU, Sizedld * sizeof(int), 0, cudaMemcpyHostToDevice);

	float* C, * Ccpu, * LL, * LLcpu, * Ver, * Vercpu, * R, *q;
	int* num, * numcpu; // contains the true number of vertices and levels

	testCUDA(cudaMalloc(&C, sizeCB * nop * sizeof(float)));
	Ccpu = (float*)malloc(sizeCB * nop * sizeof(float));

	testCUDA(cudaMalloc(&LL, 2 * NbVertices * nop * sizeof(float))); // twice the size to be able to have a sorted list
	LLcpu = (float*)malloc(      NbVertices * nop * sizeof(float));
	testCUDA(cudaMalloc(&Ver,    NbVertices * nop * dCPU[0] * sizeof(float)));
	Vercpu = (float*)malloc(     NbVertices * nop * dCPU[0] * sizeof(float));

	int siV  = (4 + 3 + 2);
	int siVD = (3 + 2);
	int siR  = (9 + 4);
	int siRD = (4);

	testCUDA(cudaMalloc(&R, siR * nop * sizeof(float)));
	testCUDA(cudaMalloc(&num, 2 * nop * sizeof(int)));
	numcpu = (int*)malloc(    2 * nop * sizeof(float));

	testCUDA(cudaMalloc(&q, nop * dCPU[0] * sizeof(float)));
	
	float* WBGPU;

	testCUDA(cudaMalloc(&WBGPU, sizeWB * sizeof(float)));

	std::string path = "weights.txt";  // chemin du fichier avec une valeur par ligne

	// Lecture du fichier sur CPU
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

	int low = 0;
	int up = nop;
	int nbN = 16; // Number of neurones;
	
	testCUDA(cudaMemset(C, 0, nop * sizeCB * sizeof(float)));

	float Tim;
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

	testCUDA(cudaMemset(LL , 0, 2 * NbVertices * nop * sizeof(float)));
	testCUDA(cudaMemset(Ver, 0,     NbVertices * nop * dCPU[0] * sizeof(float)));
	testCUDA(cudaMemset(num, 0, 2 *              nop * sizeof(float)));

	Part_k << <16 * 8, 16 * 2 * (dCPU[0] + 1), (sizeWB + 16 * 2 * max(nbN * 4, 15 * dCPU[0])) * sizeof(float) >> > (WBGPU,
																		C, LL, 2 * dCPU[0], sizeWB,
																		low, up, R, q, Ver, num, nbN, 4,
																		siV, siVD, siR, siRD, MinMax);
		
	cudaDeviceSynchronize();
	testCUDA(cudaMemcpy(Ccpu  , C  , nop * sizeCB * sizeof(float)              , cudaMemcpyDeviceToHost));
	testCUDA(cudaMemcpy(LLcpu , LL , NbVertices * nop * sizeof(float)          , cudaMemcpyDeviceToHost));
	testCUDA(cudaMemcpy(Vercpu, Ver, NbVertices * nop * dCPU[0] * sizeof(float), cudaMemcpyDeviceToHost));
	testCUDA(cudaMemcpy(numcpu, num, 2 * nop * sizeof(float)                   , cudaMemcpyDeviceToHost));


	printf("The estimated Minimum level %f\n", (float)0.0000001f * MinMax[0]);
	printf("The estimated Maximum level %f\n", (float)0.0000001f * MinMax[1]);
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
	cudaEventElapsedTime(&Tim,			// GPU timer instructions
		start, stop);					// GPU timer instructions
	cudaEventDestroy(start);			// GPU timer instructions
	cudaEventDestroy(stop);				// GPU timer instructions

	printf("Execution time %f ms\n", Tim);

	testCUDA(cudaFree(MinMax));
	testCUDA(cudaFree(C));
	free(Ccpu);
	testCUDA(cudaFree(LL));
	free(LLcpu);
	testCUDA(cudaFree(Ver));
	free(Vercpu);
	testCUDA(cudaFree(R));
	testCUDA(cudaFree(q));
	testCUDA(cudaFree(num));
	free(numcpu);
	testCUDA(cudaFree(WBGPU));

	return 0;
}