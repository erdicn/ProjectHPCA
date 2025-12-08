#include <iostream>
#include <fstream>
#include <vector>
#include <stdio.h>
#include <curand_kernel.h>


// Function that catches the error 
void testCUDA(cudaError_t error, const char* file, int line) {
    
	if (error != cudaSuccess) {
        printf("There is an error in file %s at line %d\n", file, line);
		exit(EXIT_FAILURE);
	}
}

// Has to be defined in the compilation in order to get the correct value of the 
// macros __FILE__ and __LINE__
#define testCUDA(error) (testCUDA(error, __FILE__ , __LINE__))

void printArray(int* a, int len){
    for(int i = 0; i < len; i++){
        std::cout << a[i] << std::endl;
    }
}

// Given an empty vector fills it with the weigt and biases vector from file 
int readArrayFromFile(int sizeWB, std::string path, std::vector<float>& WB_data){
	// std::string path = "weights.txt";  // path of the file with one value per row

	// Reading the file
	std::ifstream file(path);
	if (!file.is_open()) {
		std::cerr << "Error: impossible to open " << path << std::endl;
		return 1;
	}

	// std::vector<float> WB_data;
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
    return 0;
}

int initializeDomainsArray(int nb_layers, int input_layer_dim, int output_layer_dim, int* domains){
    domains[0] = input_layer_dim; // input layer
    for(int i = 1; i < nb_layers-1; i++)
        domains[i] = 4; // hidden layers 
    domains[nb_layers-1] = 1; // output layer
}

int getSizeOfWeightAndBiases(int nb_layers, int* domains){
    int sizeWB = 0;
    for(int i = 0; i < nb_layers-1; i++){
        sizeWB += (domains[i]+1) * domains[i+1];
    }
}

int getNbOfPossibleCOmbinationsOfNeurons(int nb_layers, int* domains){
    int nb_subpolytopes = 0;
    for(int i = 1; i < nb_layers-1; i++){
        nb_subpolytopes += domains[i];
    }
    nb_subpolytopes = 1 << nb_subpolytopes; // 2^nb_neurons
}

#define INPUT_LAYER_DIM 2
#define NB_LAYERS 4+1 // TODO read from external file so it syncs with python

__constant__ int dld[NB_LAYERS];

typedef struct {
    int ver;
    int lvl;
} Num_t;


typedef struct {
	int min;
    int max;
    int final_max;
} MinMax_t;

int main(void){
    // TOOD from file 
    int domains_CPU[NB_LAYERS];
    initializeDomainsArray(NB_LAYERS, INPUT_LAYER_DIM, 1, domains_CPU);

	// nb_connections_layers+biases = nn_i * nn_i+1 + nn_i+1
    int sizeWB = getSizeOfWeightAndBiases(NB_LAYERS, domains_CPU);
    std::cout << sizeWB << std::endl;

    MinMax_t *min_max;
	testCUDA(cudaMallocManaged(&min_max, sizeof(MinMax_t)));
	min_max->min =  5000000;
	min_max->max = -5000000;
	min_max->final_max =  0;
	
	
    int nb_subpolytopes = getNbOfPossibleCOmbinationsOfNeurons(NB_LAYERS, domains_CPU); // number of possible combinations of neurons active inactive 2^nb_neurons in the hidden layers 

    // BLACKBOX
    int deltaSize = 2*domains_CPU[0]; 
    int sizeB     = 2*domains_CPU[0]; 
    int sizeCB    = 2*domains_CPU[0];
    for(int i = 1; i < NB_LAYERS; i++){
        deltaSize += domains_CPU[i];
        sizeB     += domains_CPU[i];
        sizeCB    += domains_CPU[i];
    }
    sizeCB *= 1 + domains_CPU[0];

	for(int k = 1; k < 1 + domains_CPU[0]; k++){
		sizeCB += (deltaSize - k) * (1 + domains_CPU[0] - k);
		sizeB += (deltaSize - k);
	}
	
	// maximum number of vertices by subpolytope: d0 combinations among m-d0
    int m = 0; for(int i = 1; i < NB_LAYERS; i++) m += domains_CPU[i];
    int nb_vertices = 1000; // TODO m parmi d0

    int magic_valuesCPU[5], *magic_values;
	// testCUDA(cudaMallocManaged(&magic_values, 5*sizeof(int)));
    magic_valuesCPU[0] = sizeCB - deltaSize * (1 + domains_CPU[0]); // Values should be stored at the end Algo (4.4) deltaC
	magic_valuesCPU[1] = sizeCB;  // The total size needed for each configuration s DeltaC
    magic_valuesCPU[2] = nb_vertices;
	magic_valuesCPU[3] = sizeB - deltaSize; // Values should be stored at the end Algo (4.4) deltaB
	magic_valuesCPU[4] = sizeB; // The total size needed for each configuration s DeltaB TODO never used ???

	testCUDA(cudaMalloc(&magic_values, 5 * sizeof(int)));
	testCUDA(cudaMemcpy( magic_values, magic_valuesCPU, 5 * sizeof(int), cudaMemcpyHostToDevice));
    testCUDA(cudaMemcpyToSymbol(dld, domains_CPU, NB_LAYERS * sizeof(int), 0, cudaMemcpyHostToDevice));
	
	// C is C matrix and beta, levels: list of levels, Ver contains the list of vertices,
	// R contains the isometry matrices, q contains the translation values
	float* C, * Ccpu, * levels, * levelscpu, * Ver, * Vercpu, * R, *q;
	Num_t* num, * numcpu; // contains the true number of vertices and levels

    std::string pathWB = "weights.txt";
    std::vector<float> WB_data;
    if(!readArrayFromFile(sizeWB, pathWB, WB_data)){ return 1; }


	testCUDA(cudaMalloc(&C, sizeCB * nb_subpolytopes * sizeof(float)));
	Ccpu = (float*)malloc(  sizeCB * nb_subpolytopes * sizeof(float));

	testCUDA(cudaMalloc(&levels, 2 * nb_vertices * nb_subpolytopes * sizeof(float))); // twice the size to be able to have a sorted list
	levelscpu = (float*)malloc(      nb_vertices * nb_subpolytopes * sizeof(float));
	testCUDA(cudaMalloc(&Ver,        nb_vertices * nb_subpolytopes * domains_CPU[0] * sizeof(float)));
	Vercpu = (float*)malloc(         nb_vertices * nb_subpolytopes * domains_CPU[0] * sizeof(float));

    // TODO how those these work
	int siV  = (domains_CPU[0] == 2) ? (3 + 2) : (4 + 3 + 2); // binding index for volume V
	int siVD = (domains_CPU[0] == 2) ? (  2  ) : (3 + 2);     // binding index for volume V TODO what are the differnces between them
	int siR  = (domains_CPU[0] == 2) ? (  4  ) : (9 + 4);     // binding index for volume R
	int siRD = (domains_CPU[0] == 2) ?    0    : (  4  );     // binding index for volume R

	if(domains_CPU[0] > 3) std::cout << "The code is not made for dimension > 3\n";

	// TODO instead of doing 4 layers do 3 then add it 
	testCUDA(cudaMalloc(&R, siR * nb_subpolytopes * sizeof(float)));
	testCUDA(cudaMalloc(&num,     nb_subpolytopes * sizeof(Num_t)));
	numcpu = (Num_t*)malloc(      nb_subpolytopes * sizeof(Num_t));

	testCUDA(cudaMalloc(&q, nb_subpolytopes * domains_CPU[0] * sizeof(float)));
	
	float* WBGPU; // Coefficents of matrices and bias vectors

	testCUDA(cudaMalloc(&WBGPU, sizeWB * sizeof(float)));



	testCUDA(cudaMemcpy(WBGPU, WB_data.data(), sizeWB * sizeof(float), cudaMemcpyHostToDevice));

	int low = 0;  // the starting index
	int up = nb_subpolytopes; // the ending index
	int nbN = 16; // Number of neurones TODO  we should have 15 no ? in the 3d we should have 16
	
	testCUDA(cudaMemset(C, 0, nb_subpolytopes * sizeCB * sizeof(float)));

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

	testCUDA(cudaMemset(levels , 0, 2 * nb_vertices * nb_subpolytopes * sizeof(float)));
	testCUDA(cudaMemset(Ver    , 0,     nb_vertices * nb_subpolytopes * domains_CPU[0] * sizeof(float)));
	testCUDA(cudaMemset(num    , 0, 2 *               nb_subpolytopes * sizeof(float)));

	//        TODO     TODO      why + 1         
	// Part_k <<<16 * 8, 16 * 2 * (dCPU[0] + 1), (sizeWB + 16 * 2 * max(nbN * 4, 15 * dCPU[0])) * sizeof(float) >>>
	// levL_k <<<16 * 8, 16 * 2 * (dCPU[0] + 1), (sizeWB + 16 * 2 * max(nbN * 4, 15 * dCPU[0])) * sizeof(float) >>>
	int min_nb_vertices_by_polytope = domains_CPU[0] + 1; // d0 + 1 = minimum number of vertices for a d0 dimension polytope
	// partialPart_k <<<16 * 8, 16 * 2 * min_nb_vertices_by_polytope, (sizeWB + 16 * 2 * max(nbN * 4, 15 * domains_CPU[0])) * sizeof(float) >>>
	// 		(WBGPU, C, levels, 2 * domains_CPU[0], sizeWB, 
	// 			low, up, R, q, Ver, num, nbN, 4, 
	// 			siV, siVD, siR, siRD, min_max, magic_values);

	testCUDA(cudaDeviceSynchronize()); // TODO not sure if i need this sync
	
	// TODO maybe there is a better way to calculate and create a new array
	testCUDA(cudaMemcpy(numcpu, num, nb_subpolytopes * sizeof(Num_t), cudaMemcpyDeviceToHost));
	int* non_empty_num_indices, *non_empty_num_indicesCPU, counter_non_empty_num = 0;
	non_empty_num_indicesCPU = (int*) malloc(nb_subpolytopes * sizeof(int));
	
	// Non empty num calculation
	for(int i = 0; i < nb_subpolytopes; i++){
		if (numcpu[i].ver != 0){
			non_empty_num_indicesCPU[counter_non_empty_num++] = i;
			// std::cout << i << std::endl;
		}
	}

	testCUDA(cudaMalloc(&non_empty_num_indices, counter_non_empty_num * sizeof(int)));
	testCUDA(cudaMemcpy(non_empty_num_indices, non_empty_num_indicesCPU, 
											counter_non_empty_num * sizeof(int), cudaMemcpyHostToDevice));


	
	// levL_k <<<counter_non_empty_num, 16 * 2 * min_nb_vertices_by_polytope, (sizeWB + 16 * 2 * max(nbN * 4, 15 * domains_CPU[0])) * sizeof(float) >>>
	// 		(WBGPU, C, levels, 2 * domains_CPU[0], sizeWB, 
	// 			low, up, R, q, Ver, num, nbN, 4, 
	// 			siV, siVD, siR, siRD, min_max, magic_values, non_empty_num_indices);
	
	testCUDA(cudaDeviceSynchronize());
	
	testCUDA(cudaMemcpy(Ccpu  , C  , nb_subpolytopes * sizeCB * sizeof(float)              , cudaMemcpyDeviceToHost));
	testCUDA(cudaMemcpy(levelscpu , levels , nb_vertices * nb_subpolytopes * sizeof(float)          , cudaMemcpyDeviceToHost));
	testCUDA(cudaMemcpy(Vercpu, Ver, nb_vertices * nb_subpolytopes * domains_CPU[0] * sizeof(float), cudaMemcpyDeviceToHost));
	testCUDA(cudaMemcpy(numcpu, num, 2 * nb_subpolytopes * sizeof(float)                   , cudaMemcpyDeviceToHost));


	printf("The computed Minimum level %f\n", (float)0.0000001f * min_max->min);
	printf("The computed Maximum level %f\n", (float)0.0000001f * min_max->max);
	printf("Number of levels %i\n", min_max->final_max);
	printf("With notation (index of pol, number of vertices, number of levels), the non-empty polytopes are:\n");
	int count = 0;
	for (int k = 0; k < nb_subpolytopes; k++) {
		if (numcpu[k].ver > 0) {
			printf("(%d, %d, %d), ", k, numcpu[k].ver, numcpu[k].lvl);
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

	testCUDA(cudaFree(min_max));
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
	testCUDA(cudaFree(magic_values));
	free(non_empty_num_indicesCPU);

    return 0;
}