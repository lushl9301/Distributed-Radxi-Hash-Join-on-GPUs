#include <iostream>

#ifndef DEBUG_CUH_

#define DEBUG_CUH_
/**displayGPUBuffer(templateType *, templateType *, long )
* Template function that can read back any GPU buffer and display its content.
* The function is purely used for debugging purposes to verify the value of
* any GPU buffer during execution.
* deviceBuffer  : The device side buffer that needs to be read.
* hostBuffer    : The host side buffer to which the GPU buffer contents are copied.
* len           : The length of buffer or the number of elements that needs to be copied and displayed.
* modValue      : The value used for perfoming modulus operation before displaying the value.
*/
template <typename templateType>
void displayGPUBuffer(templateType deviceBuffer, templateType hostBuffer, long len, int modValue = 0){

	//setting up the logger

	//making sure that all GPU processes are completed
	cudaDeviceSynchronize();

	//copying the buffer back to the host memory
	cudaMemcpy(hostBuffer, deviceBuffer, len * sizeof(int), cudaMemcpyDeviceToHost);

	//making sure the data copy is complete
	cudaDeviceSynchronize();


	//displaying the contents of the buffer
	for(long i = 0; i < len; i++){

		//checking if a modulus operation needs to be performed before displaying the value
		if(modValue != 0){
			std::cout << hostBuffer[i] % modValue << std::endl;
		}
		else{
			std::cout << hostBuffer[i] << std::endl;
		}
	}
}

/**displayGPUBuffer(templateType *, templateType *, long )
* Template function that can read back any GPU buffer and display its content.
* The function is purely used for debugging purposes to verify the value of
* any GPU buffer during execution.
* deviceBuffer  : The device side buffer that needs to be read.
* hostBuffer    : The host side buffer to which the GPU buffer contents are copied.
* len           : The length of buffer or the number of elements that needs to be copied and displayed.
* modValue      : The value used for perfoming modulus operation before displaying the value.
*/
template <typename templateType>
void displayCPUBuffer(templateType hostBuffer, long len, int modValue = 0){

	//setting up the logger

	//making sure that all GPU processes are completed
	cudaDeviceSynchronize();

	for(long i = 0; i < len; i++){

		//checking if a modulus operation needs to be performed before displaying the value
		if(modValue != 0){
			std::cout << hostBuffer[i] % modValue << std::endl;
		}
		else{
			std::cout << hostBuffer[i] << std::endl;
		}

	}
}

/**displayGPUBuffer(templateType *, templateType *, long )
* Template function that can read back any GPU buffer and display the negative contents in the buffer.
* The function is purely used for debugging purposes to verify the value of any GPU buffer during execution.
* deviceBuffer  : The device side buffer that needs to be read.
* hostBuffer    : The host side buffer to which the GPU buffer contents are copied.
* len           : The length of buffer or the number of elements that needs to be copied and displayed.
* modValue      : The value used for perfoming modulus operation before displaying the value.
*/
template <typename templateType>
void displayGPUBufferNegative(templateType deviceBuffer, templateType hostBuffer, long len){

	//setting up the logger

	//making sure that all GPU processes are completed
	cudaDeviceSynchronize();

	//copying the buffer back to the host memory
	cudaMemcpy(hostBuffer, deviceBuffer, len * sizeof(int), cudaMemcpyDeviceToHost);

	//making sure the data copy is complete
	cudaDeviceSynchronize();

	for(long i = 0; i < len; i++){

		if(hostBuffer[i] < 0){
			std::cout << hostBuffer[i] << std::endl;
		}
	}
}

#endif