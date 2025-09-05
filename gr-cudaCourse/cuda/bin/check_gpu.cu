/******************************************************************************
 * Check to see if we can run compiled cuda kernels.
 * Cuda code should be compatible will newer driver versions than it was
 * compiled with, but not older.

 * This program gets the host driver version and the driver version that it
 * was compiled with.  It verifies that these are compatible.
******************************************************************************/

#include <iostream>

int get_cuda_compiled_version() {
    int ver = 0;
    cudaRuntimeGetVersion(&ver);
    return ver;
}

int get_cuda_driver_version() {
    int ver = 0;
    cudaDriverGetVersion(&ver);
    return ver;
}
/*
Check if we have access to a GPU.
Return 0 if we do.
*/
int main() {
    int driver_version = get_cuda_driver_version();
    int compiled_version = get_cuda_compiled_version();
    if (driver_version == 0) {
        std::cout << "No cuda driver detected...Can't use GPU" << std::endl;
        return 1;
    } else if (driver_version < compiled_version) {
        std::cout << "Incompatible driver: Compiled = " << compiled_version << ", Driver version = " << driver_version
                  << std::endl;
        return 2;
    } else {
        std::cout << "Driver version: " << driver_version << std::endl;
        return 0;
    }
}
