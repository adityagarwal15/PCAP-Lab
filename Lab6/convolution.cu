#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<stdio.h>

__global__ void kernel(float*da,float*db,float*dc,int mw,int w){
    int i=blockIdx.x*blockDim.x+threadIdx.x;

    int s=i-(mw)/2;  // Determine the starting index for the current element
    float pv=0;  // Initialize the result for the current element
    
    // Convolution operation: sum of element-wise product between `da` and `db`
    for(int j=0;j<w;j++){
        if(s+j>=0&&s+j<w){  // Check if the indices are within bounds
            pv+=da[s+j]*db[j];  // Sum the products
        }
    }
    dc[i]=pv;  // Store the result in the output vector
}

int main(){
    int n1,n2;

    // Ask for the length of the input vector and mask
    printf("Length of the vector : ");
    scanf("%d",&n1);
    printf("Enter the length of mask : ");
    scanf("%d",&n2);

    // Allocate memory for the input vectors and the output vector
    float a[n1],b[n2],c[n1];
    float *da,*db,*dc;

    cudaMalloc((void **)&da,n1*sizeof(float));
    cudaMalloc((void **)&db,n2*sizeof(float));
    cudaMalloc((void **)&dc,n1*sizeof(float));

    // Input the vectors a and b
    printf("Enter vector one : ");
    for(int i=0;i<n1;i++)
        scanf("%f",&a[i]);
    printf("Enter vector two (aka mask) : ");
    for(int i=0;i<n2;i++)
        scanf("%f",&b[i]);
    
    // Copy the input vectors to device memory
    cudaMemcpy(da,a,n1*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(db,b,n2*sizeof(float),cudaMemcpyHostToDevice);

    // Define the grid and block dimensions for CUDA
    dim3 grid(n1,1,1);
    dim3 blk(1,1,1);

    // Call the kernel to perform the convolution
    kernel<<<grid,blk>>>(da,db,dc,n2,n1);

    // Copy the result back to the host
    cudaMemcpy(c,dc,n1*sizeof(float),cudaMemcpyDeviceToHost);

    // Print the output vector
    for(int i=0;i<n1;i++)
        printf("%f\t",c[i]);
    printf("\n");

    // Free device memory
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
}

/*Length of the vector : 5
//Enter the length of mask : 3
Enter vector one : 1 2 3 4 5
Enter vector two (aka mask) : 0.2 0.5 0.2

0.400000   1.000000   1.600000   2.000000   1.600000
*/

