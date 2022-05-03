
#include <stdio.h>

void cuda_test();
int filter_average();

int main()
{
	int ret = 0;

	//ret = kernel();
	ret = filter_average();

	//cuda_test();

	//getchar();
	return ret;
}