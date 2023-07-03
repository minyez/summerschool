#pragma once

#pragma omp declare target
double my_sum(double a, double b);
#pragma omp end declare target 
