#include <stdint.h>
#include <stdio.h>
#include <values.h>

#include <float.h>

# 1 
double min(double *val, int len) {
 double w;
 int i;
 w = MAXDOUBLE;
 for (i = 0; i < len; i++) if (val[i] < w) w = val[i];
 return w;
}