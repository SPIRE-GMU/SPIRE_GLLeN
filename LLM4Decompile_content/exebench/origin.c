#include <stdint.h>
#include <stdio.h>

# 1 
__attribute__ ((weak)) void ADC0_THCMP_IRQHandler(void)
{ ADC0_THCMP_DriverIRQHandler();
}