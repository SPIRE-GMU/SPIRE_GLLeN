#include <stdint.h>
void main()
{}
inline uint32_t pcg_oneseq_64_xsl_rr_32_boundedrand_r(struct pcg_state_64* rng,
                                                      uint32_t bound)
{
    uint32_t threshold = -bound % bound;
    for (;;) {
        uint32_t r = pcg_oneseq_64_xsl_rr_32_random_r(rng);
        if (r >= threshold)
            return r % bound;
    }
}
