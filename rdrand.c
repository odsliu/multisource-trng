#include <stdint.h>

// Generate a random 64-bit value using RDRAND
// Returns 1 on success, 0 on failure
int rdrand64(uint64_t *value) {
    unsigned char ret;
    asm volatile (
        "rdrand %0;"   // Generate random number into `value`
        "setc %1"      // Set `ret` to 1 if successful, 0 otherwise
        : "=r" (*value), "=qm" (ret)
    );
    return (int)ret;
}