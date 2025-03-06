#include <stdint.h>

// BitVMX-specific memory layout
#define CODE_SEGMENT_SIZE (1024 * 1024)  // 1MB
#define DATA_SEGMENT_SIZE (1024 * 1024)  // 1MB

// Memory segments
__attribute__((section("__DATA,__code")))
static uint8_t code_segment[CODE_SEGMENT_SIZE] __attribute__((aligned(16)));

__attribute__((section("__DATA,__data")))
static uint8_t data_segment[DATA_SEGMENT_SIZE] __attribute__((aligned(16)));

// Entry point for UPLC program
__attribute__((section("__TEXT,__text")))
void _start(void) {
    // Program entry point
    // The actual UPLC program code will be placed here by the compiler
    asm volatile("nop");
} 