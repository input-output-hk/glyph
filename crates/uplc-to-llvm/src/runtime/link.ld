ENTRY(_start)

MEMORY {
    /* BitVMX-specific memory layout */
    FLASH (rx)  : ORIGIN = 0x00000000, LENGTH = 1M   /* Code segment */
    RAM   (rw)  : ORIGIN = 0x20000000, LENGTH = 1M   /* Data segment */
}

SECTIONS {
    .text : {
        *(.text.start)
        *(.text*)
    } > FLASH

    .code_segment : {
        *(.code_segment*)
    } > FLASH

    .rodata : {
        *(.rodata*)
    } > FLASH

    .data : {
        *(.data*)
    } > RAM AT > FLASH

    .bss : {
        *(.bss*)
        *(COMMON)
    } > RAM
} 