ENTRY(_start)


MEMORY
{
  TEXT  (rx)  : ORIGIN = 0x80000000, LENGTH = 0x1000000   /* 16MB */
  DATA  (r)  : ORIGIN = 0x90000000, LENGTH = 0x1000000   /* 16MB */
  BSS   (rw)  : ORIGIN = 0xA0000000, LENGTH = 0x1000000   /* 16MB */
  HEAP  (rw)  : ORIGIN = 0xC0000000, LENGTH = 0x1000000   /* 16MB */
  FRAME (rw)  : ORIGIN = 0xD0000000, LENGTH = 0x1000000   /* 16MB */
  STACK (rw)  : ORIGIN = 0xE0000000, LENGTH = 0x1000000   /* 16MB */
}

SECTIONS
{
  . = ORIGIN(TEXT);
  .text ALIGN(4) :
  {
    *(.text*)
  } > TEXT

  . = ORIGIN(DATA);
  .data ALIGN(4) :
  {
    *(.data*)
  } > DATA

  . = ORIGIN(BSS);
  .bss ALIGN(4) :
  {
    __bss_bottom = .;
    *(.bss*)
    *(COMMON)
    __bss_top = .;
  } > BSS

  . = ORIGIN(HEAP);
  .heap ALIGN(4) :
  {
     __heap_bottom = .;
    *(.heap*)
    . = . + 0x100000;  /* Allocate 1MB of heap space */
    __heap_top = .;
  } > HEAP

  . = ORIGIN(FRAME);
  .frame ALIGN(4) :
  {
    __frame_bottom = .;
    *(.frame*)
    . = . + 0x100000;  /* Allocate 1MB of frame space */
    __frame_top = .;
  } > FRAME

  . = ORIGIN(STACK);
  .stack ALIGN(4) :
  {
    __stack_bottom = .;
    *(.stack*)
    . = . + 0x100000;  /* Allocate 1MB of stack space */
    __stack_top = .;
  } > STACK
}
