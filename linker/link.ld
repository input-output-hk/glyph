ENTRY(_start)


MEMORY
{
  TEXT  (rx)  : ORIGIN = 0x80000000, LENGTH = 0x1000000   /* 16MB */
  DATA  (rw)  : ORIGIN = 0x90000000, LENGTH = 0x1000000   /* 16MB */
  BSS   (rw)  : ORIGIN = 0xA0000000, LENGTH = 0x1000000   /* 16MB */
  HEAP  (rw)  : ORIGIN = 0xC0000000, LENGTH = 0x1000000   /* 16MB */
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
    *(.bss*)
  } > BSS

  . = ORIGIN(HEAP);
  .heap ALIGN(4) :
  {
     __heap_bottom = .;
    . = . + LENGTH(HEAP);
    __heap_top = .;
  } > HEAP

  . = ORIGIN(STACK);
  .stack ALIGN(4) :
  {
    __stack_bottom = .;
    . = . + LENGTH(STACK);
    __stack_top = .;
  } > STACK
}
