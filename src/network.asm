section .data
    EULER       dq  0x4005BF0A8B145769
    FILENAME    db  'file.txt', 0
    READONLY    equ 0
    WRITEONLY   equ 1
    READWRITE   equ 2
    INTBUFFER   db '00000', 0xA, 0

section .bss
    
section .text
    global _start

_start:
    ; just shit out what is in a file
    mov rax, 2
    mov rdi, FILENAME
    mov rsi, READWRITE
    syscall 

    ; file handle in rax
    ; read from file into buffer
    mov rdi, rax
    mov rax, 0
    mov rsi, INTBUFFER
    mov rdx, 5
    syscall

    ; write to console
    mov rax, 1
    push rdi
    mov rdi, 1
    mov rsi, INTBUFFER
    mov rdx, 7
    syscall

    ; write back into file
    mov rax, 1
    pop rdi
    mov rsi, INTBUFFER
    mov rdx, 5
    syscall

    ; return 0
    mov rax, 60             ; return 0, 60 to exit
    mov rdi, 0
    syscall