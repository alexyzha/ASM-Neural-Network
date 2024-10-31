; nasm -f elf64 network.asm -o network.o && gcc -m64 network.o -o network -nostartfiles -no-pie -lc
section .data
    ; file i/o
        FILENAME    db    'file.txt', 0
        READONLY    equ   0
        WRITEONLY   equ   1
        READWRITE   equ   2
    ; cstd i/o
        FLOAT_PERC  db    '%f', 0xA, 0
        INT_PERC    db    '%d', 0xA, 0
    ; network consts
        INPUT_CT    equ   784
        HIDDEN_CT   equ   30 
        OUTPUT_CT   equ   10
        EULER       dq    2.718281828
    ; for random float generation
        INT_MAX     equ   2147483647
    ; random number generator xoshiro256
        XORSHI_4    dq    0xA22CD65BFF6B2CE0, 0x7AF05449A5465795, 0xF46F7AAF65D601E1, 0xC5DB0BDFE6A51996
        XORSHI_T    dq    0x0, 0x0, 0x0, 0x0

section .bss
    HIDDEN_WEIGHTS resq 23520
    OUTPUT_WEIGHTS resq 300

section .text
    global _start
    extern printf

_start:
    
    ; ss = scalar single-precision
    ; scalar = single val
    fninit
    fld qword [EULER]
    fld qword [EULER]
    fmul st0, st1
    
    sub rsp, 16             ; allocate memory to stack
    fstp qword [rsp]        ; plop st0 into stack
    movsd xmm0, qword [rsp] ; put float into xmm0
    add rsp, 16             ; deallocate memory
    call PRINT_FLOAT        ; print

    ; return 0
    mov rax, 60             ; return 0, 60 to exit
    xor rdi, rdi
    syscall

GEN_FLOAT:



    ret


PRINT_FLOAT:
    ; need stack alignment (sub/add 8) beacuse rsp moves down with function calls
    ; assumes stack alignment @16
    sub rsp, 8
    ; input float in xmm0
    ; printf requires:
    ; - float in xmm0
    ; - type in rdi
    ; - 1 in rax
        push rax
            push rdi
                mov rdi, FLOAT_PERC
                mov rax, 1
                call printf
            pop rdi
        pop rax
    ; move stack back into place
    add rsp, 8
    ret