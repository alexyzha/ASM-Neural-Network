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
        INT_MAX     equ   0x7FFFFFFFFFFFFFFF
    ; prng xoshiro256
        XORSHI_0    dq    0xA22CD65BFF6B2CE0
        XORSHI_1    dq    0x7AF05449A5465795
        XORSHI_2    dq    0xF46F7AAF65D601E1
        XORSHI_3    dq    0xC5DB0BDFE6A51996

section .bss
    HIDDEN_WEIGHTS resq 23520
    OUTPUT_WEIGHTS resq 300

section .text
    global _start
    extern printf
    extern exp
_start:
    
    ; ss = scalar single-precision
    ; scalar = single val
    fninit
    fld qword [EULER]
    fld qword [EULER]
    fmul st0, st1
    
    sub rsp, 8              ; allocate memory to stack
    fstp qword [rsp]        ; plop st0 into stack
    movsd xmm0, qword [rsp] ; put float into xmm0
    add rsp, 8              ; deallocate memory
    call PRINT_FLOAT        ; print

    mov rsi, INT_MAX
    call PRINT_INT

    call SIGMOID
    call PRINT_FLOAT

    mov rbx, 20
    
    LBG:
        dec rbx
        call RANDOM_FLOAT
        call PRINT_FLOAT
        test rbx, rbx
        jnz LBG

    LED:
    

    call EXIT

RANDOM_FLOAT:
    ; return float in xmm0
    sub rsp, 16
    mov [rsp+8], rax
    mov rax, INT_MAX
    mov [rsp], rax
    call XORSHIT
    fild qword [rsp+8]
    fild qword [rsp]
    fdivp                   ; st1 = st1/st0, pop st0
    sub rsp, 8
    fstp qword [rsp]
    movsd xmm0, qword [rsp]
    add rsp, 24             ; 16 (original) + 8 (for xmm0)
    ret

XORSHIT:
    mov rax, [XORSHI_0]
    add rax, [XORSHI_3]     ; return x[0] + x[3]
    push rbx
    push rcx
    push rdx

    mov rbx, [XORSHI_1]
    shl rbx, 17             ; t = x[1] << 17
    mov rcx, [XORSHI_2]     ; x[2] ^= x[0]
    xor rcx, [XORSHI_0]     
    mov [XORSHI_2], rcx     
    
    mov rcx, [XORSHI_3]     ; x[3] ^= x[1]
    xor rcx, [XORSHI_1]     
    mov [XORSHI_3], rcx     
    
    mov rcx, [XORSHI_1]     ; x[1] ^= x[2]
    xor rcx, [XORSHI_2]     
    mov [XORSHI_1], rcx     
    
    mov rcx, [XORSHI_0]     ; x[0] ^= x[3]
    xor rcx, [XORSHI_3]     
    mov [XORSHI_0], rcx  
    
    mov rcx, [XORSHI_2]     ; x[2] ^= t
    xor rcx, rbx
    mov [XORSHI_2], rcx

    mov rcx, [XORSHI_3]     ; s[3] = (s[3] << 45) | (s[3] >> 19)
    shl rcx, 45
    mov rdx, [XORSHI_3]
    shr rdx, 19
    or rcx, rdx
    mov [XORSHI_3], rcx

    pop rdx
    pop rcx
    pop rbx
    ret

    

SIGMOID:
    ; 1/1+e^-x
    ; x in xmm0
    sub rsp, 8                  ; -8 for fstp and fld
    call exp                    ; xmm0 = e^x
    movsd [rsp], xmm0           ; [rsp] = e^x
    fld qword [rsp]
    fld1                        ; [1][e^x]
    fdivrp                      ; st0 = 1/e^x
    fld1                        ; [1][e^-x]
    faddp                       ; st1 = st1 + st0, pop st0
    fld1                       
    fdivrp                      ; st1 = st0/st1, pop st0
    fstp qword [rsp]            ; load result into xmm0
    movsd xmm0, qword [rsp]
    add rsp, 8                  ; clean stack
    ret

PRINT_INT:
    ; "%d" in rdi
    ; int in rsi
    ; 1 in rax?
    push rax
    push rdi
    mov rax, 1
    mov rdi, INT_PERC
    call printf
    pop rdi
    pop rax
    ret

PRINT_FLOAT:
    ; double in xmm0
    ; "%f" in rdi
    ; 1 in rax
    push rax
    push rdi
    mov rdi, FLOAT_PERC
    mov rax, 1
    call printf
    pop rdi
    pop rax
    ret

EXIT:
    mov rax, 60
    xor rdi, rdi
    syscall