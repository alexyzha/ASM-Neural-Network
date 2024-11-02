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
        HIDDEN_NCT  equ   23520
        OUTPUT_NCT  equ   300
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
    INPUTS          resq  784
    HIDDEN_WEIGHTS  resq  23520
    OUTPUT_WEIGHTS  resq  300
    HIDDEN_BIASES   resq  30 
    OUTPUT_BIASES   resq  10
    HIDDEN_OUTPUT   resq  30
    OUTPUT_OUTPUT   resq  10

section .text
    global _start
    extern printf
    extern exp
_start:
    fninit                  ; initialize the fpu
    call INIT_WEIGHTS       ; create prand weights
    call INIT_BIASES        ; create prand biases
    ; call PRINT_WEIGHTS
    ; call PRINT_BIASES
    call TEST1_INPUTS       ; tested
    ; call PRINT_INPUTS
    call ZERO_OUTPUTS       ; tested
    ; call PRINT_OUTPUTS
    ; call FORWARD_PROP     ; ...
                            ; needs more work

    call EXIT               ; return 0;



FORWARD_PROP:
    ; ALL HIDDEN OUTPUTS
    mov rcx, 0              ; 0->23520, h node ct
    mov r9, 0               ; 0->30, node number
    HIDDEN_OUTER_BG:
        mov rdx, 0          ; 0->784
        fld1                ; set st0 = 0.0
        fld1
        fsubp
        sub rsp, 16
        HIDDEN_INNER_BG:
            lea rax, [HIDDEN_WEIGHTS+rcx*8]
            lea r8, [INPUTS+rdx*8]
            mov [rsp+8], r8
            mov [rsp], rax
            fld qword [rsp+8]
            fld qword [rsp]
            fmulp
            faddp           ; st0 holds sum
            inc rcx
            inc rdx
            cmp rdx, INPUT_CT
            jne HIDDEN_INNER_BG

        ; st0 = \sum(weight[node][i]*input[i])
        ; add bias
        lea rax, [HIDDEN_BIASES+r9*8]
        mov [rsp], rax
        fld qword [rsp]
        faddp               ; st0 = \sum(weights*inputs) + bias
        fstp qword [rsp]
        movsd xmm0, qword [rsp]
        call SIGMOID        ; \sigma(calc) in xmm0

        ; NEED TO SAVE IN HIDDEN_OUTPUT[x] HERE
        lea rax, [HIDDEN_OUTPUT+r9*8]
        movsd qword [rax], xmm0

        add rsp, 16
        add rcx, INPUT_CT
        inc r9
        cmp rcx, HIDDEN_NCT
        jne HIDDEN_OUTER_BG

    ; ALL OUTPUT OUTPUTS
    ; LARGEST NUMBER IN RAX
    ret

PRINT_OUTPUTS:
    ; print hidden outputs
    lea rbx, HIDDEN_OUTPUT
    mov r12, 0
    PO_LBG_H:
        movsd xmm0, qword [rbx]
        call PRINT_FLOAT
        add rbx, 8
        inc r12
        cmp r12, HIDDEN_CT
        jne PO_LBG_H
    ; print output outputs
    lea rbx, OUTPUT_OUTPUT
    mov r12, 0
    PO_LBG_O:
        movsd xmm0, qword [rbx]
        call PRINT_FLOAT
        add rbx, 8
        inc r12
        cmp r12, OUTPUT_CT
        jne PO_LBG_O
    ret

TEST1_INPUTS:
    ; put 1.0 in xmm0
    sub rsp, 8
    fld1
    fstp qword [rsp]
    movsd xmm0, qword [rsp]
    add rsp, 8
    ; put 1.0 in all inputs
    mov rcx, 0
    lea rax, [INPUTS]
    TI_LBG:
        movsd qword [rax], xmm0
        add rax, 8
        inc rcx 
        cmp rcx, INPUT_CT
        jne TI_LBG
    ret

PRINT_INPUTS:
    mov r12, 0
    lea rbx, [INPUTS]
    PI_LBG:
        movsd xmm0, qword [rbx]
        call PRINT_FLOAT
        add rbx, 8
        inc r12
        cmp r12, INPUT_CT
        jne PI_LBG
    ret

ZERO_OUTPUTS:
    fld1
    fld1
    fsubp
    sub rsp, 8
    fstp qword [rsp]
    movsd xmm0, qword [rsp]
    add rsp, 8              ; 0.0 in xmm0
    ; iterate through all hidden outputs
    lea rax, [HIDDEN_OUTPUT]
    mov rcx, HIDDEN_CT
    ZO_HIDDEN_LBG:
        movsd [rax], xmm0
        add rax, 8
        dec rcx
        test rcx, rcx
        jnz ZO_HIDDEN_LBG
    ; iterate through all outputs
    lea rax, [OUTPUT_OUTPUT]
    mov rcx, OUTPUT_CT
    ZO_OUTPUT_LBG:
        movsd [rax], xmm0
        add rax, 8
        dec rcx
        test rcx, rcx
        jnz ZO_OUTPUT_LBG
    ret

PRINT_WEIGHTS:
    ; print hidden weights
    lea rbx, [HIDDEN_WEIGHTS]
    mov r12, HIDDEN_NCT
    PW_LBG_H:
        movsd xmm0, qword [rbx]
        call PRINT_FLOAT
        add rbx, 8
        dec r12
        test r12, r12
        jnz PW_LBG_H
    ret
    ; print output weights
    lea rbx, [OUTPUT_WEIGHTS]
    mov r12, OUTPUT_NCT
    PW_LBG_O:
        movsd xmm0, qword [rbx]
        call PRINT_FLOAT
        add rbx, 8
        dec r12
        test r12, r12
        jnz PW_LBG_O
    ret

PRINT_BIASES:
    ; print hidden biases
    lea rbx, [HIDDEN_BIASES]
    mov r12, HIDDEN_CT
    PB_LBG_H:
        movsd xmm0, qword [rbx]
        call PRINT_FLOAT
        add rbx, 8
        dec r12
        test r12, r12
        jnz PB_LBG_H
    ; print output biases
    lea rbx, [OUTPUT_BIASES]
    mov r12, OUTPUT_CT
    PB_LBG_O:
        movsd xmm0, qword [rbx]
        call PRINT_FLOAT
        add rbx, 8
        dec r12
        test r12, r12
        jnz PB_LBG_O
    ret

INIT_WEIGHTS:
    ; init all hidden weights
    lea rbx, [HIDDEN_WEIGHTS]
    mov r12, HIDDEN_NCT     ; 23520
    IW_LBG_H:
        call RANDOM_FLOAT
        movsd [rbx], xmm0
        add rbx, 8
        dec r12
        test r12, r12
        jnz IW_LBG_H
    ; init all output weights
    lea rbx, [OUTPUT_WEIGHTS]
    mov r12, OUTPUT_NCT     ; 300
    IW_LBG_O:
        call RANDOM_FLOAT
        movsd [rbx], xmm0
        add rbx, 8
        dec r12
        test r12, r12
        jnz IW_LBG_O
    ret

INIT_BIASES:
    ; init hidden biases
    lea rbx, [HIDDEN_BIASES]
    mov r12, HIDDEN_CT
    IB_LBG_H:
        call RANDOM_FLOAT
        movsd [rbx], xmm0
        add rbx, 8
        dec r12
        test r12, r12
        jnz IB_LBG_H
    ; init output biases
    lea rbx, [OUTPUT_BIASES]
    mov r12, OUTPUT_CT
    IB_LBG_O:
        call RANDOM_FLOAT
        movsd [rbx], xmm0
        add rbx, 8
        dec r12
        test r12, r12
        jnz IB_LBG_O
    ret

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
    push rbx
    push rcx
    push rdx
    mov rax, [XORSHI_0]
    add rax, [XORSHI_3]     ; return x[0] + x[3]

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