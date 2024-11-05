; nasm -f elf64 network.asm -o network.o && gcc -m64 network.o -o network -nostartfiles -no-pie -lc
section .data
    ; file i/o
    FILENAME        db    'file.txt', 0
    TRAINFILE       db    'data/text/train.txt', 0
    VALIDFILE       db    'data/text/validtion.txt', 0
    TESTFILE        db    'data/text/test.txt', 0
    READONLY        equ   0
    WRITEONLY       equ   1
    READ            db    'r', 0
    READWRITE       equ   2
    ; cstd i/o
    DOUBLE_PERC     db    '%lf', 0xA, 0
    FLOAT_PERC      db    '%f', 0xA, 0
    INT_PERC        db    '%d', 0xA, 0
    STR_PERC        db    '%s', 0xA, 0
    ENDL            db    0
    ; network consts
    HIDDEN_NCT      equ   23520
    OUTPUT_NCT      equ   300
    INPUT_CT        equ   784
    HIDDEN_CT       equ   30 
    OUTPUT_CT       equ   10
    EULER           dq    2.718281828
    ALPHA           dq    3.0
    T_OR_V_CT       dq    10000
    ; for random float generation
    INT_MAX         equ   0x7FFFFFFFFFFFFFFF
    ; prng xoshiro256
    XORSHI_0        dq    0x7AF05449A5465795
    XORSHI_1        dq    0xA22CD65BFF6B2CE0
    XORSHI_2        dq    0xF46F7AAF65D601E1
    XORSHI_3        dq    0xC5DB0BDFE6A51996

section .bss
    ; weights
    HIDDEN_WEIGHTS  resq  23520
    OUTPUT_WEIGHTS  resq  300
    ; biases
    HIDDEN_BIASES   resq  30 
    OUTPUT_BIASES   resq  10
    ; i/o
    INPUTS          resq  784
    HIDDEN_OUTPUT   resq  30
    OUTPUT_OUTPUT   resq  10
    ACTUAL_VALUE    resq  10
    ; backprop
    HOUT_SIGMOID    resq  30
    OOUT_SIGMOID    resq  10
    ; file
    FILE_HANDLE     resq  1
    FILE_IN         resq  1

section .text
    global _start
    extern printf
    extern exp
    extern fscanf
    extern fopen
    extern fclose
_start:
    fninit                  ; initialize the fpu
    call INIT_WEIGHTS       ; create prand weights
    call INIT_BIASES        ; create prand biases
    call TRAIN
    mov rdi, TESTFILE
    call TEST_OR_VALIDATE
    call EXIT               ; return 0;

TEST_OR_VALIDATE:
    ; file name in rdi
    mov rsi, READ
    call fopen
    mov [FILE_HANDLE], rax
    ; get count from file line 0
    mov rdi, [FILE_HANDLE]
    mov rsi, INT_PERC
    mov rdx, FILE_IN
    sub rsp, 8              ; stack align
    call fscanf
    add rsp, 8
    mov r14, [FILE_IN]      ; 10000->0
    mov r12, 0              ; count correct
    mov r13, INT_MAX
    TV_LBG:
        call READ_IMAGE
        mov rdi, 0
        call FEED_FORWARD   ; result in rax
        lea rcx, [ACTUAL_VALUE+(rax*8)]
        mov rcx, [rcx]      ; rcx & 0x7f... = 0 then 0, excl sign bit
        test rcx, r13
        jz TV_NOINC
        inc r12
        TV_NOINC:
        ; mov rsi, r14        ; print
        ; sub rsp, 8
        ; call PRINT_INT
        ; add rsp, 8
        dec r14
        test r14, r14
        jnz TV_LBG
    sub rsp, 8              ; compute/save percent correct result
    mov [rsp], r12
    fild qword [rsp]
    fild qword [T_OR_V_CT]
    fdivrp
    fstp qword [rsp]
    movsd xmm0, qword [rsp]
    add rsp, 8
    ; call PRINT_FLOAT
    ret
        
TRAIN:
    mov rdi, TRAINFILE
    mov rsi, READ
    call fopen
    mov [FILE_HANDLE], rax  ; get file handle
    ; get count from file line 0
    mov rdi, [FILE_HANDLE]
    mov rsi, INT_PERC
    mov rdx, FILE_IN
    sub rsp, 8
    call fscanf
    add rsp, 8
    mov r13, [FILE_IN]      ; 50000->0
    TRAIN_LBG:
        call READ_IMAGE
        mov rdi, 0          ; no print ff
        call FEED_FORWARD
        call BACK_PROPAGATE
        ; mov rsi, r13
        ; sub rsp, 8
        ; call PRINT_INT
        ; add rsp, 8
        dec r13
        test r13, r13
        jnz TRAIN_LBG
    ret

READ_IMAGE:
    mov rbx, 0
    ; get result vector
    RI_LBG_A:
        mov rdi, [FILE_HANDLE]
        mov rsi, DOUBLE_PERC
        lea rdx, [ACTUAL_VALUE+(rbx*8)]
        call fscanf
        inc rbx
        cmp rbx, OUTPUT_CT
        jne RI_LBG_A
    ; get image values
    mov rbx, 0
    RI_LBG_I:
        mov rdi, [FILE_HANDLE]
        mov rsi, DOUBLE_PERC
        lea rdx, [INPUTS+(rbx*8)]
        call fscanf
        inc rbx
        cmp rbx, INPUT_CT
        jne RI_LBG_I
    ret

BACK_PROPAGATE:
    call OUTPUT_BACK
    call HIDDEN_BACK
    ret

HIDDEN_BACK:
    ; output deltas in OUTPUT_OUTPUT
    mov rbx, 0              ; 0->30
    HB_LOOP_HIDDEN:
        ; get contribution from output nodes
        lea rax, [HIDDEN_OUTPUT+(rbx*8)]
        movsd xmm0, qword [rax] 
        call SIGDER         ; sigder(hidden inputs) in xmm0
        mov rcx, 0          ; 0->10
        lea rax, [OUTPUT_WEIGHTS+(rbx*8)]
        lea rdx, [OUTPUT_OUTPUT]
        xorpd xmm1, xmm1    ; 0 out xmm1 for sum
        HB_LOOP_OUTPUT:
            movsd xmm2, qword [rdx]
            movsd xmm3, qword [rax]
            mulsd xmm2, xmm3
            addsd xmm1, xmm2
            add rdx, 8
            add rax, 240    ; skip to next node weight offset
            inc rcx
            cmp rcx, OUTPUT_CT
            jne HB_LOOP_OUTPUT
        mulsd xmm0, xmm1    
        lea rax, [HIDDEN_OUTPUT+(rbx*8)]
        movsd [rax], xmm0   ; hidden output[i] no longer used, replace with hidden delta[i]
        ; get effect on input->hidden weights
        mov rax, rbx
        mov rcx, INPUT_CT
        mov rdx, 0
        mul rcx
        lea rax, [HIDDEN_WEIGHTS+(rax*8)]
        lea rcx, [INPUTS]
        HB_LOOP_HIDDEN_UPDATE:
            movsd xmm1, qword [INPUTS]
            mulsd xmm1, xmm0
            movsd xmm2, qword [ALPHA]
            mulsd xmm1, xmm0
            movsd xmm2, qword [rax]
            subsd xmm2, xmm1
            movsd [rax], xmm0
            ; update weight^
            add rax, 8
            add rcx, 8
            inc rdx 
            cmp rdx, INPUT_CT
            jne HB_LOOP_HIDDEN_UPDATE
        ; now update node bias
        lea rax, [HIDDEN_BIASES+(rbx*8)]
        movsd xmm1, qword [rax]
        movsd xmm2, qword [ALPHA]
        mulsd xmm2, xmm0
        subsd xmm1, xmm2
        movsd [rax], xmm1
        inc rbx
        cmp rbx, HIDDEN_CT
        jne HB_LOOP_HIDDEN
    ret

OUTPUT_BACK:
    mov rbx, 0              ; 0->10, all output nodes
    OB_NODE_ITER:
        lea rax, [OOUT_SIGMOID+(rbx*8)]
        fld qword [rax]
        lea rax, [ACTUAL_VALUE+(rbx*8)]
        fld qword [rax]
        fsubp               ; estimate (st1) - actual (st0), pop st0
        lea rax, [OUTPUT_OUTPUT+(rbx*8)]
        movsd xmm0, qword [rax]
        call SIGDER
        lea rax, [OUTPUT_OUTPUT+(rbx*8)]
        movsd [rax], xmm0   ; override OUTPUT_OUTPUT[i], since its unused later
        fld qword [rax]
        fmulp               ; st0 = delta output node i
        fld qword [ALPHA]   ; alpha = learning rate
        fmulp
        lea rax, [OUTPUT_BIASES+(rbx*8)]
        fld qword [rax]     ; update bias
        fsubr               ; st0 = st1-st0
        fstp qword [rax]    ; save updated bias
        mov rax, rbx        
        mov rcx, 30         ; get weight offset
        mul rcx
        mov rcx, 0
        lea rax, [OUTPUT_WEIGHTS+(rax*8)]
        OB_WEIGHTS_ITER:
            lea rdx, [HIDDEN_OUTPUT+(rcx*8)]
            fld qword [rdx] ; load output node input[i]
            fmul            ; st0 = st0*st1 -> = delta weight[i][j]
            fld qword [rax] ; load weight[i][j]
            fsubrp          ; st1 = st0-st1, pop st0
            fstp qword [rax]
            add rax, 8
            inc rcx
            cmp rcx, 30
            jne OB_WEIGHTS_ITER
        inc rbx
        cmp rbx, 10
        jne OB_NODE_ITER
    ret

FEED_FORWARD:
    ; rdi = 0b0001 = print estimate 
    ; rdi = 0b0010 = print output

    sub rsp, 8
    mov [rsp], rdi
    call TEST1_INPUTS
    call FORWARD_PROP
    call GET_ESTIMATE
    mov rdi, [rsp]
    mov [rsp], rax

    test rdi, 1             ; rdi & 0b01
    jz FF_P_OUT
    mov rsi, rax
    call PRINT_INT
    call PRINT_NEWL

    FF_P_OUT:               ; rdi & 0b10
        test rdi, 2
        jz FF_EXIT
        ; print normal output
        mov rdi, HIDDEN_OUTPUT
        mov rsi, OUTPUT_OUTPUT
        call PRINT_OUTPUTS
        call PRINT_NEWL
        ; print sigmoid output
        mov rdi, HOUT_SIGMOID
        mov rsi, OOUT_SIGMOID
        call PRINT_OUTPUTS
        call PRINT_NEWL

    FF_EXIT:
    add rsp, 8
    ret

GET_ESTIMATE:
    push rbx
    fldz
    fld1
    fsubp                   ; st1 = 0-1 = -1, pop st0
    sub rsp, 8
    fstp qword [rsp]
    movsd xmm1, qword [rsp]
    add rsp, 8              ; xmm0 = -1.0
    mov rcx, 0
    mov rbx, 0
    GE_LBG:
        lea rax, [OOUT_SIGMOID+(rbx*8)]
        movsd xmm0, qword [rax]
        ucomisd xmm0, xmm1
        ja OVERRIDE_EST     ; jump if xmm0 > xmm1
        inc rbx
        cmp rbx, OUTPUT_CT
        jne GE_LBG
        jmp GE_END
    OVERRIDE_EST:
        movsd xmm1, xmm0      ; xmm1 holds largest value
        mov rcx, rbx
        inc rbx
        cmp rbx, OUTPUT_CT
        jne GE_LBG
    GE_END:
    pop rbx
    mov rax, rcx        ; return rax
    ret

FORWARD_PROP:
    ; input->hidden = HIDDEN_OUTPUT
    mov rbx, 0              ; count 0->30
    FP_LOOP_HNODES:
        fldz
        mov r12, 0          ; count 0->784
        mov rax, rbx        ; rbx * 784
        mov rcx, INPUT_CT
        mul rcx             ; rax = node offset
        lea rax, [HIDDEN_WEIGHTS+(rax*8)]
        lea rdi, INPUTS
        FP_LOOP_INPUTS:
            fld qword [rax]
            fld qword [rdi]
            fmulp
            faddp           ; st0 always contains sum
            inc r12
            add rax, 8
            add rdi, 8
            cmp r12, INPUT_CT
            jne FP_LOOP_INPUTS
        lea r12, [HIDDEN_BIASES+(rbx*8)]
        fld qword [r12]
        faddp
        lea r12, [HIDDEN_OUTPUT+(rbx*8)]
        fstp qword [r12]
        movsd xmm0, qword [r12]
        movsd [r12], xmm0
        lea r12, [HOUT_SIGMOID+(rbx*8)]
        call SIGMOID
        movsd [r12], xmm0   ; sig(w*i+b) in [r12], H_OUTPUT+offset
        inc rbx
        cmp rbx, HIDDEN_CT
        jne FP_LOOP_HNODES   ; add bias to st0, sigmoid all
    ; hidden->output = OUTPUT_OUTPUT
    mov rbx, 0              ; count 0->10
    FP_LOOP_HOUTPUT:
        fldz
        mov r12, 0          ; count 0->30
        mov rax, rbx        ; rbx * 30
        mov rcx, HIDDEN_CT
        mul rcx
        lea rax, [OUTPUT_WEIGHTS+(rax*8)]
        lea rdi, HOUT_SIGMOID
        FP_LOOP_HIDDENS:
            fld qword [rax]
            fld qword [rdi]
            fmulp
            faddp 
            inc r12
            add rax, 8
            add rdi, 8
            cmp r12, HIDDEN_CT
            jne FP_LOOP_HIDDENS
        lea r12, [OUTPUT_BIASES+(rbx*8)]
        fld qword [r12]
        faddp
        lea r12, [OUTPUT_OUTPUT+(rbx*8)]
        fstp qword [r12]
        movsd xmm0, qword[r12]
        movsd [r12], xmm0
        lea r12, [OOUT_SIGMOID+(rbx*8)]
        call SIGMOID
        movsd [r12], xmm0
        inc rbx
        cmp rbx, OUTPUT_CT
        jne FP_LOOP_HOUTPUT
    ret

PRINT_OUTPUTS:
    ; print hidden outputs
    ; 
    lea rbx, [rdi]
    mov r12, 0
    PO_LBG_H:
        movsd xmm0, qword [rbx]
        call PRINT_FLOAT
        add rbx, 8
        inc r12
        cmp r12, HIDDEN_CT
        jne PO_LBG_H
    ; align stack to print newline
    sub rsp, 8
    call PRINT_NEWL
    add rsp, 8
    ; print output outputs
    lea rbx, [rsi]
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

SIGDER:
    ; x in xmm0
    ; return in xmm0
    push rax
    sub rsp, 8
    call SIGMOID
    movsd [rsp], xmm0
    fld1
    fld qword [rsp]
    fsubp
    fld qword [rsp]
    fmulp
    fstp qword [rsp]
    movsd xmm0, qword [rsp]
    add rsp, 8
    pop rax
    ret

PRINT_INT:
    ; "%d" in rdi
    ; int in rsi
    ; 1 in rax?
    push rdi
    mov rdi, INT_PERC
    call printf
    pop rdi
    ret

PRINT_FLOAT:
    ; double in xmm0
    ; "%f" in rdi
    push rsi
    push rdi
    mov rdi, FLOAT_PERC
    call printf
    pop rdi
    pop rsi
    ret

PRINT_NEWL:
    ; '%s' in rdi
    ; 1 in rax
    ; endl in rsi
    push rsi
    push rdi
    push rax
    mov rsi, ENDL
    mov rdi, STR_PERC
    mov rax, 1
    call printf
    pop rax
    pop rdi
    pop rsi
    ret

EXIT:
    mov rax, 60
    xor rdi, rdi
    syscall