section .data
    hw db 'Hello, World!', 0xA, 0
    buffer db '00000000000000000000', 0xA, 0
    EULER_CONST dq 2.718281828459045
section .text
    global _start                   ; define main entry point
    fld qword [EULER_CONST]
    fstp st0
_start:
    call print_string

    mov rdi, 100
    mov rsi, buffer
    call print_int

    mov rax, 60                     ; syscall exit
    xor rdi, rdi                    ; return 0
    syscall                         ; make exit syscall

print_string:
    mov rax, 1                      ; syscall write
    mov rdi, 1                      ; file desc = stdout, rdi passed to caller
    mov rsi, hw                     ; p->string
    mov rdx, 14                     ; # bytes to write
    syscall                         ; make write syscall
    ret

print_int:                          ; rdx contains [div rbx], rdx = rax % rbx, rax = rax / rbx
    mov rax, rdi                    ; numerator
    mov rbx, 10                     ; denominator
    lea rcx, [rsi+19]               ; p->str in rsi, num in rdi, assumes rsi = buffer

    pint_loop_start:
        cmp rax, 0                  ; end loop if no digits left
        je pint_loop_end
        xor rdx, rdx
        div rbx
        add dl, '0'
        mov [rcx], dl
        dec rcx
        jmp pint_loop_start

    pint_loop_end:
    ; rcx holds int (buffer)
    ; need string len

    inc rcx
    mov rdi, rcx
    call strlen
    ; len in rax
    
    mov rdx, rax
    mov rsi, rcx
    mov rdi, 1
    mov rax, 1
    syscall
    ret

strlen:                             ; assumes there will be a null terminator
    xor rax, rax
    strlen_loop_start:
        cmp byte [rdi], 0
        je strlen_loop_end
        inc rax
        inc rdi
        jmp strlen_loop_start

    strlen_loop_end:
    ret