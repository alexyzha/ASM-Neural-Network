section .data
    hw db 'Hello, World!', 0xA      ; 0xA = newline

section .text
    global _start                   ; define main entry point

_start:
    call print_string

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