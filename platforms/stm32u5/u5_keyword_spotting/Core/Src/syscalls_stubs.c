#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>

/*
 * Minimal syscall stubs for newlib/newlib-nano.
 * These silence linker warnings from libc_nano.
 * They are marked weak so they do not conflict if CubeMX later generates real versions.
 */

__attribute__((weak)) int _close(int file)
{
    (void)file;
    errno = ENOSYS;
    return -1;
}

__attribute__((weak)) int _fstat(int file, struct stat *st)
{
    (void)file;
    if (st) {
        st->st_mode = S_IFCHR;
    }
    return 0;
}

__attribute__((weak)) int _getpid(void)
{
    return 1;
}

__attribute__((weak)) int _kill(int pid, int sig)
{
    (void)pid;
    (void)sig;
    errno = EINVAL;
    return -1;
}

__attribute__((weak)) int _isatty(int file)
{
    (void)file;
    return 1;
}

__attribute__((weak)) int _lseek(int file, int ptr, int dir)
{
    (void)file;
    (void)ptr;
    (void)dir;
    return 0;
}

__attribute__((weak)) int _read(int file, char *ptr, int len)
{
    (void)file;
    (void)ptr;
    (void)len;
    return 0;
}
