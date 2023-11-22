import errno

import seccomp


def make_filter(blocked_syscalls: list[str] | None):
    if blocked_syscalls is None:
        return None

    filter = seccomp.SyscallFilter(defaction=seccomp.ALLOW)
    for syscall in blocked_syscalls:
        filter.add_rule(seccomp.ERRNO(errno.EACCES), syscall)
    return filter
