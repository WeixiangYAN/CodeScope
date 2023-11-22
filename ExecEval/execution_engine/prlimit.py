from resource_limit import ResourceLimits


def process_args(arg: str, val: int) -> str:
    if arg.startswith("_"):
        arg = arg[1:]

    return f"--{arg}={val}"


def get_prlimit_str(limits: ResourceLimits, timelimit_factor: int = 1) -> str:
    temp = []
    for field in limits.fields():
        if field == "cpu":
            continue
        val = getattr(limits, field)
        temp.append(process_args(field, val))

    return f"prlimit {' '.join(temp)}"
