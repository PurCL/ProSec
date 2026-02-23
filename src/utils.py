def parse_code_block(code_in, lang="java", only_capture_succ=False):
    code_out = []
    is_in_block = False
    for line in code_in.split("\n"):
        if "```%s" % lang in line and not is_in_block:
            is_in_block = True
        elif line.strip() == "```" and is_in_block:
            is_in_block = False
        elif is_in_block:
            code_out.append(line)
    joined_ret = "\n".join(code_out)
    if joined_ret.strip() == "" and not only_capture_succ:
        return code_in.strip()
    return joined_ret


print()
