def cn_to_unicode(in_str, need_str=True, debug=False):
    out = []

    for s in in_str:
        # 获得该字符的数值
        val = ord(s)
        # print(val)

        # 小于0xff则为ASCII码，手动构造\u00xx格式
        if val <= 0xff:
            hex_str = hex(val).replace('0x', '').zfill(4)
            # 这里不能以unicode_escape编码，不然会自动增加一个'\\'
            res = bytes('\\u' + hex_str, encoding='utf-8')
        else:
            res = s.encode("unicode_escape")

        out.append(res)
    
    # 调试
    if debug:
        print(out)
        print(len(out), len(out[0]), len(out[-1]))

    # 转换为str类
    if need_str:
        out_str = ''
        for s in out:
            out_str += str(s, encoding='utf-8')
        return out_str
    else:
        return out
    