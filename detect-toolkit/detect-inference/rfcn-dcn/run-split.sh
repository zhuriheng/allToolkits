split -l 245100 qpulp_origin_20180227_urllist.txt -d -a 4 qpulp_origin_20180227_urllist-

将 文件 BLM.txt 分成若干个小文件，每个文件2482行(-l 2482)，文件前缀为BLM_ ，
系数不是字母而是数字（-d），后缀系数为四位数（-a 4）


split -l 508974 normal_180318_derep_urlPath.lst -d -a 4 normal_180318-