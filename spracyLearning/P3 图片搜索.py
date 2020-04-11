# import requests
# path = "/Users/faguangnanhai/Downloads/abc.jpg"
# url = "http://img0.dili360.com/ga/M00/48/F7/wKgBy1llvmCAAQOVADC36j6n9bw622.tub.jpg"
# r = requests.get(url)
# r.status_code
#
# with open(path,'wb') as f:
#     f.write(r.content)
#     f.close()
#

import requests
import os
url = "http://img0.dili360.com/ga/M00/48/F7/wKgBy1llvmCAAQOVADC36j6n9bw622.tub.jpg"
root = "/Users/faguangnanhai/Downloads/"
path = root + url.split('/')[-1]
try:
    if not os.path.exists(root):
        os.mkdir(root)
    if not os.path.exists(path):
        r = requests.get(url)
        r.raise_for_status()
        with open(path, 'wb') as f:
            f.write(r.content)
            f.close()
            print("文件保存成功")
    else:
        print("文件已存在")
except:
    print("爬取失败")
# r.content表示返回内容的二进制形式， # 图片是以二进制形式存储的








