import requests
r = requests.get("https://python123.io/ws/demo.html")
demo = r.text
from bs4 import BeautifulSoup
soup = BeautifulSoup(demo,"html.parser")
# for sibling in soup.a.next_siblings: # 遍历后续节点
#     print(sibling)
# for sibling in soup.a.previous_siblings: # 遍历前续节点
#     print(sibling)
import re
# for tag in soup.find_all(re.compile('p')):
#     print(tag.name)
print(soup.find_all(string = re.compile('python')))