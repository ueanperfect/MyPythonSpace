import matplotlib.pyplot as plt
import numpy as np
def f(t):
    s1 = np.sin(2*np.pi*t)
    e1 = np.exp(-t)
    return np.multiply(s1, e1)
t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)
fig, ax = plt.subplots()
plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')
plt.text(3.0, 0.6, 'f(t) = exp(-t) sin(2 pi t)')
ttext = plt.title('Jizhixueyuan Python 666 !')
ytext = plt.ylabel('Damped oscillation') # 提示:纵坐标
xtext = plt.xlabel('time (s)') # 提示:横坐标
plt.setp(ttext, size='large', color='r', style='italic')
plt.setp(xtext, size='medium', name=['Courier', 'DejaVu Sans Mono'], weight='bold', color='g') # 提示:绿色
plt.setp(ytext, size='medium', name=['Helvetica', 'DejaVu Sans'], weight='light', color='b') # 提示:蓝色
plt.show() # 提示:显示图片