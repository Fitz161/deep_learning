import os
path = r'D:\CloudMusic'


def files(path):
    with open("1.txt", 'w', encoding='utf8') as f:
        for root, dirs, files in os.walk(path):
            for file_name in files:
                f.write(file_name + '\n')
for i in range(10): #生成器generator
    print(i)
for root, dirs, files in os.walk("D:"): #迭代器iterator
    # os.walk 返回一个只有一个元素（元组）的迭代器，for循环一次就将元组取出来了，然后尝试将tuple拆成三个元素并赋值给对应对象
    print(root, dirs, files)
for item in os.walk("D:"):
    print(item)
#os.walk 返回一个只有一个元素（元组）的迭代器，for循环一次就将元组取出来了，所以item为一个tuple，循环也只会迭代一次

# D: [] ['1.txt', 'concat.py', 'concat2.py', 'concat3.py', 'del_ncm_files.py', 'test.py']
# ('D:', [], ['1.txt', 'concat.py', 'concat2.py', 'concat3.py', 'del_ncm_files.py', 'test.py'])