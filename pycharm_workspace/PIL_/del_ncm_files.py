import os
path = r'D:\CloudMusic'
def delete_ncm_files(path):
    for root, dirs, files in os.walk(path):
        print(root, dirs, files)
        exit()
        for file_name in files:
            file_path = os.path.join(root, file_name)
            if file_name[-3:] == 'ncm':
                print("delete", file_name)
                os.remove(file_path)
delete_ncm_files(path)