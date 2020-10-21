from operator import truediv


def to_pickle(obj, file_name):
    import pickle
    import os

    dir_name = os.path.dirname(file_name)
    if not os.path.exists(dir_name):
        print(f'Directory {dir_name} does not exist.. ')
        print(f'.. create direcotry')
        os.makedirs(dir_name)

    with open(file_name, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        return(True)

    return(False)


def from_pickle(file_name):
    import pickle
    import os

    if os.path.isfile(file_name) and os.access(file_name, os.R_OK):
        print("File exists and is readable")
    else:
        print("Either file is missing or is not readable")

    with open(file_name, 'rb') as f:
        obj = pickle.load(f)
        return(obj)


def print_sys_stats():
    import psutil
    cpu = psutil.cpu_percent()
    mem = psutil.virtual_memory()

    print(f'cpu: {cpu}%, mem: {mem.available/1000000:,.2f}, percent used: {mem.percent}%, available: {mem.available * 100 / mem.total:.2f}%')
    return(True)