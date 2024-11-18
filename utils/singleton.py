def singleton(cls):
    instances = {}

    def getinstance(*args, **kargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kargs)
        return instances[cls]

    return getinstance
