from functools import wraps

def do_log(func):
    @wraps(func)
    def wrapper(*args, **kw):
        if func.__name__ == "debug":
            msg = "debug {}".format(args[0])
        elif func.__name__ == "info":
            msg = "info {}".format(args[0])
        else:
            msg =  "unknown {}".format(args[0])
        return func(msg, **kw)
    return wrapper

@do_log
def debug(msg):
    print(msg)

@do_log
def info(msg):
    print(msg)

if __name__ == "__main__":
    debug("123")
    info("abc")