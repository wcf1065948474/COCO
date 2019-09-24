import time

def autotrain(start_epoch = 0,max_time = 3600*9):
    def decorator(func):
        max_epochtime = 0
        rest_time = max_time
        def wrapper(net,dataloader):
            nonlocal rest_time
            nonlocal max_epochtime
            nonlocal start_epoch
            if start_epoch != 0:
                net.load_network(start_epoch-1)
            while True:
                starttime = time.time()
                res = func(net,dataloader)
                epochtime = time.time()-starttime
                rest_time -= epochtime
                start_epoch += 1
                if epochtime > max_epochtime:
                    max_epochtime = epochtime
                if rest_time < max_epochtime:
                    break
            net.save_network(start_epoch)
            return res
        return wrapper
    return decorator

