import time

def autotrain(start_epoch = 0,max_time = 3600*9,save_each_epoch = True):
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
                print('epochtime:{},epoch:{}'.format(epochtime,start_epoch))
                if epochtime > max_epochtime:
                    max_epochtime = epochtime
                if rest_time < max_epochtime:
                    net.save_network(start_epoch)
                    break
                if save_each_epoch:
                    net.save_network(start_epoch)
                start_epoch += 1
            return res
        return wrapper
    return decorator

