import numpy as np
import multiprocessing
import datetime
import time



def myfunc(par, pro):


    my_new = np.array(par * 16)

    time.sleep(2)

    print(f"{pro} ended in {datetime.datetime.now()}")




    pipe.send(13 * par)
    pipe.close()



pipes = []
results = []
processes = []
multiprocessingg = 'pool'
processes_limit = 2






# if multiprocessingg == 'process':



#     for i in range(3):
#         pipes.append(multiprocessing.Pipe())

p1 = multiprocessing.Process(target=myfunc, kwargs={'par': np.array(np.random.randn(100000, 100)), 'pro': 'pro1'})
p2 = multiprocessing.Process(target=myfunc, kwargs={'par': np.array(np.random.randn(100000, 3)) * 2, 'pro': 'pro2'})
p3 = multiprocessing.Process(target=myfunc, kwargs={'par': np.array(np.random.randn(100000, 200)) * 2, 'pro': 'pro3'})

ps = [p1, p2, p3]
#     # ps = [p1]

#     for i in ps:
#         i.start()

#     for i in pipes:
#         try:
#             results.append(i[0].recv())
#         except Exception:
#             pass

# elif multiprocessing == 'pool':

#     pool.apply_async(predictit.main_loop.train_and_predict, (), predict_parameters, callback=return_result)

# else:pool


if multiprocessingg == 'pool':
    pool = multiprocessing.Pool(3)

    # It is not possible easy share data in multiprocessing, so results are resulted via callback function
    def return_result(result):
        results.append(result)

for i in ps:
    pool.apply_async(myfunc, (), {'par': np.array(np.random.randn(100000, 100)), 'pro': 'pro1'}, callback=return_result)

if multiprocessingg == 'pool':
    pool.close()
    pool.join()



try:
    print(results[0][0, 0])
    print(results[1][0, 0])
    print(results[2][0, 0])


except Exception:
    pass
