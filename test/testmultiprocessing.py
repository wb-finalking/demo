import multiprocessing
from time import sleep
import Queue

def calculate(process_name, tasks, results):
    print('[%s] evaluation routine starts'% process_name)

    while True:
        new_value = tasks.get()
        print(new_value)
        if new_value <0:
            print('[%s] evaluation routine quits' % process_name)
            results.put(-1)
            break
        else:
            compute = 1 * 1
            print(dict)

            print('[%s] received value: %s' % (process_name, new_value))
            print('[%s] calculated value: %i' % (process_name, compute))

            results.put(compute)

    return


if __name__ == '__main__':
    manager = multiprocessing.Manager()

    tasks = manager.Queue()
    results = manager.Queue()
    q = Queue.Queue(-1)
    r = Queue.Queue(-1)
    dic = manager.dict()
    dict = {'1':1, '2':3}

    num_processes = 4
    pool = multiprocessing.Pool(processes=num_processes)
    processes = []

    for i in range(num_processes):
        process_name = 'P%i' % i
        new_process = multiprocessing.Process(target=calculate, args=(process_name,tasks,results))
        processes.append(new_process)
        new_process.start()

    task_list = [43,1,780,256,142,68,183,334,325,3]
    for single_task in task_list:
        tasks.put('{}'.format(single_task))
        q.put('{}'.format(single_task))

    sleep(5)

    for i in range(num_processes):
        tasks.put(-1)
        q.put(-1)

    num_finished_processes = 0

    while True:

        new_result = results.get()

        if new_result == -1:

            num_finished_processes += 1
            if num_finished_processes == num_processes:
                break
        else:
            print('Result:' + str(new_result))