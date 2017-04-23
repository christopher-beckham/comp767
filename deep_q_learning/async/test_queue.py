import multiprocessing

queue = multiprocessing.Queue(maxsize=10)

for i in range(10):
    queue.put(i+1)
    
while not queue.empty():
    print queue.get()
