from fetch_training_data import *
import random
patterns = fetch_data()
w = list()
def random_weights(patterns,max_clusters,vector_length):
    # print(patterns)
    # print(max_clusters)
    flag = 0
    while(flag<max_clusters):
        random_instance = random.choice(patterns)
        if random_instance not in w:
            flag = flag + 1
            w.append(random_instance)
        else:
            continue
    # for x in range(max_clusters):
    #     random_instance = random.choice(patterns)
    #     # print(random_instance)
    #     if random_instance not in w:
    #         # weight = [y for y in random_instance]
    #         w.append(random_instance)
    #         print(x,len(w))
    #     else:
    #         random_instance = random.choice(patterns)
    #         w.append(random_instance)
    #         print(x, len(w))

    # print(len(w))
    return w
