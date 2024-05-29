from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
import time
import os

os.environ["LOKY_MAX_CPU_COUNT"] = '8'

start_time = time.time()

digits = load_digits()
scaled = scale(digits.data)

kmeans = KMeans(init='random', n_clusters=3, n_init=10)
kmeans.fit(scaled)

cluster = kmeans.labels_

end_time = time.time()
execution_time = end_time - start_time
print(f'Run time: {round(execution_time, 5)} seconds')

ari = adjusted_rand_score(digits.target, cluster)
ami = adjusted_mutual_info_score(digits.target, cluster)

print('ARI:', round(ari, 4))
print('AMI:', round(ami, 4))
