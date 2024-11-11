import numpy as np
from scipy.stats import pearsonr

hours_studied = [2, 3, 5, 7, 9]
exam_scores = [50, 60, 70, 80, 90]

correlation,_ = pearsonr(hours_studied, exam_scores)
print(f'Pearson Correlation: {correlation}')

from sklearn.metrics.pairwise import cosine_similarity

doc1 = np.array([1, 1, 0, 0])
doc2 = np.array([1, 1, 1, 0])

doc1 = doc1.reshape(1, -1)
doc2 = doc2.reshape(1, -1)

cosine_sim = cosine_similarity(doc1, doc2)[0][0]
print("Cosine Similarity:", cosine_sim)

from sklearn.metrics import jaccard_score

person1 = np.array([1, 1, 0, 0])
person2 = np.array([1, 1, 1, 0])

jaccard_sim = jaccard_score(person1, person2)
print("Jaccard Similarity:", jaccard_sim)

from scipy.spatial.distance import euclidean

point1 = np.array([1, 2])
point2 = np.array([4, 6])

euclidean_dist = euclidean(point1, point2)
print("Euclidean Distance:", euclidean_dist)

from scipy.spatial.distance import cityblock

point1 = np.array([1, 2])
point2 = np.array([4, 6])

manhattan_dist = cityblock(point1, point2)
print("Manhattan Distance:", manhattan_dist)
