import numpy as np


def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')

def NDCGatK_r(test_data, r, k):
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.
    # return np.sum(ndcg)
    return ndcg

def test_one_batch(sorted_items, groundTrue, k):
    sorted_items = sorted_items
    groundTrue = groundTrue
    r = getLabel(groundTrue, sorted_items)
    return NDCGatK_r(groundTrue, r, k)



# ====== 测试用例 ======
# ====== 测试用例 ======

# 假设 batch 有 2 个用户
sorted_items = [
    [101, 102, 103, 104, 105],  # 推荐给第1个用户的top-k排序结果
    [201, 202, 203, 204, 205]   # 推荐给第2个用户的top-k排序结果
]

groundTrue = [
    [101, 103, 106],  # 第1个用户真正感兴趣的items
    [202, 204]        # 第2个用户真正感兴趣的items
]

k = 3

ndcg_score = test_one_batch(sorted_items, groundTrue, k)
print(f"NDCG@{k} for the batch:", ndcg_score)