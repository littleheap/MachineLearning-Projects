from math import sqrt

# 电影打分数据字典
critics = {
    'Lisa Rose': {
        'Lady in the Water': 2.5,
        'Snakes on a Plane': 3.5,
        'Just My Luck': 3.0,
        'Superman Returns': 3.5,
        'You, Me and Dupree': 2.5,
        'The Night Listener': 3.0
    },
    'Gene Seymour': {
        'Lady in the Water': 3.0,
        'Snakes on a Plane': 3.5,
        'Just My Luck': 1.5,
        'Superman Returns': 5.0,
        'The Night Listener': 3.0,
        'You, Me and Dupree': 3.5
    },
    'Michael Phillips': {
        'Lady in the Water': 2.5,
        'Snakes on a Plane': 3.0,
        'Superman Returns': 3.5,
        'The Night Listener': 4.0
    },
    'Claudia Puig': {
        'Snakes on a Plane': 3.5,
        'Just My Luck': 3.0,
        'The Night Listener': 4.5,
        'Superman Returns': 4.0,
        'You, Me and Dupree': 2.5
    },
    'Mick LaSalle': {
        'Lady in the Water': 3.0,
        'Snakes on a Plane': 4.0,
        'Just My Luck': 2.0,
        'Superman Returns': 3.0,
        'The Night Listener': 3.0,
        'You, Me and Dupree': 2.0
    },
    'Jack Matthews': {
        'Lady in the Water': 3.0,
        'Snakes on a Plane': 4.0,
        'The Night Listener': 3.0,
        'Superman Returns': 5.0,
        'You, Me and Dupree': 3.5
    },
    'Toby': {
        'Snakes on a Plane': 4.5,
        'You, Me and Dupree': 1.0,
        'Superman Returns': 4.0
    }
}

'''
    欧式距离计算
'''


# 利用欧几里得计算两个人之间的相似度
def sim_distance(prefs, person1, person2):
    # 首先把这个两个用户共同拥有评过分电影给找出来，方法是建立一个字典，字典的key电影名字，电影的值就是1
    si = {}
    for item in prefs[person1]:
        if item in prefs[person2]:
            si[item] = 1
    # 如果没有共同之处
    if len(si) == 0:
        return 0
    # 有共同之处情况下，计算所有差值的平方和
    sum_of_squares = sum(
        [pow(prefs[person1][item] - prefs[person2][item], 2) for item in prefs[person1] if item in prefs[person2]])
    return 1 / (1 + sqrt(sum_of_squares))


# 欧式距离计算
print(sim_distance(critics, 'Lisa Rose', 'Claudia Puig'))  # 0.38742588672279304

'''
    皮尔逊相关系数
'''


# 返回两个人的皮尔逊相关系数
def sim_pearson(prefs, p1, p2):
    si = {}
    for item in prefs[p1]:
        if item in prefs[p2]:
            si[item] = 1
    # 得到列表元素的个数
    n = len(si)
    # 如果两者没有共同之处，则返回0
    if n == 0:
        return 1
    # 对共同拥有的物品的评分，分别求和
    sum1 = sum([prefs[p1][it] for it in si])
    sum2 = sum([prefs[p2][it] for it in si])
    # 求平方和
    sum1Sq = sum([pow(prefs[p1][it], 2) for it in si])
    sum2Sq = sum([pow(prefs[p2][it], 2) for it in si])
    # 求乘积之和
    pSum = sum([prefs[p1][it] * prefs[p2][it] for it in si])
    # 计算皮尔逊评价值
    num = pSum - (sum1 * sum2 / n)
    den = sqrt((sum1Sq - pow(sum1, 2) / n) * (sum2Sq - pow(sum2, 2) / n))
    if den == 0:
        return 0
    r = num / den
    return r


# 皮尔孙系数计算
print(sim_pearson(critics, 'Lisa Rose', 'Gene Seymour'))  # 0.39605901719066977

'''
    Tanimoto系数
'''


def tanimoto(a, b):
    c = [v for v in a if v in b]
    return float(len(c)) / (len(a) + len(b) - len(c))


'''
    对某一用户寻找相似用户
'''


def topMatches(prefs, person, n=5, similarity=sim_pearson):
    scores = [(similarity(prefs, person, other), other) for other in prefs if other != person]
    scores.sort()
    scores.reverse()
    return scores[0:n]


print(topMatches(critics, 'Toby', n=5))
'''
    [(0.9912407071619299, 'Lisa Rose'), 
    (0.9244734516419049, 'Mick LaSalle'), 
    (0.8934051474415647, 'Claudia Puig'), 
    (0.66284898035987, 'Jack Matthews'), 
    (0.38124642583151164, 'Gene Seymour')]
'''

'''
    利用所有人对电影的打分，然后根据不同的人的相似度，预测目标用户对某个电影的打分  
    所以函数名叫做得到推荐列表，我们会推荐预测分数较高的电影 
'''


def getRecommendations(prefs, person, similarity=sim_pearson):
    totals = {}
    simSums = {}
    for other in prefs:
        # 不用和自己比较了
        if other == person:
            continue
        sim = similarity(prefs, person, other)
        # 忽略相似度为0或者是小于0的情况
        if sim <= 0:
            continue
        for item in prefs[other]:
            # 只对自己还没看过的电影进行评价
            if item not in prefs[person] or prefs[person][item] == 0:
                # 相似度*评价值。setdefault就是如果没有就新建，如果有，就取那个item
                totals.setdefault(item, 0)
                totals[item] += prefs[other][item] * sim
                # 相似度之和
                simSums.setdefault(item, 0)
                simSums[item] += sim
    rankings = [(total / simSums[item], item) for item, total in totals.items()]
    # 返回好经过排序的列表
    rankings.sort()
    rankings.reverse()
    return rankings


print(getRecommendations(critics, 'Toby'))
'''
    [(3.3477895267131017, 'The Night Listener'), 
    (2.8325499182641614, 'Lady in the Water'), 
    (2.530980703765565, 'Just My Luck')]
'''
