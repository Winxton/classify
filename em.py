import random
import matplotlib.pyplot as plt
import numpy as np

SLOEPNEA = 0
FORIENNDITIS = 1
DEGAR = 2
TRIMONO = 3
DUNNETTS = 4

class ExpectationMaximize(object):

    def __init__(self, delta):
        self.delta = delta

        # Initial probabilities        
        self.CPT = [
            [[0.05,0.30,0.30],  # Sloepnea (Given T and D, first row T=false)
             [0.01,0.05,0.05]], # (Dunnetts = none, mild, severe)
            [0.02,0.30,0.10],   # Foriennditis
            [0.02,0.10,0.30],   # Degar spots
            [0.1],              # TRIMONO-HT/S
            [0.5, 0.25, 0.25],  # Dunetts (none, mild, severe)
        ]

        # randomize CPTs
        self.randomize(self.CPT)

    def randomize(self, lst):
        for ind, item in enumerate(lst):
            if isinstance(item, float):
                pTrue = lst[ind] + random.uniform(0, self.delta)
                pFalse = 1 - lst[ind] + random.uniform(0, self.delta)
                lst[ind] = pTrue / (pTrue + pFalse)
            else:
                self.randomize(item)

    """ 
    P(D, TRIM, SLOEP, FORI, DEGAR)
    sloepnea: 0, 1
    foriennditi: 0, 1
    degar: 0,1
    trimono: 0, 1
    dunetts: 0, 1, 2 (none, mild, severe)
    """
    def jointProbability(self, sloepnea, foriennditis, degar, trimono, dunetts):
        mult = 1.0
        mult *= self.CPT[SLOEPNEA][trimono][dunetts] if sloepnea == 1 else (1 - self.CPT[SLOEPNEA][trimono][dunetts])
        mult *= self.CPT[FORIENNDITIS][dunetts] if foriennditis == 1 else (1 - self.CPT[FORIENNDITIS][dunetts])
        mult *= self.CPT[DEGAR][dunetts] if degar == 1 else (1 - self.CPT[DEGAR][dunetts])
        mult *= self.CPT[TRIMONO][0] if trimono else (1 - self.CPT[TRIMONO][0])
        mult *= self.CPT[DUNNETTS][dunetts]
        return mult

    """ P(D| TRIM, SLOEP, FORI, DEGAR)
    """
    def condProbability(self, sloepnea, foriennditis, degar, trimono, dunetts):
        sump = 0 # Normalize over dunetts
        for i in range(0,3):
            sump += self.jointProbability(sloepnea, foriennditis, degar, trimono, i)
        return self.jointProbability(sloepnea, foriennditis, degar, trimono, dunetts)/sump

    def _queryMatchesData(self, data, query):
        return (data[SLOEPNEA]==query[SLOEPNEA] or query[SLOEPNEA] == -1) \
            and (data[FORIENNDITIS]==query[FORIENNDITIS] or query[FORIENNDITIS] == -1) \
            and (data[DEGAR]==query[DEGAR] or query[DEGAR] == -1) \
            and (data[TRIMONO]==query[TRIMONO] or query[TRIMONO] == -1) \
            and (data[DUNNETTS]==query[DUNNETTS] or query[DUNNETTS] == -1)

    def expandedTableWeights(self, expandedData, query):
        # Query [sloepnea, foriennditis, degar, trimono, dunetts]
        # -1 => Any value
        return sum(data[6] for data in expandedData if self._queryMatchesData(data, query))

    def learn(self, traindata):
        # Iterate until the joint probability sum does not change
        jointProbabilitySum = 0

        while (True):
            oldjointProbabilitySum = jointProbabilitySum

            # Expand the data [sloep, for, degar, trimono, dunetts, joint probability, conditional probability]
            expandedData = []
            for data in traindata:
                if (data[DUNNETTS] == -1):
                    for dunetts in range(0,3):
                        jointProbability = self.jointProbability(data[0], data[1], data[2], data[3], dunetts)
                        condProbability = self.condProbability(data[0], data[1], data[2], data[3], dunetts)
                        expandedData.append( data[:4] + [dunetts, jointProbability, condProbability] )
                else:
                    # Observed data, put probability at 1
                    expandedData.append( data + [0, 1])

            total = len(traindata)
            dunnettsWeights = [ self.expandedTableWeights(expandedData, [-1,-1,-1,-1, 0]),
                                self.expandedTableWeights(expandedData, [-1,-1,-1,-1, 1]),
                                self.expandedTableWeights(expandedData, [-1,-1,-1,-1, 2])]

            # Update guesses for dunnetts
            for i, weight in enumerate(dunnettsWeights):
                self.CPT[DUNNETTS][i] = weight / total
            
            # Update guesses for trimono-ht/s
            self.CPT[TRIMONO][0] = self.expandedTableWeights(expandedData, [-1,-1,-1, 1,-1])/total

            # Update guesses for foriennditis
            for i, weight in enumerate(dunnettsWeights):
                # #{Foriennditis = True, Dunnetts = {none, mild, severe}} / #{Dunnetts = {none, mild, severe}}
                self.CPT[FORIENNDITIS][i] = self.expandedTableWeights(expandedData, [-1,1,-1,-1, i]) / weight

            # Update guesses for degar spots
            for i, weight in enumerate(dunnettsWeights):
                # #{Degar = True, Dunnetts = {none, mild, severe}} / #{Dunnetts = {none, mild, severe}}
                self.CPT[DEGAR][i] = self.expandedTableWeights(expandedData, [-1,-1,1,-1, i]) / weight

            # Update guesses for sloepnea
            # #{Sloepnea = True, Dunnetts = {none, mild, severe}, Trimono = { True, False }} / 
            # #{Dunnetts = {none, mild, severe}, Trimono = { True, False }}}
            for trimono in range(0,2):
                for dunnetts in range(0,3):
                    self.CPT[SLOEPNEA][trimono][dunnetts] = \
                        self.expandedTableWeights(expandedData, [1,-1,-1,trimono,dunnetts]) \
                        / self.expandedTableWeights(expandedData, [-1,-1,-1,trimono,dunnetts])

            jointProbabilitySum = sum(data[5] for data in expandedData)
            
            if (abs(jointProbabilitySum - oldjointProbabilitySum) < 0.01):
                break

    """ Returns prediction of 0,1,or 2
    """
    def _classifyOne(self, data):
        results = [self.condProbability(data[0], data[1], data[2], data[3], i) for i in range(0,3)]
        return results.index(max(results))

    """ Returns the % of correct predictions
    """
    def classify(self, testdata):
        correct = 0
        for data in testdata:
            classification = self._classifyOne(data[:4])
            correct += 1 if classification == data[4] else 0

        return 1.0*correct / len(testdata)

def plot(beforeMeans, beforeStd, afterMeans, afterStd, deltas):
    N = len(beforeMeans)

    ind = np.arange(N)  # the x locations for the groups
    width = 0.35       # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, beforeMeans, width, color='r', yerr=beforeStd)
    rects2 = ax.bar(ind + width, afterMeans, width, color='g', yerr=afterStd)

    # add some text for labels, title and axes ticks
    ax.set_xlabel('Noise')
    ax.set_ylabel('% Correct')
    ax.set_title('Prediction Accuracies')
    ax.set_xticks(ind + width)
    ax.set_xticklabels(deltas)

    ax.legend((rects1[0], rects2[0]), ('Before EM', 'After EM'))

    def autolabel(rects):
        # attach some text labels
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                    '{}'.format(height),
                    ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    plt.show()

def runEMMultiple(times, delta, traindata, testdata):
    resultsBeforeEM = []
    resultsAfterEM = []

    for i in range(0, times):
        print "Delta {}, iteration {}".format(delta, i)
        em = ExpectationMaximize(delta)
        resultBefore = em.classify(testdata)

        em.learn(traindata)
        resultAfter = em.classify(testdata)

        resultsBeforeEM.append(resultBefore)
        resultsAfterEM.append(resultAfter)

    before = np.array(resultsBeforeEM)
    after = np.array(resultsAfterEM)

    return np.mean(before), np.std(before), np.mean(after), np.std(after)

if __name__ == '__main__':
    f = open('traindata.txt')
    traindata = []
    for line in f.readlines():
        traindata.append([int(i) for i in line.strip().split(' ')])

    f = open('testdata.txt')
    testdata = []
    for line in f.readlines():
        testdata.append([int(i) for i in line.strip().split(' ')])

    beforeMeans = []
    beforeStds = []
    afterMeans = []
    afterStds = []
    deltas = []

    for i in range(0,2):
        delta = 0.2 * i
        meanbefore, stdbefore, meanafter, stdafter = runEMMultiple(3, delta, traindata, testdata)
        beforeMeans.append(meanbefore)
        beforeStds.append(stdbefore)
        afterMeans.append(meanafter)
        afterStds.append(stdafter)
        deltas.append(delta)

    plot(beforeMeans, beforeStds, afterMeans, afterStds, deltas)
