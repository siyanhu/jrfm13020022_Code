from datetime import datetime

def dateToInterval(startDate, towardDate):
    d1 = datetime.fromisoformat(startDate)
    d2 = datetime.fromisoformat(towardDate)
    interval = d2 - d1
    return interval.days

def processDateListToDict(dateList):
    if (len(dateList) <= 0):
        return [-1]
    firstDate = dateList[0]
    index = 0
    intervals = dict()
    while (index < len(dateList)):
        towardDate = dateList[index]
        interval = dateToInterval(firstDate, towardDate)
        intervals[towardDate] = interval
        index = index + 1
    return intervals

def processDateListToList(dateList):
    if (len(dateList) <= 0):
        return [-1]
    firstDate = dateList[0]
    index = 0
    intervals = list()
    while (index < len(dateList)):
        towardDate = dateList[index]
        interval = dateToInterval(firstDate, towardDate)
        intervals.append(interval)
        index = index + 1
    return intervals