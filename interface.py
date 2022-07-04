import numpy as np


"""
data: 초기에 랜덤으로 던질 질문이 담긴 pandas dataframe
init: 처음인지 알려주기. 처음이라면 질문을 던지고 아니라면 사용자의 답을 받아옴(input)
"""
def interface(data, init=False):
    if init:
        idx = np.random.randint(0, len(data) - 1, size=1)
        print("Chatbot: ", data.iloc[idx])

    query = input("You say: ")

    if query == "exit":
        exit()

    return query


