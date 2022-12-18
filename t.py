# import csv


# with open("submission.csv", 'r') as file:
#     csvreader = csv.reader(file)
#     for row in csvreader:
#         # file_name =  row[0].split('/')
#         # pred = row[1]
#         # print(file_name, pred)
#         print(row)
#         exit()


import pandas as pd
import cv2
# read the contents of csv file
dataset = pd.read_csv("submission.csv")
# print(dataset.shape)
for  i in range(dataset.shape[0]):
    if  dataset.loc[i]['filename'].split('/')[0]=='task3':
        file_name =  dataset.loc[i]['filename'].split('/')[1]
        label =  dataset.loc[i]['label']
        img = cv2.imread("E:/ml/hw5/test/task3/"+file_name)
        # print(img.shape)
        print(i, ':', label)
        cv2.imshow('s', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
#     exit()