import os
import cv2
import numpy as np


def load_cifar100():
    # 데이터셋 경로 설정
    train_folder = './CIFAR100/train'
    test_folder = './CIFAR100/test'
    class_file = './classes.txt'

    # 클래스 매핑 정보 로드
    class_mapping = {}
    with open(class_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split('\n')[0]
            line = line.split(',')
            class_mapping[line[1]] = int(line[0])
    print(class_mapping)

    # 이미지와 라벨을 저장할 리스트 생성
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    # 학습용 데이터셋 구성
    for filename in os.listdir(train_folder):
        if filename.endswith('.png'):
            img_path = os.path.join(train_folder, filename)
            label = img_path.split('_', maxsplit=1)[1]
            label = label.split('.')[0]
            label = int(class_mapping[label])
            
            # 이미지 로드
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # 필요에 따라 이미지 전처리를 수행할 수 있습니다.
            # 예: img = cv2.resize(img, (width, height))
            
            # 데이터셋에 이미지와 라벨 추가
            x_train.append(img)
            y_train.append(label)

    # 검증용 데이터셋 구성
    for filename in os.listdir(test_folder):
        if filename.endswith('.png'):
            img_path = os.path.join(test_folder, filename)
            label = img_path.split('_', maxsplit=1)[1]
            label = label.split('.')[0]
            label = int(class_mapping[label])
            # 이미지 로드
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # 필요에 따라 이미지 전처리를 수행할 수 있습니다.
            # 예: img = cv2.resize(img, (width, height))
            
            # 데이터셋에 이미지와 라벨 추가
            x_test.append(img)
            y_test.append(label)

    # 리스트를 넘파이 배열로 변환
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    # 데이터셋의 크기 출력
    print('학습용 데이터셋:', x_train.shape, y_train.shape)
    print('검증용 데이터셋:', x_test.shape, y_test.shape)

    return (x_train, y_train), (x_test, y_test)