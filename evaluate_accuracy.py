import matplotlib.pyplot as plt
import tensorflow as tf

# 현재 디렉토리의 utils 폴더 안에 위치시켜주세요.
# 꼭 classes.txt를 현재 디렉토리에 위치시켜주세요.
# tensorflow-addons 와 albumentations가 필요합니다.(모듈 임포트시 필요)
from utils.Network import Build_Network
from utils.SAM import train_step_sam
from utils.load_cifar100 import load_cifar100
from utils.Generator import DataGenerator
from utils.Metrices import recall, precision, f1score

network = Build_Network(100, booster=True)
model = tf.keras.Sequential()
model.add(network)
model.add(tf.keras.layers.GlobalAveragePooling2D())
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(100, activation='softmax'))

class MyModel(tf.keras.Model):
    def __init__(self,num_classes=100):
        super(MyModel, self).__init__(name='model')
        self.base=model
    def call(self, x):
        x = self.base(x)
        return x
    def train_step(self, data):
        return train_step_sam(self, data, rho=0.05)
    
model = MyModel(model)
model.build((None, 112, 112, 3))

model.summary()
model.load_weights('CIFAR100-weights.h5')

model.compile(
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy', precision, recall, f1score])

(x_train, y_train), (x_test, y_test) = load_cifar100()

x_test = x_test.astype('float32') / 255.0
y_test = tf.keras.utils.to_categorical(y_test, 100)

valid_data_generator = DataGenerator(x_test, y_test, augment=False)
_loss, _acc, _precision, _recall, _f1score = model.evaluate_generator(generator = valid_data_generator, verbose = 1)

print('loss: {:.4f}, accuracy: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1score: {:.4f}'.format(_loss, _acc, _precision, _recall, _f1score))