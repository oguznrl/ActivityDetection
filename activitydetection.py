from __future__ import print_function
from matplotlib import pyplot as plt
import numpy as np
from numpy.core.fromnumeric import mean
import pandas as pn
from scipy.sparse.sputils import matrix
import seaborn as sns
from scipy import stats

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import preprocessing

import keras
from keras.models import Sequential
from keras.layers import Dense,MaxPooling1D,Conv1D,Dropout,Reshape,GlobalAveragePooling1D
from keras.utils import np_utils

#Ornekleri normalize eder
def feature_normalize(datasets):
    mu=np.mean(datasets,axis=0)#verilerin ort
    sigma=np.std(datasets,axis=0)#verilerin standart sapmasi
    return (datasets-mu)/sigma

#Karmasiklik matrisini cizdirir
def show_confusion_matrix(validations,predictions):
    """
    validation:Verilerin mevcut degeri
    prediction:Verilerin tahmin degeri
    """
    matrix=metrics.confusion_matrix(validations,predictions)
    plt.figure(figsize=(6,4))
    sns.heatmap(matrix,
                cmap="coolwarm",
                linecolor='white',
                linewidths=1,
                xticklabels=LABELS,
                yticklabels=LABELS,
                annot=True,
                fmt="d")
    plt.title("Karmasiklik matrisi")
    plt.ylabel("Gercek Degerler")
    plt.xlabel("Tahmin Degerleri")
    plt.show()

#Veri seti ile ilgli verileri gosterir
def show_basic_dataframe_info(dataframe,preview_rows=20):
    """
    dataframe: Veri seti
    preview_rows: Veri setinde gormek istenen satir sayisi
    """
    print("Veri setinde yer alan sutun sayisi: ",dataframe.shape[1],"\n")
    print("Veri setinde yer alan satir sayisi: ",dataframe.shape[0],"\n")
    print("Veri setinde yer alan ilk {0} veri: ".format(preview_rows),dataframe.head(preview_rows),"\n")
    print("Veri setinin istatiksel bilgileri(max-min deg, ort. vb)")
    print(dataframe.describe())

#Veri setini txt uzantili dosyadan okur
def read_data(file_path):

    column_names = ['user-id',
                    'activity',
                    'timestamp',
                    'x-axis',
                    'y-axis',
                    'z-axis']
    df = pn.read_csv(file_path,
                     header=None,
                     names=column_names)
    #Veri setindeki bilgililerin son satrinda  yer alan ";" kaldirir
    df['z-axis'].replace(regex=True,
      inplace=True,
      to_replace=r';',
      value=r'')
    df['z-axis'] = df['z-axis'].apply(convert_to_float)

    df.dropna(axis=0, how='any', inplace=True)

    return df

#Txt dosyasindaki sayisal degerleri setindeki degerleri floata cevirir
def convert_to_float(x):

    try:
        return np.float(x)
    except:
        return np.nan

#Grafikteki eksenleri belirler
def plot_axis(ax, x, y, title):
    ax.plot(x, y)
    ax.set_title(title)
    ax.xaxis.set_visible(False)
    ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])#y nin sinirlarini belirler
    ax.set_xlim([min(x), max(x)])#x in sinirlarini belirler
    ax.grid(True)

#Eylemlerin degerleri ile alakalı grafigi cizer
def plot_activity(activity,data):
    """
    activity:Veri setinde yapilan eylemler
    data:Veri seti
    """
    fig,(ax0,ax1,ax2)=plt.subplots(nrows=3,
         figsize=(15, 10),
         sharex=True)
    plot_axis(ax0,data['timestamp'],data['x-axis'],'x-axis')
    plot_axis(ax1,data['timestamp'],data['y-axis'],'y-axis')
    plot_axis(ax2,data['timestamp'],data['z-axis'],'z-axis')
    plt.subplots_adjust(hspace=0.2)
    fig.suptitle(activity)
    plt.subplots_adjust(top=0.9)
    plt.show()

"""
Bu fonksiyon bir veri seti alir ve 
x,y,z verilerinin yeniden şekillendirilmiş bölümlerini ve
ayrıca karşılık gelen etiketleri döndürür."""
def create_segments_and_labels(df,time_steps,step,label_name):
    """
    df:Beklenilen formattaki veri seti
    time_steps: Oluşturulan bir bolumun uzunluğunun tamsayı değeri

    """
    N_FEATURES=3
    segments=[]
    labels=[]
    for i in range(0,len(df)-time_steps,step):
        xs=df['x-axis'].values[i:i+time_steps]
        ys=df['y-axis'].values[i:i+time_steps]
        zs=df['z-axis'].values[i:i+time_steps]
        label=stats.mode(df[label_name][i:i+time_steps])[0][0]
        segments.append([xs,ys,zs])
        labels.append(label)
    reshaped_segments=np.asarray(segments, dtype= np.float32).reshape(-1, time_steps, N_FEATURES)
    labels=np.asarray(labels)
    return reshaped_segments,labels

pn.options.display.float_format='{:.1f}'.format
sns.set()
plt.style.use('ggplot')

LABELS=["Downstairs",
          "Jogging",
          "Sitting",
          "Standing",
          "Upstairs",
          "Walking"]

TIME_PERIODS=80

STEP_DISTANCE=40

print("***** Verileri yukle, incele ve donustur *****")

df=read_data("WISDM_ar_v1.1_raw.txt")

show_basic_dataframe_info(df,20)

df['activity'].value_counts().plot(kind='bar',
                                   title='Eylem Turune Gore Egitim Ornekleri')    

plt.show()

df['user-id'].value_counts().plot(kind='bar',
                                  title='Kullaniciya Gore Egitim Ornekleri')
plt.show()

for activity in np.unique(df["activity"]):
    subsets=df[df['activity']==activity][:180]
    plot_activity(activity,subsets)

LABEL = "ActivityEncoded"

le=preprocessing.LabelEncoder()

df[LABEL]=le.fit_transform(df["activity"].values.ravel())

print("\n***** Bolum icersindeki verileri tekrar sekillendir *****\n")

##Egtilecek ve test edilecek degerleri veri setinden ayristirir
df_test = df[df['user-id'] > 28]
df_train = df[df['user-id'] <= 28]

#Verileri normalize eder
df_train['x-axis'] = feature_normalize(df['x-axis'])
df_train['y-axis'] = feature_normalize(df['y-axis'])
df_train['z-axis'] = feature_normalize(df['z-axis'])

#Verleri bir veri setine(tablo) donusturur
df_train = df_train.round({'x-axis': 6, 'y-axis': 6, 'z-axis': 6})


x_train,y_train=create_segments_and_labels(df_train,
                                              TIME_PERIODS,
                                              STEP_DISTANCE,
                                              LABEL)

print("\n***** Keras tarafindan kullanilacak verileri tekrar sekillendirin *****\n")

print('x_train boyutu: ', x_train.shape)

print(x_train.shape[0], 'egitim orneklemleri')

print('y_train boyutu: ', y_train.shape)

num_time_periods, num_sensors = x_train.shape[1], x_train.shape[2]
num_classes = le.classes_.size
print(list(le.classes_))

input_shape = (num_time_periods*num_sensors)
x_train = x_train.reshape(x_train.shape[0], input_shape)

print('x_train boyutlari:', x_train.shape)

print('Girdi boyutlari:', input_shape)

x_train = x_train.astype("float32")
y_train = y_train.astype("float32")

y_train = np_utils.to_categorical(y_train, num_classes)
print('y_train nin yeni boyutu: ', y_train.shape)

model_m=Sequential()

model_m.add(Reshape((TIME_PERIODS,num_sensors),input_shape=(input_shape,)))
model_m.add(Conv1D(100,10,activation='relu',input_shape=(TIME_PERIODS,num_sensors)))
model_m.add(Conv1D(100,10,activation='relu'))
model_m.add(MaxPooling1D(3))
model_m.add(Conv1D(160,10,activation='relu'))
model_m.add(Conv1D(160,10,activation='relu'))
model_m.add(GlobalAveragePooling1D())
model_m.add(Dropout(0.5))
model_m.add(Dense(num_classes, activation='softmax'))
print(model_m.summary())

print("\n ***** Modeli egit ***** \n")

callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
        monitor='val_loss', save_best_only=True),
    keras.callbacks.EarlyStopping(monitor='acc', patience=1)
]

model_m.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
BATCH_SIZE=400
EPOCH=50
history=model_m.fit(x_train,y_train,batch_size=BATCH_SIZE,epochs=EPOCH,callbacks=callbacks_list,validation_split=0.2,verbose=1)

print("\n *****Egitilmis verilerin ogrenme egrileri ***** \n")

plt.figure(figsize=(6,4))
plt.plot(history.history['accuracy'],"g--",label="Egitim verilerinin dogrulugu")
plt.plot(history.history['val_accuracy'],"g",label="Dogrulama verilerinin dogrulugu")
plt.plot(history.history['loss'], "r--", label="Egitim verilerinin kaybi")
plt.plot(history.history['val_loss'], "r", label="Dogrulama verilerinin kaybi")
plt.title('Model Dogrulugu Ve Kaybi')
plt.ylabel('Dogruluk Ve Kayip')
plt.xlabel('Tur')
plt.ylim(0)
plt.legend()
plt.show()

print("\n***** Verileri test et *****\n")

df_test['x-axis'] = feature_normalize(df_test['x-axis'])
df_test['y-axis'] = feature_normalize(df_test['y-axis'])
df_test['z-axis'] = feature_normalize(df_test['z-axis'])

df_test = df_test.round({'x-axis': 6, 'y-axis': 6, 'z-axis': 6})

x_test, y_test = create_segments_and_labels(df_test,
                                            TIME_PERIODS,
                                            STEP_DISTANCE,
                                            LABEL)

x_test = x_test.reshape(x_test.shape[0], input_shape)

x_test = x_test.astype("float32")
y_test = y_test.astype("float32")

y_test = np_utils.to_categorical(y_test, num_classes)

score = model_m.evaluate(x_test, y_test, verbose=1)

print("\nTest verisinin dogrulugu: %0.2f" % score[1])
print("\nTest verisinin kaybi: %0.2f" % score[0])

print("\n***** Test verisinin karmasiklik matrisi *****\n")

y_pred_test = model_m.predict(x_test)
max_y_pred_test = np.argmax(y_pred_test, axis=1)
max_y_test = np.argmax(y_test, axis=1)

show_confusion_matrix(max_y_test, max_y_pred_test)

print("\n***** Test verileri icin siniflandirma raporu *****\n")

print(classification_report(max_y_test, max_y_pred_test))