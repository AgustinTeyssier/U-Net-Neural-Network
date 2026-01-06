import pandas as pd 
import tensorflow as tf
import matplotlib.pyplot as plt
import os

from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


files=[]
paths = []
for dirname, _, filenames in os.walk('png_masks'):
    for filename in filenames:
        path = os.path.join(dirname, filename)    
        paths.append(path)
        files.append(filename)

mfiles=[]
mpaths = []
for dirname, _, filenames in os.walk('png_images'):
    for filename in filenames:
        path = os.path.join(dirname, filename)    
        mpaths.append(path)
        mfiles.append(filename)

df0=pd.read_csv('stage_1_train_images.csv')

dir0='png_images'
dir1='png_masks'
df0['label']=df0['has_pneumo']
df0['path']=df0['new_filename'].apply(lambda x:os.path.join(dir0,x))
df0['mpath']=df0['new_filename'].apply(lambda x:os.path.join(dir1,x))
df0=df0[df0['label']==1].reset_index(drop=True)

n=len(df0)
df=df0.iloc[0:(n//10)*3]
test_df=df0.iloc[(n//10)*3:(n//10)*4]
img_size = [256,256]


def data_augmentation(car_img, mask_img):
    if tf.random.uniform(()) > 0.5:
        car_img = tf.image.flip_left_right(car_img)
        mask_img = tf.image.flip_left_right(mask_img)

    return car_img, mask_img


IMG_SIZE = (256, 256)

def preprocessing(image_path, mask_path, img_size=IMG_SIZE, augment=False):
    # Leer y procesar imagen
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, img_size)
    
    # Leer y procesar máscara
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.convert_image_dtype(mask, tf.float32)
    mask = tf.image.resize(mask, img_size, method='nearest')
    mask = tf.where(mask > 0.5, 1.0, 0.0)
    
    if augment:
        # Aumentación aleatoria
        if tf.random.uniform(()) > 0.5:
            img = tf.image.flip_left_right(img)
            mask = tf.image.flip_left_right(mask)
        if tf.random.uniform(()) > 0.5:
            img = tf.image.flip_up_down(img)
            mask = tf.image.flip_up_down(mask)
    
    return img, mask

def create_dataset(df, batch_size=8, augment=False, shuffle=False):
    
    dataset = tf.data.Dataset.from_tensor_slices((df['path'].values, df['mpath'].values))
    
    # Preprocesamiento con o sin aumentación
    def process_paths(image_path, mask_path):
        return preprocessing(image_path, mask_path, augment=augment)
    
    dataset = dataset.map(process_paths, num_parallel_calls=tf.data.AUTOTUNE)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )

def iou_score(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection

    return (intersection + smooth) / (union + smooth)



train_df = df0.iloc[:(n//10)*3]
val_df   = df0.iloc[(n//10)*3:(n//10)*4]
test_df  = df0.iloc[(n//10)*4:(n//10)*5]

train_dataset = create_dataset(train_df, batch_size=8, augment=True, shuffle=True)
val_dataset   = create_dataset(val_df, batch_size=8)
test_dataset  = create_dataset(test_df, batch_size=8)


#Variacion de acuerod al modelo:


def simple_unet(input_size=(256, 256, 3)):
    inputs = layers.Input(input_size)
    
    # Encoder
    conv1 = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(32, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = layers.Conv2D(64, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Middle
    conv3 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv3)
    
    # Decoder
    up1 = layers.UpSampling2D(size=(2, 2))(conv3)
    up1 = layers.Conv2D(64, 2, activation='relu', padding='same')(up1)
    merge1 = layers.concatenate([conv2, up1], axis=3)
    
    conv4 = layers.Conv2D(64, 3, activation='relu', padding='same')(merge1)
    conv4 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv4)
    
    up2 = layers.UpSampling2D(size=(2, 2))(conv4)
    up2 = layers.Conv2D(32, 2, activation='relu', padding='same')(up2)
    merge2 = layers.concatenate([conv1, up2], axis=3)
    
    conv5 = layers.Conv2D(32, 3, activation='relu', padding='same')(merge2)
    conv5 = layers.Conv2D(32, 3, activation='relu', padding='same')(conv5)
    
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(conv5)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Crear modelo
model = simple_unet()

# Compilar
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=[dice_coefficient, iou_score, 'accuracy']
)



callbacks = [
    ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_dice_coefficient', mode='max'),
    EarlyStopping(patience=10, monitor='val_dice_coefficient', mode='max', restore_best_weights=True),
    ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7)
]

history = model.fit(train_dataset,
    validation_data=val_dataset,
    epochs=2,
    callbacks=callbacks,
    verbose=1
)

# Evaluar en test
test_results = model.evaluate(test_dataset, verbose=1)
print(f"Test Loss: {test_results[0]:.4f}")
print(f"Test Dice: {test_results[1]:.4f}")
print(f"Test IoU: {test_results[2]:.4f}")

# Visualizar predicciones
def plot_predictions(model, dataset, num_samples=3):
    images, masks = next(iter(dataset.take(1)))
    preds = model.predict(images[:num_samples])
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    
    for i in range(num_samples):
        axes[i, 0].imshow(images[i])
        axes[i, 0].set_title('Input')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(masks[i].numpy().squeeze(), cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(preds[i].squeeze(), cmap='gray')
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.show()

plot_predictions(model, test_dataset)

