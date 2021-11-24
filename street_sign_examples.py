from my_utils import split_data, order_test_set, create_generator
from deeplearning_models import streetsign_model
from tensorflow.keras.callbacks import ModelCheckpoint

if __name__ == "__main__":

    if False:
        path_to_data = "C:\\Users\\ashishpahal\\ML_projects\\traffic_sign\\Train"
        path_to_save_train ="C:\\Users\\ashishpahal\\ML_projects\\traffic_sign\\training_data\\train"
        path_to_save_val ="C:\\Users\\ashishpahal\\ML_projects\\traffic_sign\\training_data\\val"

        split_data(path_to_data, path_to_save_train=path_to_save_train, path_to_save_val=path_to_save_val)
    
    if False:
        path_to_images ="C:\\Users\\ashishpahal\\ML_projects\\traffic_sign\\Test"
        path_to_csv ="C:\\Users\\ashishpahal\\ML_projects\\traffic_sign\\Test.csv"
        order_test_set(path_to_images, path_to_csv)

    path_to_train ="C:\\Users\\ashishpahal\\ML_projects\\traffic_sign\\training_data\\train"
    path_to_val ="C:\\Users\\ashishpahal\\ML_projects\\traffic_sign\\training_data\\val"
    path_to_test ="C:\\Users\\ashishpahal\\ML_projects\\traffic_sign\\Test"
    batch_size = 64
    epochs = 15

    train_generator, val_generator, test_generator = create_generator(batch_size, path_to_train, path_to_val, path_to_test)
    no_classes = train_generator.num_classes

    path_to_save_model = './Models'
    checkpoint_saver = ModelCheckpoint(
        path_to_save_model,
        monitor= "val_accuracy",
        mode='max',
        save_best_only=True,
        save_freq='epoch',
        verbose=1
    )

    model = streetsign_model(no_classes)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=[checkpoint_saver]
    )