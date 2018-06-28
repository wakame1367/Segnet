import dataset
import tensorflow as tf
from model import SegNet
from keras.callbacks import TensorBoard
from keras.backend.tensorflow_backend import set_session


def main():
    input_shape = (360, 480, 3)
    classes = 12
    epochs = 100
    batch_size = 1
    log_path = './logs/'

    class_weighting = [0.2595, 0.1826, 4.5640, 0.1417, 0.5051, 0.3826, 9.6446, 1.8418, 6.6823, 6.2478, 3.0, 7.3614]
    # set gpu usage
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True,
                                                      per_process_gpu_memory_fraction=0.8))
    session = tf.Session(config=config)
    set_session(session)

    print("loading data...")
    ds = dataset.Dataset(classes=classes,
                         train_file="CamVid/train.txt",
                         test_file="CamVid/test.txt")
    # need to implement, y shape is (None, 360, 480, classes)
    train_x, train_y = ds.load_data(root_path="CamVid",
                                    mode='train')

    train_x = ds.preprocess_inputs(train_x)
    train_y = ds.reshape_labels(train_y)
    print("input data shape...", train_x.shape)
    print("input label shape...", train_y.shape)

    # need to implement, y shape is (None, 360, 480, classes)
    test_x, test_y = ds.load_data(root_path="CamVid",
                                  mode='test')
    test_x = ds.preprocess_inputs(test_x)
    test_y = ds.reshape_labels(test_y)

    tb_cb = TensorBoard(log_dir=log_path, histogram_freq=1, write_graph=True, write_images=True)

    print("creating model...")
    model = SegNet(input_shape=input_shape, classes=classes)
    model.compile(loss="categorical_crossentropy", optimizer='adadelta', metrics=["accuracy"])

    model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs,
              verbose=1, class_weight=class_weighting, validation_data=(test_x, test_y), shuffle=True
              , callbacks=[tb_cb])

    model.save('seg.h5')


if __name__ == '__main__':
    main()
