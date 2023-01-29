# first neural network with keras tutorial
from numpy import loadtxt, random, binary_repr
import numpy as np
from tensorflow.keras.models import Sequential, model_from_json, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.metrics import binary_accuracy
import tensorflow as tf
from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint
from configManager import getConfig, createConfig
from bitarray import bitarray

config = getConfig()

def getKeys(task):
    conv = {
        0: lambda x: intToBitarray(x),
        1: lambda x: intToBitarray(x),
        2: lambda x: intToBitarray(x),
    }

    dataset = loadtxt(config[task]["file_name"], delimiter=',', converters=conv)

    # if len(dataset) == 0:
    #     config[task]["skip_rows"] = 0
    #     createConfig(config)
    #     return None

    # config[task]["skip_rows"] = config[task]["skip_rows"] + len(dataset)
    # createConfig(config)

    pub_keys = np.hstack((dataset[:,1], dataset[:,2]))
    priv_keys = dataset[:,0]
    return (pub_keys, priv_keys)

def intToBitarray(x):
    if x is None:
        return None
    dec = int(x, 10)
    bin = binary_repr(dec, width=256)
    bits = [int(bin[i:i+1], 2) for i in range(0, len(bin), 1)]
    return bits

def testDataset():
    import io
    s = io.StringIO("91967691945608368349102856055644117753724469214902087563120796900728361989898,95106705493839555854878801883332190373380144124659858715293165822832899526788,67299438172294193893827607627742771697094893804791984902334728367127054480414\n1,2,3\n")
    conv = {
        0: lambda x: intToBitarray(x),
        1: lambda x: intToBitarray(x),
        2: lambda x: intToBitarray(x),
    }
    dataset = loadtxt(s, delimiter=",", converters=conv) 
    print(dataset)
    print('np.r_1:3')
    print(np.hstack((dataset[:,1], dataset[:,2]))[0])
    print(np.hstack((dataset[:,1], dataset[:,2]))[1])
    print('0')
    print(dataset[:,0])


def testDataset2():
    import io
    s = io.StringIO("91967691945608368349102856055644117753724469214902087563120796900728361989898,95106705493839555854878801883332190373380144124659858715293165822832899526788,67299438172294193893827607627742771697094893804791984902334728367127054480414\n1,2,3\n")
    conv = {
        0: lambda x: intToBitarray(x),
        1: lambda x: intToBitarray(x),
        2: lambda x: intToBitarray(x),
    }
    dataset = loadtxt(s, delimiter=",", converters=conv)
    print(dataset)
    print('np.r_1:3')
    print(np.hstack((dataset[:,1], dataset[:,2]))[0])
    print(np.hstack((dataset[:,1], dataset[:,2]))[1])
    print('0')
    print(dataset[:,0])

def reshapeDataLine(input_string):
    # Split the string by comma
    numbers = tf.io.decode_csv(input_string, record_defaults=['0','0','0'], field_delim=',')
    # numbers = tf.slice(numbers, [0], [1])
    # print(numbers)
    # for f in numbers:
    #     print(f"type: {f.dtype.name}, shape: {f.shape}")
    #     print(f)
    #     print(type(f))

    return tf.string.substr(input_string, 0, 2)

    (p, x, y) = numbers
    # print(input_string)
    # print(numbers[1])
    # result = []
    # print(tf.strings.length(input_string))
    # for i in range(0, len(p), 8):
    #     slice = p[-8-i:-i] if i else p[-8:]
    #     result.append(slice)

    # print(result)

    p_bytes = bytes.fromhex(p)

    return p_bytes
    exit(1)

    # Assign the numbers to three variables
    # Start a Tensorflow session
    with tf.compat.v1.Session() as sess:
        a, b, c = sess.run([p, x, y])
        a_bits = bitarray(a)
        b_bits = bitarray(b)
        c_bits = bitarray(c)
        
        print(a_bits)
        print(b_bits)
        print(c_bits)
        
    # Convert the numbers to bitarrays
    p_bits = bitarray(tf.shape(p)[0])
    # x_bits = bitarray(tf.shape(x)[0])
    # y_bits = bitarray(tf.shape(y)[0])

    # Concatenate the last two bitarrays
    # xy_bits = x_bits + y_bits

    return (p_bits)

def hex_string_to_int_arrays(hex_string):
    # Split the string into a list of 3 hex numbers
    hex_numbers = hex_string.split(',')
    
    # Initialize an empty list to store the integer arrays
    int_arrays = []
    
    # Iterate over the hex numbers
    for hex_num in hex_numbers:
        # Convert the hex number to an integer
        int_num = int(hex_num, 16)
        
        # Convert the integer to bytes and reverse the order
        int_bytes = int_num.to_bytes(32, byteorder='big')[::-1]
        
        # Append the bytes to the list of integer arrays
        int_arrays.append(np.array(int_bytes))
    
    # Return the tuple of integer arrays
    return tuple(int_arrays)

def testFileSlicing():
    # Define the model
    model = prepareModel()

    # Define the CSV file path
    csv_file = config['train']['file_name']

    # Load the data from the CSV file
    batch_size = 32
    ds = tf.data.TextLineDataset(csv_file)
    ds = ds.map(reshapeDataLine)#(lambda line: line)#print(line))#tf.io.decode_csv(line, [[0.]]*cols, field_delim=","))
    for p in ds.as_numpy_iterator():
        print(p)
    #ds = ds.shuffle(buffer_size=1000).batch(batch_size).repeat()

    # Use the dataset to train the model
    #model.fit(ds, epochs=5)

def testFileSlicing2():
    file = tf.keras.utils.get_file('predict.csv', "./predict.csv")
    ds = tf.data.experimental.make_csv_dataset(
        file,
        batch_size=256,
        label_name='predict_ds',
        num_epochs=1)
    print(ds)

################ ОПИСАНИЕ НЕЙРОННОЙ СЕТИ НАЧАЛО ######################
# Здесь описывается сама нейронная сеть, можно играться, добавлять слои нейронов,
# изменять функции и прочее.
def prepareModel():
    # define the keras model
    model = Sequential()
    model.add(Dense(1024, input_shape=(15,), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])#metrics=[binary_accuracy])
    plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
    return model
################ ОПИСАНИЕ НЕЙРОННОЙ СЕТИ КОНЕЦ ######################

def hexStrToInt(p):
    key_len = 64#tf.strings.length(p)
    arr_len = 8#int((key_len + 7) / 8)
    steps = tf.range(0, key_len, 8)
    #return steps
    step_lengths = tf.fill([8], arr_len)
    #return step_lengths
#     if key_len % 8 != 0:
#         step_lengths = tf.tensor_scatter_nd_update(step_lengths, [[arr_len - 1]], [key_len % 8])

    slice = tf.strings.substr(p, [0,8,16,24,32,40,48,56], [8,8,8,8,8,8,8,8])
    re = tf.py_function(func=toInt, inp=[slice], Tout=[tf.uint32]*8)
    return re
    
def reshapeDataLine(input_string):
    window_size = config["window_size"]
    n = window_size - 1

    numbers = tf.io.decode_csv(input_string, record_defaults=[0.0]*window_size, field_delim=',')

    private = numbers[n]
    public = numbers[0:n]
    
    return (public, private)


################ ОБУЧЕНИЕ НЕЙРОННОЙ СЕТИ КОНЕЦ ######################
# Здесь нейронная сеть обучается и сохраняется, если показатели улучшились.
def trainModel(model):
    checkpoint = ModelCheckpoint(config["model_file_name"], monitor='loss', verbose=1, save_best_only=True, mode='min')

    # Define the CSV file path
    csv_file = config["train"]["file_name"]

    # Load the data from the CSV file
    batch_size = 1
    ds = tf.data.TextLineDataset(csv_file)
    ds = ds.map(reshapeDataLine).batch(batch_size)
#     for p in ds.as_numpy_iterator():
#         print(p)
    
    model.fit(ds, epochs=1, verbose=1, callbacks=[checkpoint])

    return model
################ ОБУЧЕНИЕ НЕЙРОННОЙ СЕТИ КОНЕЦ ######################



def evaluate(model):
    keys = getKeys("evaluate")
    if keys is None:
        return
    (pub_keys, priv_keys) = keys
    
    # evaluate the keras model
    _, accuracy = model.evaluate(pub_keys, priv_keys, verbose=1)
    print('Model accuracy: %.2f' % (accuracy*100))


def saveModel(model):
    model.save(config["model_file_name"])
    print('Saved model to disk')


def loadModel():
    model = load_model(config["model_file_name"])
    model.summary()
    return model


def predict(model):
    keys = getKeys("predict")
    if keys is None:
        return
    (pub_keys, priv_keys) = keys

    # make class predictions with the model
    predictions = model.predict(pub_keys)
    for i in range(10):
        print('')
        print(i)
        print(predictions[i])
        print('Prediction: ')
        str = ''
        for j in predictions[i]:
            if j == 1:
                str = str + '1'
            else:
                str = str + '0'

        #print(predictions[i])
        # print(str)
        # print(int(str, base=2))
        print(hex(int(str, base=2)))
        predicted = int(str, base=2)
        print('Real:       ')
        str = ''
        for j in priv_keys[i]:
            if j == 1:
                str = str + '1'
            else:
                str = str + '0'
        #print(priv_keys[i])
        # print(str)
        # print(int(str, base=2))
        print(hex(int(str, base=2)))
        actual = int(str, base=2)

        print('XOR')
        print(hex(predicted ^ actual))