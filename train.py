from model import *
from dataHandler import *
import matplotlib.pyplot as plt


SAMPLES_PER_EPOCH = 30000
EPOCHS = 8
VALIDATION_SAMPLES = 6400
LEARNING_RATE = 1e-4

def train(train_data_generator,validate_data_generator):
    car_model = get_model(learning_rate=LEARNING_RATE)
    train_history = car_model.fit_generator(train_data_generator, 
                                samples_per_epoch=SAMPLES_PER_EPOCH, 
                                verbose=1, nb_epoch=EPOCHS,
                                validation_data = validate_data_generator, 
                                nb_val_samples=VALIDATION_SAMPLES)
    car_model.save('model.h5')
    return train_history
    

def plot_training_history(training_history):    
    plt.plot(training_history.history['loss'])
    plt.plot(training_history.history['val_loss'])
    plt.show()

if __name__ == '__main__':

    train_history = train(fetch_data(data_type='train'),fetch_data(data_type='valid'))
    plot_training_history(train_history)