from files import file_paths
import utilities
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import conllu
import pickle

language = 'eng' # for Spanish use language = 'es'

num_features = 14
emb_size = 50
left_arc = np.array([[1,0,0]])
right_arc = np.array([[0,1,0]])
shift = np.array([[0,0,1]])

# loading the dataset
data_train = utilities.get_sentences(file_paths[language+'_train'])
data_dev = utilities.get_sentences(file_paths[language+'_dev'])
data_test = utilities.get_sentences(file_paths[language+'_test'])


'''
    Loading word embeddings, pos_embeddings and other
    required files
'''
with open(file_paths[language+'_word_emb'], 'rb') as f:
    word_emb = pickle.load(f)

with open(file_paths['pos_emb'], 'rb') as f:
    pos_emb = pickle.load(f)

with open(file_paths[language+'_train_children'], 'rb') as f:
    train_children = pickle.load(f)

with open(file_paths[language+'_dev_children'], 'rb') as f:
    dev_children = pickle.load(f)

with open(file_paths[language+'_test_children'], 'rb') as f:
    test_children = pickle.load(f)

with open(file_paths['left_labels'], 'rb') as f:
    left_labels = pickle.load(f)

with open(file_paths['right_labels'], 'rb') as f:
    right_labels = pickle.load(f)

with open(file_paths['shift_label'], 'rb') as f:
    shift_label = pickle.load(f)


'''
    Creating the model
'''
inputs = keras.Input(shape=(num_features*emb_size,))
x1 = layers.Dense(200, activation="relu")(inputs)
outputs = layers.Dense(101, name="predictions")(x1)
model = keras.Model(inputs=inputs, outputs=outputs)


optimizer = keras.optimizers.Adam(learning_rate=1e-3)
loss_func = keras.losses.CategoricalCrossentropy(from_logits=True)
train_acc_metric = keras.metrics.Accuracy()


'''
    Training
'''
epochs = 10
for epoch in range(epochs):
    print(f'Start of epoch: {epoch+1}')

    for sent_index, sentence in enumerate(data_train):
        stack = ['ROOT']
        buffer = [word for word in sentence]
        while (len(stack)>1 or len(buffer)>0):
            correct_act = utilities.correct_action_labelled(stack, buffer, left_labels, right_labels, shift_label)
            if correct_act is None:
                break

            action = correct_act[0]
            correct_act = correct_act[1]

            ip = utilities.get_input(stack, buffer, sentence, sent_index, train_children, word_emb, pos_emb)
            if ip == 'break':
                break
            
            ip = np.resize(ip, (1, num_features*emb_size,))
            with tf.GradientTape() as tape:
                predicted_act = model(ip, training=True)
                loss_val = loss_func(correct_act, predicted_act)
            
            grads = tape.gradient(loss_val, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            train_acc_metric.update_state(tf.math.argmax(correct_act[0]), tf.math.argmax(predicted_act[0]))
            if (action == shift).all():
                stack.append(buffer[0])
                buffer.pop(0)
            elif (action == left_arc).all():
                stack.pop(-2)
            else:
                stack.pop(-1)
                   
        if sent_index % 50 == 49:
            print(f'Training accuracy at epoch {epoch+1} sentence {sent_index+1}: {train_acc_metric.result()}')

    print(f'Accuracy at epoch {epoch+1}: {train_acc_metric.result()}')
    train_acc_metric.reset_states()

'''
    Evaluating the model on dev set
'''
for sent_index, sentence in enumerate(data_dev):
    stack = ['ROOT']
    buffer = [word for word in sentence]
    while (len(stack)>1 or len(buffer)>0):
        correct_act = utilities.correct_action_labelled(stack, buffer, left_labels, right_labels, shift_label)
        if correct_act is None:
            break

        action = correct_act[0]
        correct_act = correct_act[1]

        ip = utilities.get_input(stack, buffer, sentence, sent_index, dev_children, word_emb, pos_emb)
        if ip == 'break':
            break
        
        ip = np.resize(ip, (1, num_features*emb_size,))
        predicted_act = model(ip, training=False)
        
        train_acc_metric.update_state(tf.math.argmax(correct_act[0]), tf.math.argmax(predicted_act[0]))
        if (action == shift).all():
            stack.append(buffer[0])
            buffer.pop(0)
        elif (action == left_arc).all():
            stack.pop(-2)
        else:
            stack.pop(-1)
               
    if sent_index % 50 == 49:
        print(f'Dev accuracy at sentence {sent_index+1}: {train_acc_metric.result():.5f}')

print(f'Accuracy on entire dev set: {train_acc_metric.result()}')
train_acc_metric.reset_states()


'''
    Evaluating the model on test set
'''
for sent_index, sentence in enumerate(data_test):
    stack = ['ROOT']
    buffer = [word for word in sentence]
    while (len(stack)>1 or len(buffer)>0):
        correct_act = utilities.correct_action_labelled(stack, buffer, left_labels, right_labels, shift_label)
        if correct_act is None:
            break

        action = correct_act[0]
        correct_act = correct_act[1]

        ip = utilities.get_input(stack, buffer, sentence, sent_index, test_children, word_emb, pos_emb)
        if ip == 'break':
            break
        
        ip = np.resize(ip, (1, num_features*emb_size,))
        predicted_act = model(ip, training=False)
        
        train_acc_metric.update_state(tf.math.argmax(correct_act[0]), tf.math.argmax(predicted_act[0]))
        if (action == shift).all():
            stack.append(buffer[0])
            buffer.pop(0)
        elif (action == left_arc).all():
            stack.pop(-2)
        else:
            stack.pop(-1)
               
    if sent_index % 50 == 49:
        print(f'Test accuracy at sentence {sent_index+1}: {train_acc_metric.result():.5f}')

print(f'Accuracy on entire test set: {train_acc_metric.result()}')
train_acc_metric.reset_states()