import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
import datetime
from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# fetch dataset 
combined_cycle_power_plant = fetch_ucirepo(id=294) 
  
# data (as pandas dataframes) 
X = combined_cycle_power_plant.data.features 
y = combined_cycle_power_plant.data.targets 

# Dividir los datos en entrenamiento (80%) y validación (20%)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Ver el tamaño de los conjuntos resultantes
print(f'Tamaño del conjunto de entrenamiento: {X_train.shape[0]} instancias')
print(f'Tamaño del conjunto de validación: {X_val.shape[0]} instancias')

# Crear un scaler para normalizar las características
scaler = MinMaxScaler()

# Ajustar el scaler a los datos de entrenamiento y transformar tanto el conjunto de entrenamiento como el de validación
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Ver los datos normalizados
print(f'Datos normalizados de entrenamiento:\n{X_train_scaled[:5]}')
print(f'Datos normalizados de validación:\n{X_val_scaled[:5]}')

# Guardar el scaler para su uso
joblib.dump(scaler, 'scaler.pkl')

# Definir el modelo MLP con diferentes arquitecturas
def create_model(hidden_layers=[64, 64, 64], optimizer='adam'):
    model = Sequential()
    
    # Capa de entrada (input_shape debe coincidir con el número de características)
    model.add(Dense(hidden_layers[0], input_shape=(X_train_scaled.shape[1],), activation='relu'))
    
    # Capas ocultas
    for units in hidden_layers[1:]:
        model.add(Dense(units, activation='relu'))
    
    # Capa de salida
    model.add(Dense(1))  # Salida con una sola neurona (para regresión)
    
    # Compilar el modelo
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model

# Diferentes arquitecturas para probar
architectures = [
    [32, 32],          # 2 capas ocultas de 32 neuronas cada una
    [64, 64],          # 2 capas ocultas de 64 neuronas cada una
    [128, 64],         # 2 capas ocultas, primera de 128 neuronas y segunda de 64 neuronas
    [64, 32, 16],      # 3 capas ocultas de 64, 32 y 16 neuronas respectivamente
    [128, 128, 64],    # 3 capas ocultas, dos de 128 neuronas y una de 64
]

# Diferentes optimizadores para probar
optimizers = [
    Adam(),
    tf.keras.optimizers.SGD(),
    tf.keras.optimizers.RMSprop(),
    tf.keras.optimizers.Adagrad()
]

# Directorio para guardar los logs de TensorBoard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Inicializar un DataFrame para almacenar los resultados
results_df = pd.DataFrame(columns=['Architecture', 'Optimizer', 'Val_MAE', 'Val_MSE'])

for arch in architectures:
    for opt_class in [Adam, tf.keras.optimizers.SGD, tf.keras.optimizers.RMSprop, tf.keras.optimizers.Adagrad]:
        print(f"Training model with architecture: {arch} and optimizer: {opt_class.__name__}")
        
        # Crear una nueva instancia del optimizador en cada iteración
        opt = opt_class()
        
        # Crear y entrenar el modelo
        model = create_model(hidden_layers=arch, optimizer=opt)
        history = model.fit(X_train_scaled, y_train, epochs=50, validation_data=(X_val_scaled, y_val), callbacks=[tensorboard_callback])

        # Evaluar el modelo y guardar los resultados
        loss, mae = model.evaluate(X_val_scaled, y_val)
        
        # Obtener el MAE y MSE en los datos de validación
        val_mae = history.history['val_mae'][-1]
        val_mse = history.history['val_loss'][-1]
        
        # Guardar los resultados en el DataFrame usando pd.concat()
        new_row = {
            'Architecture': str(arch),
            'Optimizer': opt_class.__name__,
            'Val_MAE': val_mae,
            'Val_MSE': val_mse
        }
        results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)


# Ahora puedes imprimir o guardar el DataFrame results_df con todos los resultados
print(results_df)

# Mostrar los resultados ordenados por el mejor MAE
results = results_df.sort_values(by='Val_MAE')
print(results)

# Guardar el DataFrame de resultados
results.to_csv('mlp_results.csv', index=False)


#ejecutar comando tensorboard --logdir=logs/fit
