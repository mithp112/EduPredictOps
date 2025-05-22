import mlflow
import mlflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

with mlflow.start_run(run_name="MLPModel12_V1"):
    model = Sequential([
        Flatten(input_shape=(6, 9)),
        Dense(128, activation='relu'),
        Dense(62, activation='relu'),
        Dense(32, activation='relu'),
        Dense(18, activation='linear')
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Train model
    history = model.fit(X_train, y_train, validation_split=0.2, epochs=10)

    # Log metrics
    for epoch in range(len(history.history['loss'])):
        mlflow.log_metric("loss", history.history['loss'][epoch], step=epoch)
        mlflow.log_metric("val_loss", history.history['val_loss'][epoch], step=epoch)

    # Log parameters
    mlflow.log_param("layers", 4)
    mlflow.log_param("units_fc1", 128)

    # Log model
    mlflow.keras.log_model(model, "model")

    mlflow.end_run()
