import numpy as np
from sklearn.model_selection import train_test_split

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ModuleNotFoundError:
    TENSORFLOW_AVAILABLE = False

# Generate Sample Data
X = np.random.rand(1000, 10)
y = np.random.randint(0, 2, 1000)

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

if TENSORFLOW_AVAILABLE:
    # Neural Network with L2 and Dropout
    model = Sequential([
        Dense(64, activation='relu',
              kernel_regularizer=l2(0.01),
              input_shape=(10,)),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    # Compile Model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Early Stopping
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    # Train Model
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        callbacks=[early_stop],
        verbose=0
    )

    # Evaluation
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy:", accuracy)
else:
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score

    # Fallback: scikit-learn MLP approximates a neural network.
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        alpha=0.01,
        learning_rate_init=0.001,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.2,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)