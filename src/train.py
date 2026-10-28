from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


def main():
    features, target = load_diabetes(return_X_y=True)
    features_train, features_test, target_train, target_test = train_test_split(
        features, target, test_size=0.4, random_state=42
    )

    model = LinearRegression()
    model.fit(features_train, target_train)
    predictions = model.predict(features_test)

    rmse = mean_squared_error(target_test, predictions) ** 0.5
    mae = mean_absolute_error(target_test, predictions)
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")


if __name__ == "__main__":
    main()