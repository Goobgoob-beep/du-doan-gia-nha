import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_excel("Housing.xlsx")

X = df.drop(columns=["gia"])
y = df["gia"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"R2 Score: {r2:.4f}")
print(f"MAE: {mae:,.0f} VND")

pd.set_option('display.float_format', lambda x: '%.0f' % x)

ket_qua = pd.DataFrame({
    'Gia_thuc_te': y_test,
    'Gia_du_doan': y_pred
})
print("\n5 căn nhà đầu tiên trong tập kiểm thử")
print(ket_qua.head(5))

test_house = pd.DataFrame([
    [5000, 10, 5, 5, 1, 1, 1, 1, 1, 0, 0, 2],
    [2000, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
], columns=X.columns)

predict_house_price = model.predict(test_house)
print("\nDự đoán giá các căn nhà mới nhập")
for idx, price in enumerate(predict_house_price, 1):
    print(f"Nhà {idx}: {price:,.0f} VND")
