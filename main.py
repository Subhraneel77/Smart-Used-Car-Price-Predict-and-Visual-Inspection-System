import pandas as pd
from models import model
from sensors import read_sensors
from visual_inspection import detect_damages

class CarInspectionSystem:
    def __init__(self):
        self.model = pd.read_pickle('models/best_model.pkl')
        self.scaler = pd.read_pickle('models/scaler.pkl')

    def preprocess_data(self, data):
        data = model.add_features(data)
        data = self.scaler.transform(data)
        return data

    def predict_price(self, car_data):
        car_data = self.preprocess_data(car_data)
        return self.model.predict(car_data)

    def inspect_car(self, image_path):
        return detect_damages.detect_damages(image_path)

if __name__ == "__main__":
    system = CarInspectionSystem()
    
    car_data = pd.DataFrame([model.X_test.iloc[0]])
    predicted_price = system.predict_price(car_data)
    print(f"Predicted Price: {predicted_price}")
    
    damage = system.inspect_car('visual_inspection/car_image.jpg')
    if damage:
        print("Damage detected during inspection")
    else:
        print("No damage detected during inspection")
