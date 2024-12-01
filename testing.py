import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

model_path = '/Users/kabeermakkar/Desktop/290.v4i.multiclass/multi_label_model.pth'

if not os.path.exists(model_path):
    print(f"Model file not found at {model_path}")
    exit()

print(f"Model file located at: {model_path}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = resnet18(weights="IMAGENET1K_V1")  

num_classes = 81
model.fc = nn.Linear(model.fc.in_features, num_classes)

try:
    state_dict = torch.load(model_path, map_location=device)
    state_dict["fc.weight"] = state_dict["fc.weight"][:num_classes, :]
    state_dict["fc.bias"] = state_dict["fc.bias"][:num_classes]
    model.load_state_dict(state_dict)
    print("Model loaded successfully with 81 classes.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

model = model.to(device)
model.eval()

class_labels = {
    "Alpro-Blueberry-Soyghurt": 0,
    "Alpro-Fresh-Soy-Milk": 1,
    "Alpro-Shelf-Soy-Milk": 2,
    "Alpro-Vanilla-Soyghurt": 3,
    "Anjou": 4,
    "Arla-Ecological-Medium-Fat-Milk": 5,
    "Arla-Ecological-Sour-Cream": 6,
    "Arla-Lactose-Medium-Fat-Milk": 7,
    "Arla-Medium-Fat-Milk": 8,
    "Arla-Mild-Vanilla-Yoghurt": 9,
    "Arla-Natural-Mild-Low-Fat-Yoghurt": 10,
    "Arla-Natural-Yoghurt": 11,
    "Arla-Sour-Cream": 12,
    "Arla-Sour-Milk": 13,
    "Arla-Standard-Milk": 14,
    "Asparagus": 15,
    "Aubergine": 16,
    "Avocado": 17,
    "Banana": 18,
    "Beef-Tomato": 19,
    "Bravo-Apple-Juice": 20,
    "Bravo-Orange-Juice": 21,
    "Brown-Cap-Mushroom": 22,
    "Cabbage": 23,
    "Cantaloupe": 24,
    "Carrots": 25,
    "Conference": 26,
    "Cucumber": 27,
    "Floury-Potato": 28,
    "Galia-Melon": 29,
    "Garant-Ecological-Medium-Fat-Milk": 30,
    "Garant-Ecological-Standard-Milk": 31,
    "Garlic": 32,
    "Ginger": 33,
    "God-Morgon-Apple-Juice": 34,
    "God-Morgon-Orange-Juice": 35,
    "God-Morgon-Orange-Red-Grapefruit-Juice": 36,
    "God-Morgon-Red-Grapefruit-Juice": 37,
    "Golden-Delicious": 38,
    "Granny-Smith": 39,
    "Green-Bell-Pepper": 40,
    "Honeydew-Melon": 41,
    "Kaiser": 42,
    "Kiwi": 43,
    "Leek": 44,
    "Lemon": 45,
    "Lime": 46,
    "Mango": 47,
    "Nectarine": 48,
    "Oatly-Natural-Oatghurt": 49,
    "Oatly-Oat-Milk": 50,
    "Orange": 51,
    "Orange-Bell-Pepper": 52,
    "Papaya": 53,
    "Passion-Fruit": 54,
    "Peach": 55,
    "Pineapple": 56,
    "Pink-Lady": 57,
    "Plum": 58,
    "Pomegranate": 59,
    "Red-Beet": 60,
    "Red-Bell-Pepper": 61,
    "Red-Delicious": 62,
    "Red-Grapefruit": 63,
    "Regular-Tomato": 64,
    "Royal-Gala": 65,
    "Satsumas": 66,
    "Solid-Potato": 67,
    "Sweet-Potato": 68,
    "Tropicana-Apple-Juice": 69,
    "Tropicana-Golden-Grapefruit": 70,
    "Tropicana-Juice-Smooth": 71,
    "Tropicana-Mandarin-Morning": 72,
    "Valio-Vanilla-Yoghurt": 73,
    "Vine-Tomato": 74,
    "Watermelon": 75,
    "Yellow-Bell-Pepper": 76,
    "Yellow-Onion": 77,
    "Yoggi-Strawberry-Yoghurt": 78,
    "Yoggi-Vanilla-Yoghurt": 79,
    "Zucchini": 80
}
class_labels = {v: k for k, v in class_labels.items()}  # Invert the dictionary

def predict(image_path, model, transform, class_labels):
    if not os.path.exists(image_path):
        print(f"Image file not found at {image_path}")
        return None
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image)
        probabilities = torch.sigmoid(logits).squeeze()
        predictions = [class_labels[i] for i, val in enumerate(probabilities) if val > 0.2]

    return predictions

def save_predictions(image_paths, model, transform, class_labels, output_file):
    all_predictions = []

    for image_path in image_paths:
        if not os.path.exists(image_path):
            print(f"Image file not found at {image_path}")
            continue

        try:
            predictions = predict(image_path, model, transform, class_labels)
            print(f"Predictions for {image_path}: {predictions}")
            all_predictions.extend(predictions)  # collect predictions for all images
        except Exception as e:
            print(f"Error during prediction for {image_path}: {e}")

    # this removes duplicates and saves to a file
    unique_predictions = list(set(all_predictions))
    with open(output_file, 'w') as f:
        f.write(", ".join(unique_predictions))  # save as a single line

test_image_paths = [
    '/Users/kabeermakkar/Desktop/peach.jpg',
    '/Users/kabeermakkar/Desktop/apple.jpg',
    '/Users/kabeermakkar/Desktop/banana.jpg',
]
output_file_path = '/Users/kabeermakkar/Desktop/predictions.txt'

save_predictions(test_image_paths, model, transform, class_labels, output_file_path)
print(f"Predictions saved to {output_file_path}")