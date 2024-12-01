# Pic2Plate, the Modern Recipe Recommender

This project simplifies meal planning by using image recognition to identify grocery items and suggest recipes based on available ingredients. Designed for students or anyone unsure of what to cook, the system uses a Raspberry Pi 5, a camera module, and a web app for seamless interaction.

## Features

- **Ingredient Recognition:** Scan multiple grocery items (e.g., fruits, vegetables, packaged goods) and classify them using a custom-trained machine learning model.
- **Recipe Suggestions:** Generate recipe ideas based on identified items, with filters like preparation time and dietary preferences.
- **User Interaction:** Upload images via the web app or scan groceries directly using the camera module.
- **Custom Dataset:** Trained on a dataset of over 5,000 images with metadata, ensuring accurate and reliable classification.
- **Transfer Learning:** Implements a fine-tuned model for grocery item recognition with PyTorch.

### Prerequisites

- **Hardware:** Raspberry Pi 5 with a camera module.
- **Software:** Python 3.9+, PyTorch, Flask (for the web app), and any necessary libraries.

## Hardware Requirements

- Raspberry Pi 5
- Raspberry Pi Camera Module
- Power Supply for Raspberry Pi

## Software Requirements

- Python 3.9+
- PyTorch (with torchvision)
- Flask (for web app development)
- Jupyter Notebook (for testing and experimentation)
- Additional libraries: `numpy`, `pandas`, `matplotlib`

This project combines computer vision and machine learning to simplify meal planning and grocery management. Using a Raspberry Pi 5, a camera module, and a web application, users can scan their groceries, identify items, and receive recipe recommendations tailored to their available ingredients.

## Current Limitations

- **Model Accuracy:** While the classification model performs well on the training dataset, testing with real-world images may highlight areas for improvement.
- **Ingredient Overlap:** Items with similar appearances (e.g., lemons and limes) might occasionally be misclassified.
- **Dataset Size:** Expanding the dataset with more diverse images could improve classification accuracy and support the addition of new items.

## Future Directions

We envision several enhancements for this project:

- **Expanded Dataset:** Incorporate more grocery categories and fine-grained labels for improved recognition accuracy.
- **Mobile App Integration:** Develop a mobile app version to allow scanning on the go.
- **Smart Kitchen Connectivity:** Integrate with IoT devices such as smart refrigerators to automatically update grocery inventory.
- **Nutritional Analysis:** Provide users with nutritional information for scanned items and suggested recipes.
- **User Feedback Loop:** Allow users to manually correct misclassified items to continuously refine the model.

## Why This Matters

- **Convenience:** Streamlines meal planning and eliminates the guesswork of deciding what to cook.
- **Resource Optimization:** Helps users make the most of what they already have, reducing food waste.
- **Scalable Technology:** This project lays the groundwork for AI-powered solutions in smart kitchens and household management.


## Acknowledgements

 - [Marcus Klasson for his GroceryStore Dataset](https://github.com/marcusklasson/vcca_grocerystore/tree/master)
 - YOLOv5 and PyTorch communities for supporting tools and documentation.

## License

[MIT](https://choosealicense.com/licenses/mit/)
