from flask import Flask, render_template, request, jsonify
import requests
from serpapi import GoogleSearch
import os


app = Flask(__name__)

# Define your functions (get_first_search_result, my_filtering_function, etc.) here.
def get_first_search_result(query):
    api_key = 'nuh uh'
    params = {
        "engine": "google",
        "q": query,
        "api_key": api_key,
    }
    search = GoogleSearch(params)
    results = search.get_dict()

    if "organic_results" in results:
        first_link = results["organic_results"][0]["link"]
        return first_link
    else:
        return None

def my_filtering_function(pair):
    wanted_keys = ['id', 'title', 'usedIngredientCount', 'missedIngredientCount', 'image']
    key, value = pair
    return key in wanted_keys

def generate_recipe_url(ingredients, number=6):
    base_url = "https://api.spoonacular.com/recipes/findByIngredients"
    formatted_ingredients = ',+'.join(ingredients)
    url = f"{base_url}?ingredients={formatted_ingredients}&number={number}&ranking=2"
    return url

def my_custom_function(url):
    api_key = "not here either"
    headers = {
        'x-api-key': api_key
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        return None

@app.route('/')
def index():
    return render_template('website.html')  # Your HTML file goes here.

@app.route('/get_recipes', methods=['POST'])
def get_recipes():
    try:
        # Specify the absolute path to predictions.txt
        file_path = '/home/dom/ELEC290 stuff/predictions.txt'

        # Check if the file exists
        if not os.path.exists(file_path):
            return jsonify({"error": f"{file_path} file not found"}), 400

        # Read ingredients from the predictions.txt file
        with open(file_path, 'r') as file:
            content = file.read().strip()
            ingredients = [ingredient.strip() for ingredient in content.split(',')]

        if not ingredients or ingredients == ['']:
            return jsonify({"error": "No ingredients found in predictions.txt"}), 400

        # Generate the recipe URL based on the ingredients
        URL = generate_recipe_url(ingredients)
        recipe_data = my_custom_function(URL)

        if not recipe_data:
            return jsonify({"error": "Failed to fetch recipes"}), 500

        # Process and filter the recipe data
        recipes = []
        for recipe in recipe_data:
            filtered_data = dict(filter(my_filtering_function, recipe.items()))
            recipe_title = filtered_data.get('title')
            image = filtered_data.get('image')
            if recipe_title:
                URL2Search = f"{recipe_title} recipe"
                URL2 = get_first_search_result(URL2Search)
                if URL2:
                    recipes.append({'title': recipe_title, 'image': image, 'url': URL2})

        return jsonify(recipes)
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "An internal error occurred"}), 500



if __name__ == '__main__':
    app.run(debug=True)
