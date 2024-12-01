from flask import Flask, render_template, request, jsonify
import requests
from serpapi import GoogleSearch

app = Flask(__name__)

# Define your functions (get_first_search_result, my_filtering_function, etc.) here.
def get_first_search_result(query):
    api_key = '6c5ed0297a457c521221d82ab650462c7d1e3b0d918a9b137ebf6593ae3df4ef'
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
    api_key = "3be2a6e8823940f499833fe6b3a591f3"
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
    return render_template('webiste.html')  # Your HTML file goes here.

@app.route('/get_recipes', methods=['POST'])
def get_recipes():
    ingredients = request.json.get('ingredients', [])
    URL = generate_recipe_url(ingredients)
    recipe_data = my_custom_function(URL)

    if recipe_data:
        recipes = []
        for recipe in recipe_data:
            filtered_data = dict(filter(my_filtering_function, recipe.items()))
            recipe_title = filtered_data.get('title')
            image = filtered_data.get('image')
            if recipe_title:
                URL2Search = f"{recipe_title} recipe"
                URL2 = get_first_search_result(URL2Search)
                if URL2:
                    recipes.append({
                        'title': recipe_title,
                        'image': image,
                        'url': URL2
                    })
        return jsonify(recipes)
    else:
        return jsonify({"error": "No recipes found"}), 400

if __name__ == '__main__':
    app.run(debug=True)
