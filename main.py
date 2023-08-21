from fastapi import FastAPI, UploadFile,Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
from pymongo import MongoClient
from pydantic import BaseModel
from typing import List
import requests
import tempfile
import joblib 



app = FastAPI()


# Configure CORS settings to allow requests from specific origins
origins = [
    "http://localhost",
    "http://localhost:3000"
]

# Add CORS middleware to the FastAPI application
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# tomato_model = tf.keras.models.load_model("../Model/Tomato.h5")
# corn_model = tf.keras.models.load_model("../Model/corn.h5")
# sugarcane_model= tf.keras.models.load_model("../Model/Sugarcane.h5")
# tea_model= tf.keras.models.load_model("../Model/Sugarcane.h5")
# grape_model= tf.keras.models.load_model("../Model/Sugarcane.h5")
# potato_model= tf.keras.models.load_model("../Model/Potato.h5")


# Define the URLs for the models store on Google drive
tomato_model_url = "https://drive.google.com/uc?export=download&id=18YxyfsBYfHkILP8Dg3TuO2EgouvMfTuV"
corn_model_url = "https://drive.google.com/uc?export=download&id=1cCDHwtVNTIRvLnh-9VXERonI5CnvvS6r"
sugarcane_model_url = "https://drive.google.com/uc?export=download&id=1_3O4ac7KOmZfToGibnh5aFOdeUR5RuU8"
tea_model_url = "https://drive.google.com/uc?export=download&id=13p2cEYw3ngxuHQmEJURIk3nSluakgJ_-"
grape_model_url = "https://drive.google.com/uc?export=download&id=1LgWpLkw1DeQvaskUeb7M3JKbYtI_y_hA"
potato_model_url = "https://drive.google.com/uc?export=download&id=1XO8PrxE-xTRiSfN6vInrpXodF-sTvysn"


# Function to load a model from a URL
def load_model_from_url(url):
    response = requests.get(url)
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(response.content)
    temp_file.close()
    model = joblib.load(temp_file.name)
    return model

# Load the models from Google drive using the function
tomato_model = load_model_from_url(tomato_model_url)
corn_model = load_model_from_url(corn_model_url)
sugarcane_model = load_model_from_url(sugarcane_model_url)
tea_model = load_model_from_url(tea_model_url)
grape_model = load_model_from_url(grape_model_url)
potato_model = load_model_from_url(potato_model_url)



# Define a route for the '/ping' endpoint, accessible via HTTP GET request
@app.get("/ping")
async def ping():
    return "Hello, I am alive"

# Define a utility function to read the uploaded file as an image and convert it to a numpy array
def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data))
    # Resize the image to 256x256
    image = image.resize((256, 256))
    image_array = np.array(image)
    return image_array

# localhost http://10.0.2.2:8000/predict
@app.post("/predict")
async def predict(
    plant: str = Form(...), #plant name
    file: UploadFile = Form(...)  # image file
):
    # Read the uploaded image file and convert it to a numpy array
    image = read_file_as_image(await file.read())

    # Convert the image into a batch format with an additional dimension
    img_batch = np.expand_dims(image, 0)

    
    if plant == "Tomato":
        predictions = tomato_model.predict(img_batch)
        predicted_class = CLASS_NAMES_Tomato[np.argmax(predictions[0])] # Find the index of the class with the highest probability and get its corresponding class name
        solution = disease_solution_Tomato.get(predicted_class, "No solution found.") # Get the solution for the predicted disease from the dictionary
        name = disease_name_Tomato.get(predicted_class, "No name found.")
    elif plant == "Corn":
        predictions = corn_model.predict(img_batch)
        predicted_class = CLASS_NAMES_Corn[np.argmax(predictions[0])]
        solution = disease_solution_Corn.get(predicted_class, "No solution found.")
        name = disease_name_Corn.get(predicted_class, "No name found.")
    elif plant == "Sugarcane":
        predictions = sugarcane_model.predict(img_batch)
        predicted_class = CLASS_NAMES_Sugarcane[np.argmax(predictions[0])]
        solution = disease_solution_Sugarcane.get(predicted_class, "No solution found.")
        name = disease_name_Sugarcane.get(predicted_class, "No name found.")
    elif plant == "Tea":
        predictions = tea_model.predict(img_batch)
        predicted_class = CLASS_NAMES_Tea[np.argmax(predictions[0])]
        solution = disease_solution_Tea.get(predicted_class, "No solution found.")
        name = disease_name_Tea.get(predicted_class, "No name found.")
    elif plant == "Grape":
        predictions = grape_model.predict(img_batch)
        predicted_class = CLASS_NAMES_Grape[np.argmax(predictions[0])]
        solution = disease_solution_Grape.get(predicted_class, "No solution found.")
        name = disease_name_Grape.get(predicted_class, "No name found.")
    elif plant == "Potato":
        predictions = potato_model.predict(img_batch)
        predicted_class = CLASS_NAMES_Potato[np.argmax(predictions[0])]
        solution = disease_solution_Potato.get(predicted_class, "No solution found.")
        name = disease_name_Potato.get(predicted_class, "No name found.")
    else:
        return {
            'error': 'Invalid plant name.'
        }

    
    
    # Get the highest probability value as the confidence level
    confidence = np.max(predictions[0]*100)
    confidence = "{:.2f}".format(confidence)

    # Return the predicted class and confidence level as a JSON response
    return {
        'disease': name,
        'confidence': float(confidence),
        'solution': solution
    }


class FeedbackModel(BaseModel):
    name: str
    feedback: str
    rating: str

def get_mongo_client():
    client = MongoClient("mongodb+srv://sajeewa:sajeewa1234@cluster0.stk5p6n.mongodb.net/?retryWrites=true&w=majority")
    return client

@app.post("/feedback")
async def feedback(rating: str = Form(...), name: str =Form(...), feedback: str = Form(...)):
    if rating and feedback:
        # Store the feedback and rating in MongoDB
        feedback_data = FeedbackModel(rating=rating, name=name, feedback=feedback)
        client = get_mongo_client()
        db = client["feedback_db"]
        feedback_collection = db["feedbacks"]
        feedback_collection.insert_one(feedback_data.dict())
        client.close()
        return {"message": "Thank you for your feedback!"}
    else:
        return {"error": "Both rating and feedback are required."}

@app.get("/feedbacks", response_model=List[FeedbackModel])
def get_feedbacks():
    client = get_mongo_client()
    db = client["feedback_db"]
    feedback_collection = db["feedbacks"]

    # Fetch all feedbacks from MongoDB
    feedbacks = list(feedback_collection.find({}))
    
    client.close()
    return feedbacks


# Run the FastAPI application with Uvicorn server on localhost and port 8000
if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)


#Define the class names and solutions for each plant
CLASS_NAMES_Tomato = ["Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
               "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
               "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy"]
disease_solution_Tomato = {
    "Tomato___Bacterial_spot": "Follow crop rotation, sanitization, water management, fungicides, resistant varieties, biocontrol, and proper soil nutrition. Copper fungicides prevent bacterial spots, while resistant varieties and beneficial bacteria aid in biocontrol. Proper soil nutrition supports plant growth and resistance.",
    "Tomato___Early_blight": "Implementing crop rotation, adequate spacing, watering at the base, and pruning techniques can help manage early blight. Wet leaves should be avoided, as they can encourage blight. Natural copper fungicides, chlorothalonil, or mancozeb can be used as foliar sprays or granules for control.",
    "Tomato___Late_blight": "Use resistant tomato varieties like Brandywine, Iron Lady, Mountain Magic, and Mountain Merit, rotate crops, prune, stake, use fungicides, and harvest ripe tomatoes promptly. Clean up debris and sanitize the environment by discarding or burning infected plants.",
    "Tomato___Leaf_Mold": "Proper air circulation is essential for preventing tomato leaf mold. Proper soil watering, fungicide use, and crop rotation are crucial. Infected leaves should be removed promptly and plants should be rotated to prevent fungus spread.",
    "Tomato___Septoria_leaf_spot": "Follow crop rotation, proper drainage, pruning lower leaves, using fungicides, choosing disease-resistant varieties, providing good air circulation, and practicing sanitation. Rotate crops, prune branches, and ensure proper air circulation to prevent fungal growth and spread.",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Water spraying, neem oil, insecticidal soaps, and predatory mites can eliminate spider mites from tomato plants. Neem oil disrupts mite life cycles, while insecticidal soaps kill them weekly. Predatory mites feed on spider mites, providing biological control.",
    "Tomato___Target_Spot": "Identify and remove infected plants, avoid planting tomatoes in the same spot for three years, use fungicides, manage water effectively, maintain soil fertility, avoid excessive nitrogen use, plant resistant varieties, practice good hygiene, and avoid soil spreading.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Use resistant tomato varieties, use reflective mulch, control whiteflies with insecticides or natural predators, and use physical barriers like netting or lacewings. Physical barriers like netting can also help prevent whiteflies from accessing plants.",
    "Tomato___Tomato_mosaic_virus": "Maintain good garden sanitation by cleaning tools, gloves, and supplies with disinfectant. Choose resistant tomato varieties, control pests with insecticides, and avoid smoking around tomatoes. If infected, remove plants, keep the area clean, monitor other plants, and consider hydroponically growing tomatoes to prevent infection.",
    "Tomato___healthy": "No disease detected. Your tomato plant is healthy!",
}
disease_name_Tomato = {
    "Tomato___Bacterial_spot": "Bacterial spot",
    "Tomato___Early_blight": "Early blight",
    "Tomato___Late_blight": "Late blight",
    "Tomato___Leaf_Mold": "Leaf Mold",
    "Tomato___Septoria_leaf_spot": "Septoria leaf spot",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Spider mites",
    "Tomato___Target_Spot": "Target Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Yellow Leaf Curl Virus",
    "Tomato___Tomato_mosaic_virus": "Mosaic Virus",
    "Tomato___healthy": "healthy",
}


CLASS_NAMES_Corn = ["Corn__common_rust","Corn__gray_leaf_spot","Corn__healthy","Corn__northern_leaf_blight"]
disease_solution_Corn = {
    "Corn__common_rust": "Rotate crops, use resistant corn varieties early, use fungicides, remove infected plants, maintain plant health with proper nutrition and irrigation, and regularly monitor for rust infection signs. Use fungicides only when necessary and remove infected plants.",
    "Corn__gray_leaf_spot": "Controlling gray leaf spot in corn requires crop rotation, resistant varieties, fungicide application, timely harvest, and cultural practices. Regular selection, fungicide use, and timely harvest reduce disease severity. Proper fertilization, irrigation, and weed control reduce stress on plants.",
    "Corn__northern_leaf_blight": "Apply fungicides like azoxystrobin, pyraclostrobin, and trifloxystrobin before symptoms appear. Practice crop rotation, select resistant varieties, and maintain good plant hygiene by removing residues and controlling weeds. These measures ensure healthy yields and prevent Northern Leaf Blight.",
    "Corn__healthy": "No disease detected. Your plant is healthy!",
}
disease_name_Corn = {
    "Corn__common_rust": "common rust",
    "Corn__gray_leaf_spot": "gray leaf spot",
    "Corn__northern_leaf_blight": "northern leaf blight",
    "Corn__healthy": "healthy",
}


CLASS_NAMES_Sugarcane= ["Sugarcane__bacterial_blight","Sugarcane__healthy","Sugarcane__red_rot","Sugarcane__rust"]
disease_solution_Sugarcane = {
    "Sugarcane__bacterial_blight": "Implement sanitation practices, use resistant cultivars, rotate with non-host crops, use chemical control, monitor fields for bacterial blight, improve plant nutrition, and maintain balanced fertilization and soil pH levels to prevent disease spread and minimize crop damage.",
    "Sugarcane__red_rot": "Effective red rot control requires crop rotation, disease-free seedlings, field sanitation, fungicides, biological control agents, nematode management, irrigation, drainage, nutrition, timely harvesting, and continuous monitoring. Implement nematode management strategies and maintain optimal soil moisture, drainage, and nutrient levels for resistance.",
    "Sugarcane__rust": "Crop rotation, fungicide use, pruning, hygiene, and resistant varieties are crucial for breaking infection cycles in sugarcane crops. Consult a professional for optimal fungicide and application methods, practice good sanitation, and grow sugarcane varieties resistant to rust for better resistance.",
    "Sugarcane__healthy": "No disease detected. Your plant is healthy!",
}
disease_name_Sugarcane = {
    "Sugarcane__bacterial_blight": "bacterial blight",
    "Sugarcane__red_rot": "Sugarcane red rot",
    "Sugarcane__rust": "Sugarcane rust",
    "Sugarcane__healthy": "healthy",
}


CLASS_NAMES_Tea= ["Tea__algal_leaf","Tea__anthracnose","Tea__bird_eye_spot","Tea__brown_blight","Tea__healthy","Tea__red_leaf_spot"]
disease_solution_Tea = {
    "Tea__algal_leaf": "Maintain a healthy plant environment by following good sanitation practices, using proper irrigation, using plant-resistant varieties, applying fungicides according to label rates, and implementing Integrated Pest Management (IPM) practices. Take preventive measures, such as sterilizing gardening tools and avoiding touching infected leaves.",
    "Tea__anthracnose": "Sanitation, cultural practices, fungicide, biological control, and plant resistance prevent tea anthracnose spread. Proper pruning, thinning and irrigation avoids wetting foliage. Copper-based products, mancozeb, chlorothalonil control anthracnose. Trichoderma spp. helps. Planting resistant cultivars reduces disease occurrence.",
    "Tea__bird_eye_spot": "Proper plant spacing, sunlight, and nitrogen use are crucial for healthy plants. Avoid over-fertilization, reduce nitrogen fertilizer usage, and use drip irrigation. Eliminate weeds, debris, and pests using natural predators like ladybugs and praying mantis.",
    "Tea__healthy": "No disease detected. Your plant is healthy!",
    "Tea__brown_blight": "Use disease-resistant tea cultivars, maintain optimal growing conditions, monitor and remove infected plant parts, use fungicides like tebuconazole, propiconazole, or mancozeb, practice good crop rotation, and prune tea plants regularly. Regular pruning increases light penetration and air circulation, reducing the risk of infection.",
    "Tea__red_leaf_spot": "Sanitation, fungicide application, and plant health improvement are crucial for preventing disease spread in tea plants. Use registered fungicides, fertilizers, and proper irrigation to maintain plant health. Adjust plant spacing and rotate crops regularly to reduce soil build-up.",
}
disease_name_Tea = {
    "Tea__algal_leaf": "algal leaf",
    "Tea__anthracnose": "anthracnose",
    "Tea__bird_eye_spot": "bird eye spott",
    "Tea__healthy": "healthy",
    "Tea__brown_blight": "brown_blight",
    "Tea__red_leaf_spot": "red leaf spot",
}


CLASS_NAMES_Grape =['Grape__black_measles','Grape__black_rot','Grape__healthy','Grape__leaf_blight_(isariopsis_leaf_spot)']
disease_solution_Grape={
    "Grape__black_measles": "Regular monitoring, pruning, fungicide sprays, vineyard hygiene, proper irrigation, and fertilization are essential for maintaining soil health and reducing fungal infections. Plant resistant or tolerant cultivars, low-watering techniques, beneficial microorganisms, sanitizing pruning tools, and crop rotation are also beneficial.",
    "Grape__black_rot": "Recommends pruning vines, removing infected plant material, using fungicides, and using plant-resistant grape varieties. Proper irrigation, fertilization, and pest control are crucial for maintaining healthy vines and preventing fungus growth. Apply fungicides at the right time and concentration.",
    "Grape__healthy": "No disease detected. Your plant is healthy!",
    "Grape__leaf_blight_(isariopsis_leaf_spot)": "Prune and thin grape vines to boost sunlight and airflow for fungal infection prevention. Use copper, sulfur, or triazole fungicides. Adopt IPM involving cultural practices, organic pesticides, and helpful insects. Ensure sanitization, dispose of infected foliage and fruit, and sterilize pruning tools for effective clean-up.",
}
disease_name_Grape={
    "Grape__black_measles": "black measles",
    "Grape__black_rot": "black rot",
    "Grape__healthy": "healthy",
    "Grape__leaf_blight_(isariopsis_leaf_spot)": "leaf_blight",
}


CLASS_NAMES_Potato =['Potato__early_blight', 'Potato__healthy', 'Potato__late_blight']
disease_solution_Potato={
    "Potato__early_blight": "Follow crop rotation, mulch, watering, fertilizing, pruning, and using fungicides. Plant potatoes in different locations each year, avoid over-fertilizing, prune off infected leaves and stems, and use appropriate fungicides if the disease persists. Prevent disease by planting resistant varieties and following good planting practices.",
    "Potato__healthy":"No disease detected. Your plant is healthy!",
    "Potato__late_blight":  "Implement crop rotation, resistant varieties, sanitation, fungicide application, timely watering, adequate plant spacing, and regular inspections. Avoid planting consecutively, use disease-resistant varieties, follow label instructions, and ensure proper sanitation and watering.",
}
disease_name_Potato={
    "Potato__early_blight": "early blight",
    "Potato__healthy": "healthy",
    "Potato__late_blight": "late blight",
}