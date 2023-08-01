from fastapi import FastAPI, UploadFile,Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

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


tomato_model = tf.keras.models.load_model("../Model/Tomato.h5")
corn_model = tf.keras.models.load_model("../Model/corn.h5")
sugarcane_model= tf.keras.models.load_model("../Model/Sugarcane.h5")
tea_model= tf.keras.models.load_model("../Model/Sugarcane.h5")
grape_model= tf.keras.models.load_model("../Model/Sugarcane.h5")
potato_model= tf.keras.models.load_model("../Model/Potato.h5")


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

# Run the FastAPI application with Uvicorn server on localhost and port 8000
if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)


#Define the class names and solutions for each plant
CLASS_NAMES_Tomato = ["Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
               "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
               "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy"]
disease_solution_Tomato = {
    "Tomato___Bacterial_spot": "Solution for Bacterial Spot...",
    "Tomato___Early_blight": "Solution for Early Blight...",
    "Tomato___Late_blight": "Solution for Late Blight...",
    "Tomato___Leaf_Mold": "Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a ",
    "Tomato___Septoria_leaf_spot": "Solution for Septoria Leaf Spot...",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Solution for Spider Mites...",
    "Tomato___Target_Spot": "Solution for Target Spot...",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Solution for Yellow Leaf Curl Virus...",
    "Tomato___Tomato_mosaic_virus": "Solution for Tomato Mosaic Virus...",
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
    "Corn__common_rust": "Solution for common_rust...",
    "Corn__gray_leaf_spot": "Solution for gray_leaf_spot...",
    "Corn__northern_leaf_blight": "Solution for northern_leaf_blight...",
    "Corn__healthy": "No disease detected. Your plant is healthy!",
}
disease_name_Corn = {
    "Corn__common_rust": "common rust",
    "Corn__gray_leaf_spot": "gray_leaf_spot",
    "Corn__northern_leaf_blight": "northern_leaf_blight",
    "Corn__healthy": "healthy",
}


CLASS_NAMES_Sugarcane= ["Sugarcane__bacterial_blight","Sugarcane__healthy","Sugarcane__red_rot","Sugarcane__rust"]
disease_solution_Sugarcane = {
    "Sugarcane__bacterial_blight": "Solution for bacterial_blight...",
    "Sugarcane__red_rot": "Solution for Sugarcane__red_rot...",
    "Sugarcane__rust": "Solution for Sugarcane__rust...",
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
    "Tea__algal_leaf": "Solution for Tea__algal_leaf...",
    "Tea__anthracnose": "Solution for Tea__anthracnose...",
    "Tea__bird_eye_spot": "Tea__bird_eye_spott...",
    "Tea__healthy": "No disease detected. Your plant is healthy!",
    "Tea__brown_blight": "Tea__brown_blight...",
    "Tea__red_leaf_spot": "Tea__red_leaf_spot...",
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
    "Grape__black_measles": "Solution for bacterial_blight...",
    "Grape__black_rot": "Solution for Sugarcane__red_rot...",
    "Grape__healthy": "Solution for Sugarcane__rust...",
    "Grape__leaf_blight_(isariopsis_leaf_spot)": "No disease detected. Your plant is healthy!",
}
disease_name_Grape={
    "Grape__black_measles": "black measles",
    "Grape__black_rot": "black rot",
    "Grape__healthy": "healthy",
    "Grape__leaf_blight_(isariopsis_leaf_spot)": "leaf_blight",
}


CLASS_NAMES_Potato =['Potato__early_blight', 'Potato__healthy', 'Potato__late_blight']
disease_solution_Potato={
    "Potato__early_blight": "Solution for bacterial_blight...",
    "Potato__healthy": "Solution for Sugarcane__red_rot...",
    "Potato__late_blight": "No disease detected. Your plant is healthy!",
}
disease_name_Potato={
    "Potato__early_blight": "early blight",
    "Potato__healthy": "healthy",
    "Potato__late_blight": "late blight",
}