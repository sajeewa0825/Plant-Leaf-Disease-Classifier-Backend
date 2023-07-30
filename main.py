from fastapi import FastAPI, File, UploadFile
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


MODEL = tf.keras.models.load_model("../Model/Tomato.h5")

# Define the class names for the different tomato leaf diseases and "healthy"
CLASS_NAMES = ["Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
               "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
               "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy"]


# Define a dictionary with leaf diseases and their corresponding solutions
disease_solutions = {
    "Tomato___Bacterial_spot": "Solution for Bacterial Spot...",
    "Tomato___Early_blight": "Solution for Early Blight...",
    "Tomato___Late_blight": "Solution for Late Blight...",
    "Tomato___Leaf_Mold": "Solution for Leaf Mold...",
    "Tomato___Septoria_leaf_spot": "Solution for Septoria Leaf Spot...",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Solution for Spider Mites...",
    "Tomato___Target_Spot": "Solution for Target Spot...",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Solution for Yellow Leaf Curl Virus...",
    "Tomato___Tomato_mosaic_virus": "Solution for Tomato Mosaic Virus...",
    "Tomato___healthy": "No disease detected. Your tomato plant is healthy!",
}

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
    file: UploadFile = File(...)  # This parameter represents the uploaded file (in this case, an image)
):
    # Read the uploaded image file and convert it to a numpy array
    image = read_file_as_image(await file.read())

    # Convert the image into a batch format with an additional dimension
    img_batch = np.expand_dims(image, 0)

    # Make predictions using the loaded machine learning model
    predictions = MODEL.predict(img_batch)

    # Find the index of the class with the highest probability and get its corresponding class name
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]

    # Get the highest probability value as the confidence level
    confidence = np.max(predictions[0])

     # Get the solution for the predicted disease from the dictionary
    solution = disease_solutions.get(predicted_class, "No solution found.")

    # Return the predicted class and confidence level as a JSON response
    return {
        'class': predicted_class,
        'confidence': float(confidence),
        'solution': solution
    }

# Run the FastAPI application with Uvicorn server on localhost and port 8000
if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
