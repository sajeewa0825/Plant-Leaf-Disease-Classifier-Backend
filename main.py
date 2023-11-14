from fastapi import FastAPI, UploadFile, Form
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
from keras.models import load_model

app = FastAPI()


# Configure CORS settings to allow requests from specific origins
origins = ["http://localhost", "http://localhost:3000"]

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
# tomato_model_url = "https://drive.google.com/uc?export=download&id=18YxyfsBYfHkILP8Dg3TuO2EgouvMfTuV"
# corn_model_url = "https://drive.google.com/uc?export=download&id=1cCDHwtVNTIRvLnh-9VXERonI5CnvvS6r"
sugarcane_model_url = "https://drive.google.com/uc?export=download&id=1_3O4ac7KOmZfToGibnh5aFOdeUR5RuU8"
# tea_model_url = "https://drive.google.com/uc?export=download&id=13p2cEYw3ngxuHQmEJURIk3nSluakgJ_-"
# grape_model_url = "https://drive.google.com/uc?export=download&id=1LgWpLkw1DeQvaskUeb7M3JKbYtI_y_hA"
# potato_model_url = "https://drive.google.com/uc?export=download&id=1XO8PrxE-xTRiSfN6vInrpXodF-sTvysn"



# Function to load a model from a URL
def load_model_from_url(url):
    response = requests.get(url)
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(response.content)
    temp_file.close()
    model = load_model(temp_file.name)
    return model


# Load the models from Google drive using the function
# tomato_model = load_model_from_url(tomato_model_url)
# corn_model = load_model_from_url(corn_model_url)
sugarcane_model = load_model_from_url(sugarcane_model_url)
# tea_model = load_model_from_url(tea_model_url)
# grape_model = load_model_from_url(grape_model_url)
# potato_model = load_model_from_url(potato_model_url)


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
async def predict(language: str, plant: str = Form(...), file: UploadFile = Form(...)):
    print(language)
    # Read the uploaded image file and convert it to a numpy array
    image = read_file_as_image(await file.read())

    # Convert the image into a batch format with an additional dimension
    img_batch = np.expand_dims(image, 0)
    print(language)

    if language == "si":
        if plant == "Tomato":
            predictions = tomato_model.predict(img_batch)
            predicted_class = CLASS_NAMES_Tomato[
                np.argmax(predictions[0])
            ]  # Find the index of the class with the highest probability and get its corresponding class name
            solution = disease_solution_Tomato_Si.get(
                predicted_class, "No solution found."
            )  # Get the solution for the predicted disease from the dictionary
            name = disease_name_Tomato_Si.get(predicted_class, "No name found.")
        elif plant == "Corn":
            predictions = corn_model.predict(img_batch)
            predicted_class = CLASS_NAMES_Corn[np.argmax(predictions[0])]
            solution = disease_solution_Corn_Si.get(
                predicted_class, "No solution found."
            )
            name = disease_name_Corn_Si.get(predicted_class, "No name found.")
        elif plant == "Sugarcane":
            predictions = sugarcane_model.predict(img_batch)
            predicted_class = CLASS_NAMES_Sugarcane[np.argmax(predictions[0])]
            solution = disease_solution_Sugarcane_Si.get(
                predicted_class, "No solution found."
            )
            name = disease_name_Sugarcane_Si.get(predicted_class, "No name found.")
        elif plant == "Tea":
            predictions = tea_model.predict(img_batch)
            predicted_class = CLASS_NAMES_Tea[np.argmax(predictions[0])]
            solution = disease_solution_Tea_Si.get(
                predicted_class, "No solution found."
            )
            name = disease_name_Tea_Si.get(predicted_class, "No name found.")
        elif plant == "Grape":
            predictions = grape_model.predict(img_batch)
            predicted_class = CLASS_NAMES_Grape[np.argmax(predictions[0])]
            solution = disease_solution_Grape_Si.get(
                predicted_class, "No solution found."
            )
            name = disease_name_Grape_Si.get(predicted_class, "No name found.")
        elif plant == "Potato":
            predictions = potato_model.predict(img_batch)
            predicted_class = CLASS_NAMES_Potato[np.argmax(predictions[0])]
            solution = disease_solution_Potato_Si.get(
                predicted_class, "No solution found."
            )
            name = disease_name_Potato_Si.get(predicted_class, "No name found.")
        else:
            return {"error": "Invalid plant name."}

    else:
        if plant == "Tomato":
            predictions = tomato_model.predict(img_batch)
            predicted_class = CLASS_NAMES_Tomato[
                np.argmax(predictions[0])
            ]  # Find the index of the class with the highest probability and get its corresponding class name
            solution = disease_solution_Tomato.get(
                predicted_class, "No solution found."
            )  # Get the solution for the predicted disease from the dictionary
            name = disease_name_Tomato.get(predicted_class, "No name found.")
        elif plant == "Corn":
            predictions = corn_model.predict(img_batch)
            predicted_class = CLASS_NAMES_Corn[np.argmax(predictions[0])]
            solution = disease_solution_Corn.get(predicted_class, "No solution found.")
            name = disease_name_Corn.get(predicted_class, "No name found.")
        elif plant == "Sugarcane":
            predictions = sugarcane_model.predict(img_batch)
            predicted_class = CLASS_NAMES_Sugarcane[np.argmax(predictions[0])]
            solution = disease_solution_Sugarcane.get(
                predicted_class, "No solution found."
            )
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
            solution = disease_solution_Potato.get(
                predicted_class, "No solution found."
            )
            name = disease_name_Potato.get(predicted_class, "No name found.")
        else:
            return {"error": "Invalid plant name."}

    # Get the highest probability value as the confidence level
    confidence = np.max(predictions[0] * 100)
    confidence = "{:.2f}".format(confidence)

    # Return the predicted class and confidence level as a JSON response
    return {"disease": name, "confidence": float(confidence), "solution": solution}


class FeedbackModel(BaseModel):
    name: str
    feedback: str
    rating: str


def get_mongo_client():
    client = MongoClient(
        "mongodb+srv://sajeewa:sajeewa1234@cluster0.stk5p6n.mongodb.net/?retryWrites=true&w=majority"
    )
    return client


@app.post("/feedback")
async def feedback(
    rating: str = Form(...), name: str = Form(...), feedback: str = Form(...)
):
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
    uvicorn.run(app, host="localhost", port=8000)


# Define the class names and solutions for each plant
# Define the class names and solutions for each plant
CLASS_NAMES_Tomato = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]
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
disease_solution_Tomato_Si = {
    "Tomato___Bacterial_spot": "බෝග භ්‍රමණය, සනීපාරක්ෂාව, ජල කළමනාකරණය, දිලීර නාශක, ප්‍රතිරෝධී වර්ග, ජෛව පාලන සහ නිසි පාංශු පෝෂණය අනුගමනය කරන්න. තඹ දිලීර නාශක බැක්ටීරියා ලප වළක්වන අතර ප්‍රතිරෝධී ප්‍රභේද සහ හිතකර බැක්ටීරියා ජෛව පාලනයට උපකාරී වේ. නිසි පාංශු පෝෂණය ශාක වර්ධනයට සහ ප්‍රතිරෝධයට සහාය වේ. ",
    "Tomato___Early_blight": "බෝග භ්‍රමණය ක්‍රියාත්මක කිරීම, ප්‍රමාණවත් පරතරයක් තැබීම, ශාකයේ පාමුලට ජලය දැමීම සහ කප්පාදු කිරීමේ ක්‍රම මගින් මුල් අංගමාරය පාලනය කිරීමට උපකාරී වේ. තෙත් කොළ වලින් වැළකී සිටිය යුතුය, ඒවා අංගමාරය දිරිමත් කළ හැකිය. ස්වාභාවික තඹ දිලීර නාශක, ක්ලෝරෝතලෝනිල් හෝ මැන්කොසෙබ් භාවිතා කළ හැකිය. පාලනය සඳහා පත්‍ර ඉසින හෝ කැටිති.",
    "Tomato___Late_blight": "Brandywine, Iron Lady, Mountain Magic සහ Mountain Merit වැනි ප්‍රතිරෝධී තක්කාලි වර්ග භාවිතා කරන්න, භෝග කරකවන්න, කප්පාදු කරන්න, කොටස් කරන්න, දිලීර නාශක භාවිතා කරන්න, සහ ඉදුණු තක්කාලි කඩිනමින් අස්වැන්න කරන්න. ආසාදිත සුන්බුන් පිරිසිදු කර පරිසරය පිරිසිදු කරන්න පැල.",
    "Tomato___Leaf_Mold": "තක්කාලි පත්‍ර පුස් වැලැක්වීම සඳහා නිසි වායු සංසරණය අත්‍යවශ්‍ය වේ. නිසි පාංශු ජලය දැමීම, දිලීර නාශක භාවිතය සහ බෝග භ්‍රමණය ඉතා වැදගත් වේ. ආසාදිත පත්‍ර කඩිනමින් ඉවත් කළ යුතු අතර දිලීර පැතිරීම වැළැක්වීම සඳහා පැල කරකැවිය යුතුය.",
    "Tomato___Septoria_leaf_spot": "බෝග භ්‍රමණය, නිසි ජලාපවහනය, පහළ කොළ කප්පාදු කිරීම, දිලීර නාශක භාවිතා කිරීම, රෝග-ප්‍රතිරෝධී ප්‍රභේද තෝරා ගැනීම, හොඳ වායු සංසරණය සැපයීම සහ සනීපාරක්ෂාව අනුගමනය කරන්න. බෝග කරකවන්න, අතු කප්පාදු කරන්න, සහ දිලීර වර්ධනය වැළැක්වීම සඳහා නිසි වායු සංසරණය සහතික කරන්න. බෝ වීම.",
    "Tomato___Spider_mites Two-spotted_spider_mite": "ජලය ඉසීම, නෙම් තෙල්, කෘමිනාශක සබන් සහ ආක්රමණශීලී මයිටාවන්ට තක්කාලි පැල වලින් මකුළු මයිටාවන් තුරන් කළ හැක. නෙම් තෙල් මයිටා ජීවන චක්‍ර කඩාකප්පල් කරන අතර, කෘමිනාශක සබන් සතිපතා ඒවා විනාශ කරයි. විලෝපික මයිටාවන් පෝෂණය කරයි. ජීව විද්‍යාත්මක පාලනය",
    "Tomato___Target_Spot": "ආසාදිත පැල හඳුනාගෙන ඉවත් කරන්න, වසර තුනක් එකම ස්ථානයේ තක්කාලි සිටුවීමෙන් වළකින්න, දිලීර නාශක භාවිතා කරන්න, ජලය ඵලදායී ලෙස කළමනාකරණය කරන්න, පාංශු සාරවත් බව පවත්වා ගන්න, අධික නයිට්‍රජන් භාවිතය වැළැක්වීම, ප්‍රතිරෝධී ප්‍රභේද, හොඳ සනීපාරක්ෂාව පුරුදු කිරීම සහ පස පැතිරීම වැළැක්වීම ",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "ප්‍රතිරෝධී තක්කාලි ප්‍රභේද භාවිතා කරන්න, පරාවර්තක වසුන් භාවිතා කරන්න, කෘමිනාශක හෝ ස්වභාවික විලෝපිකයන් සමඟ සුදු මැස්සන් පාලනය කරන්න, සහ දැල් හෝ ලේස් පියාපත් වැනි භෞතික බාධක භාවිතා කරන්න. දැල් දැමීම වැනි භෞතික බාධක ද සුදු මැස්සන් ශාකවලට ප්‍රවේශ වීම වැළැක්වීමට උපකාරී වේ.",
    "Tomato___Tomato_mosaic_virus": "විෂබීජ නාශක සමඟ මෙවලම්, අත්වැසුම් සහ සැපයුම් පිරිසිදු කිරීමෙන් හොඳ ගෙවතු සනීපාරක්ෂාව පවත්වා ගන්න. ප්‍රතිරෝධී තක්කාලි ප්‍රභේද තෝරන්න, කෘමිනාශක සමඟ පළිබෝධ පාලනය කරන්න, සහ තක්කාලි අවට දුම් පානය කිරීමෙන් වළකින්න. ආසාදනය වී ඇත්නම්, ශාක ඉවත් කරන්න, ප්‍රදේශය පිරිසිදුව තබා ගන්න, අනෙකුත් ශාක නිරීක්ෂණය කරන්න. , සහ ආසාදනය වැලැක්වීම සඳහා හයිඩ්‍රොපොනිකව වැඩෙන තක්කාලි සලකා බලන්න.",
    "Tomato___healthy": "කිසිදු රෝගයක් අනාවරණය වී නොමැත.",
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

disease_name_Tomato_Si = {
    "Tomato___Bacterial_spot": "බැක්ටීරීයා ආසාදිත ප්‍රදේශය",
    "Tomato___Early_blight": "පෙර අංගමාරය",
    "Tomato___Late_blight": "පසු අංගමාරය",
    "Tomato___Leaf_Mold": "පත්‍රවල හැඩය මියයාම",
    "Tomato___Septoria_leaf_spot": "සෙප්ටෝරියා පත්‍ර ලප ආසාදිත ප්‍රදේශය",
    "Tomato___Spider_mites Two-spotted_spider_mite": "මකුළු මයිටා ආසාදිත ප්‍රදේශ",
    "Tomato___Target_Spot": "Target Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "කහ පත්‍ර  ක්ලුරල් වෛරසය",
    "Tomato___Tomato_mosaic_virus": "මොසෙයික් වෛරසය",
    "Tomato___healthy": "ඔබේ ශාකය සෞඛ්‍ය සම්පන්නයි!",
}


CLASS_NAMES_Corn = [
    "Corn__common_rust",
    "Corn__gray_leaf_spot",
    "Corn__healthy",
    "Corn__northern_leaf_blight",
]
disease_solution_Corn = {
    "Corn__common_rust": "Rotate crops, use resistant corn varieties early, use fungicides, remove infected plants, maintain plant health with proper nutrition and irrigation, and regularly monitor for rust infection signs. Use fungicides only when necessary and remove infected plants.",
    "Corn__gray_leaf_spot": "Controlling gray leaf spot in corn requires crop rotation, resistant varieties, fungicide application, timely harvest, and cultural practices. Regular selection, fungicide use, and timely harvest reduce disease severity. Proper fertilization, irrigation, and weed control reduce stress on plants.",
    "Corn__northern_leaf_blight": "Apply fungicides like azoxystrobin, pyraclostrobin, and trifloxystrobin before symptoms appear. Practice crop rotation, select resistant varieties, and maintain good plant hygiene by removing residues and controlling weeds. These measures ensure healthy yields and prevent Northern Leaf Blight.",
    "Corn__healthy": "No disease detected. Your plant is healthy!",
}
disease_solution_Corn_Si = {
    "Corn__common_rust": "බෝග කරකවන්න, ප්‍රතිරෝධී ඉරිඟු වර්ග කලින් භාවිතා කරන්න, දිලීර නාශක භාවිතා කරන්න, ආසාදිත ශාක ඉවත් කරන්න, නිසි පෝෂණය හා වාරිමාර්ග සමඟ ශාක සෞඛ්‍යය පවත්වා ගන්න, සහ මලකඩ ආසාදන සලකුණු සඳහා නිතිපතා නිරීක්ෂණය කරන්න. අවශ්‍ය විට පමණක් දිලීර නාශක භාවිතා කර ආසාදිත ශාක ඉවත් කරන්න.",
    "Corn__gray_leaf_spot": "ඉරිඟු වල අළු පත්‍ර ලප පාලනය කිරීම සඳහා බෝග භ්‍රමණය, ප්‍රතිරෝධී ප්‍රභේද, දිලීර නාශක යෙදීම, කාලෝචිත අස්වැන්න සහ සංස්කෘතික පිළිවෙත් අවශ්‍ය වේ. නිතිපතා තෝරා ගැනීම, දිලීර නාශක භාවිතය සහ නියමිත වේලාවට අස්වනු නෙලීම රෝගයේ බරපතලකම අඩු කරයි. නිසි පොහොර යෙදීම, වාරිමාර්ග සහ වල් මර්දනය අඩු කරයි. ශාක මත ආතතිය.",
    "Corn__northern_leaf_blight": "රෝග ලක්ෂණ මතුවීමට පෙර azoxystrobin, pyraclostrobin, සහ trifloxystrobin වැනි දිලීර නාශක යොදන්න. බෝග මාරු කිරීම, ප්‍රතිරෝධී ප්‍රභේද තෝරා ගැනීම සහ අපද්‍රව්‍ය ඉවත් කිරීම සහ වල් පැලෑටි පාලනය කිරීම මගින් හොඳ ශාක සනීපාරක්ෂාව පවත්වා ගැනීම. මෙම පියවර මගින් නිරෝගී අස්වැන්නක් සහ උතුරු කොළ B ආලෝකය වළක්වයි.",
    "Corn__healthy": "කිසිදු රෝගයක් අනාවරණය වී නොමැත.",
}
disease_name_Corn = {
    "Corn__common_rust": "common rust",
    "Corn__gray_leaf_spot": "gray leaf spot",
    "Corn__northern_leaf_blight": "northern leaf blight",
    "Corn__healthy": "healthy",
}

disease_name_Corn_Si = {
    "Corn__common_rust": "පොදු මලකඩ",
    "Corn__gray_leaf_spot": "අළු පත්‍ර ලප ආසාදිත ප්‍රදේශය",
    "Corn__northern_leaf_blight": "උතුරු පත්‍ර අංගමාරය",
    "Corn__healthy": "ඔබේ ශාකය සෞඛ්‍ය සම්පන්නයි!",
}


CLASS_NAMES_Sugarcane = [
    "Sugarcane__bacterial_blight",
    "Sugarcane__healthy",
    "Sugarcane__red_rot",
    "Sugarcane__rust",
]
disease_solution_Sugarcane = {
    "Sugarcane__bacterial_blight": "Implement sanitation practices, use resistant cultivars, rotate with non-host crops, use chemical control, monitor fields for bacterial blight, improve plant nutrition, and maintain balanced fertilization and soil pH levels to prevent disease spread and minimize crop damage.",
    "Sugarcane__red_rot": "Effective red rot control requires crop rotation, disease-free seedlings, field sanitation, fungicides, biological control agents, nematode management, irrigation, drainage, nutrition, timely harvesting, and continuous monitoring. Implement nematode management strategies and maintain optimal soil moisture, drainage, and nutrient levels for resistance.",
    "Sugarcane__rust": "Crop rotation, fungicide use, pruning, hygiene, and resistant varieties are crucial for breaking infection cycles in sugarcane crops. Consult a professional for optimal fungicide and application methods, practice good sanitation, and grow sugarcane varieties resistant to rust for better resistance.",
    "Sugarcane__healthy": "No disease detected. Your plant is healthy!",
}
disease_solution_Sugarcane_Si = {
    "Sugarcane__bacterial_blight": "සනීපාරක්ෂක පිළිවෙත් ක්‍රියාත්මක කිරීම, ප්‍රතිරෝධී වගාවන් භාවිතා කිරීම, සත්කාරක නොවන බෝග සමඟ භ්‍රමණය වීම, රසායනික පාලනය භාවිතා කිරීම, බැක්ටීරියා අංගමාරය සඳහා ක්ෂේත්‍ර අධීක්ෂණය කිරීම, ශාක පෝෂණය වැඩි දියුණු කිරීම, සහ රෝග පැතිරීම වැළැක්වීම සහ බෝග හානි අවම කිරීම සඳහා සමතුලිත පොහොර සහ පසෙහි pH අගය පවත්වා ගැනීම. ",
    "Sugarcane__red_rot": "ඵලදායී රතු කුණුවීම පාලනය කිරීම සඳහා බෝග මාරු කිරීම, රෝගවලින් තොර බීජ පැල, ක්ෂේත්‍ර සනීපාරක්ෂාව, දිලීර නාශක, ජීව විද්‍යාත්මක පාලන කාරක, නෙමටෝඩා කළමනාකරණය, වාරිමාර්ග, ජලාපවහනය, පෝෂණය, කාලෝචිත අස්වනු නෙලීම සහ අඛණ්ඩ අධීක්ෂණය අවශ්‍ය වේ. නෙමටෝඩා කළමනාකරණ උපාය මාර්ග ක්‍රියාත්මක කිරීම සහ ප්‍රශස්ත ලෙස පවත්වා ගැනීම. පාංශු තෙතමනය, ජලාපවහනය සහ ප්‍රතිරෝධය සඳහා පෝෂක මට්ටම්.",
    "Sugarcane__rust": "උක් බෝග වල ආසාදන චක්‍ර බිඳ දැමීම සඳහා බෝග භ්‍රමණය, දිලීර නාශක භාවිතය, කප්පාදු කිරීම, සනීපාරක්ෂාව සහ ප්‍රතිරෝධී ප්‍රභේද ඉතා වැදගත් වේ. ප්‍රශස්ත දිලීර නාශක සහ යෙදුම් ක්‍රම සඳහා වෘත්තිකයෙකුගෙන් උපදෙස් ලබා ගන්න, හොඳ සනීපාරක්ෂාව පුහුණු කරන්න, සහ මලකඩ වලට ඔරොත්තු දෙන උක් වර්ග වගා කරන්න. වඩා හොඳ ප්රතිරෝධයක්.",
    "Sugarcane__healthy": "කිසිදු රෝගයක් අනාවරණය වී නොමැත.",
}
disease_name_Sugarcane = {
    "Sugarcane__bacterial_blight": "bacterial blight",
    "Sugarcane__red_rot": "Sugarcane red rot",
    "Sugarcane__rust": "Sugarcane rust",
    "Sugarcane__healthy": "healthy",
}
disease_name_Sugarcane_Si = {
    "Sugarcane__bacterial_blight": "බැක්ටීරියා අංගමාරය",
    "Sugarcane__red_rot": "උක්ගස් රතු කුණුවීම",
    "Sugarcane__rust": "උක්ගස් මලකඩ",
    "Sugarcane__healthy": "ඔබේ ශාකය සෞඛ්‍ය සම්පන්නයි!",
}


CLASS_NAMES_Tea = [
    "Tea__algal_leaf",
    "Tea__anthracnose",
    "Tea__bird_eye_spot",
    "Tea__brown_blight",
    "Tea__healthy",
    "Tea__red_leaf_spot",
]
disease_solution_Tea = {
    "Tea__algal_leaf": "Maintain a healthy plant environment by following good sanitation practices, using proper irrigation, using plant-resistant varieties, applying fungicides according to label rates, and implementing Integrated Pest Management (IPM) practices. Take preventive measures, such as sterilizing gardening tools and avoiding touching infected leaves.",
    "Tea__anthracnose": "Sanitation, cultural practices, fungicide, biological control, and plant resistance prevent tea anthracnose spread. Proper pruning, thinning and irrigation avoids wetting foliage. Copper-based products, mancozeb, chlorothalonil control anthracnose. Trichoderma spp. helps. Planting resistant cultivars reduces disease occurrence.",
    "Tea__bird_eye_spot": "Proper plant spacing, sunlight, and nitrogen use are crucial for healthy plants. Avoid over-fertilization, reduce nitrogen fertilizer usage, and use drip irrigation. Eliminate weeds, debris, and pests using natural predators like ladybugs and praying mantis.",
    "Tea__healthy": "No disease detected. Your plant is healthy!",
    "Tea__brown_blight": "Use disease-resistant tea cultivars, maintain optimal growing conditions, monitor and remove infected plant parts, use fungicides like tebuconazole, propiconazole, or mancozeb, practice good crop rotation, and prune tea plants regularly. Regular pruning increases light penetration and air circulation, reducing the risk of infection.",
    "Tea__red_leaf_spot": "Sanitation, fungicide application, and plant health improvement are crucial for preventing disease spread in tea plants. Use registered fungicides, fertilizers, and proper irrigation to maintain plant health. Adjust plant spacing and rotate crops regularly to reduce soil build-up.",
}
disease_solution_Tea_Si = {
    "Tea__algal_leaf": "හොඳ සනීපාරක්ෂක පිළිවෙත් අනුගමනය කිරීම, නිසි වාරිමාර්ග භාවිතා කිරීම, ශාක-ප්‍රතිරෝධී වර්ග භාවිතා කිරීම, ලේබල් අනුපාත අනුව දිලීර නාශක යෙදීම සහ ඒකාබද්ධ පළිබෝධ කළමනාකරණ (IPM) පිළිවෙත් ක්‍රියාත්මක කිරීම මගින් සෞඛ්‍ය සම්පන්න ශාක පරිසරයක් පවත්වාගෙන යාම. විෂබීජහරණය වැනි වැළැක්වීමේ පියවර ගන්න. ගෙවතු වගා මෙවලම් සහ ආසාදිත කොළ ස්පර්ශ කිරීමෙන් වැළකීම.",
    "Tea__anthracnos": "සනීපාරක්ෂාව, සංස්කෘතික පිළිවෙත්, දිලීර නාශක, ජීව විද්‍යාත්මක පාලනය සහ ශාක ප්‍රතිරෝධය තේ ඇන්ත්‍රැක්නෝස් පැතිරීම වළක්වයි. නිසි ලෙස කප්පාදු කිරීම, තුනී කිරීම සහ වාරිමාර්ග මගින් ශාක පත්‍ර තෙත් කිරීම වළක්වයි. තඹ මත පදනම් වූ නිෂ්පාදන, මැන්කොසෙබ්, ක්ලෝරෝතලෝනිල් ඇන්ත්‍රැක්නෝස් පාලනය කරයි. ට්‍රයිකොඩර්මා spp. සිටුවීමට උපකාරී වේ. ප්‍රතිරෝධී ප්‍රභේද රෝග ඇතිවීම අඩු කරයි.",
    "Tea__bird_eye_spot": "නිරෝගී ශාක සඳහා නිසි ශාක පරතරය, හිරු එළිය සහ නයිට්‍රජන් භාවිතය ඉතා වැදගත් වේ. අධික පොහොර යෙදීමෙන් වළකින්න, නයිට්‍රජන් පොහොර භාවිතය අඩු කරන්න, සහ බිංදු වාරිමාර්ග භාවිතා කරන්න. ලේඩි බග්ස් සහ ප්‍රේයං මැන්ටිස් වැනි ස්වභාවික විලෝපිකයන් භාවිතා කරමින් වල් පැලෑටි, සුන්බුන් සහ පළිබෝධ ඉවත් කරන්න. ",
    "Tea__healthy": "කිසිදු රෝගයක් අනාවරණය වී නොමැත.",
    "Tea__brown_blight": "රෝග-ප්‍රතිරෝධී තේ වගාවන් භාවිතා කිරීම, ප්‍රශස්ත වර්ධනය වන තත්ත්වයන් පවත්වා ගැනීම, ආසාදිත ශාක කොටස් නිරීක්ෂණය කිරීම සහ ඉවත් කිරීම, ටෙබුකොනසෝල්, ප්‍රොපිකොනසෝල් හෝ මැන්කොසෙබ් වැනි දිලීර නාශක භාවිතා කිරීම, හොඳ බෝග භ්‍රමණයක් පුහුණු කිරීම සහ තේ පැල නිතිපතා කප්පාදු කිරීම. නිතිපතා කප්පාදු කිරීම ආලෝකය විනිවිද යාම වැඩි කරයි. සහ වායු සංසරණය, ආසාදන අවදානම අඩු කරයි.",
    "Tea__red_leaf_spot": "තේ පැලවල රෝග පැතිරීම වැලැක්වීම සඳහා සනීපාරක්ෂාව, දිලීර නාශක යෙදීම සහ ශාක සෞඛ්‍ය වැඩිදියුණු කිරීම ඉතා වැදගත් වේ. ශාක සෞඛ්‍යය පවත්වා ගැනීමට ලියාපදිංචි දිලීර නාශක, පොහොර සහ නිසි වාරිමාර්ග භාවිතා කරන්න. පස ගොඩනැගීම අඩු කිරීම සඳහා ශාක පරතරය සකස් කර බෝග නිතිපතා කරකවන්න- ඉහළට.",
}
disease_name_Tea = {
    "Tea__algal_leaf": "algal leaf",
    "Tea__anthracnose": "anthracnose",
    "Tea__bird_eye_spot": "bird eye spott",
    "Tea__healthy": "healthy",
    "Tea__brown_blight": "brown_blight",
    "Tea__red_leaf_spot": "red leaf spot",
}

disease_name_Tea_Si = {
    "Tea__algal_leaf": "ඇල්ගල් පත්‍ර",
    "Tea__anthracnose": "ඇන්ත්‍රැක්නෝස්",
    "Tea__bird_eye_spot": "ආසාදිත ප්‍රදේශය තුල කුරුලු ඇස් වැනි ලප ඇතිවීම",
    "Tea__healthy": "ඔබේ ශාකය සෞඛ්‍ය සම්පන්නයි!",
    "Tea__brown_bligh": "තේ දුඹුරු දිලීරය",
    "Tea__red_leaf_spot": "රතු පත්‍ර ලප ආසාදිත ප්‍රදේශ",
}

CLASS_NAMES_Grape = [
    "Grape__black_measles",
    "Grape__black_rot",
    "Grape__healthy",
    "Grape__leaf_blight_(isariopsis_leaf_spot)",
]
disease_solution_Grape = {
    "Grape__black_measles": "Regular monitoring, pruning, fungicide sprays, vineyard hygiene, proper irrigation, and fertilization are essential for maintaining soil health and reducing fungal infections. Plant resistant or tolerant cultivars, low-watering techniques, beneficial microorganisms, sanitizing pruning tools, and crop rotation are also beneficial.",
    "Grape__black_rot": "Recommends pruning vines, removing infected plant material, using fungicides, and using plant-resistant grape varieties. Proper irrigation, fertilization, and pest control are crucial for maintaining healthy vines and preventing fungus growth. Apply fungicides at the right time and concentration.",
    "Grape__healthy": "No disease detected. Your plant is healthy!",
    "Grape__leaf_blight_(isariopsis_leaf_spot)": "Prune and thin grape vines to boost sunlight and airflow for fungal infection prevention. Use copper, sulfur, or triazole fungicides. Adopt IPM involving cultural practices, organic pesticides, and helpful insects. Ensure sanitization, dispose of infected foliage and fruit, and sterilize pruning tools for effective clean-up.",
}
disease_solution_Grape_Si = {
    "Grape__black_measles": "පාංශු සෞඛ්‍යය පවත්වා ගැනීම සහ දිලීර ආසාදන අවම කිරීම සඳහා නිතිපතා අධීක්ෂණය, කප්පාදු කිරීම, දිලීර නාශක ඉසින, මිදි වතු සනීපාරක්ෂාව, නිසි වාරිමාර්ග සහ පොහොර යෙදීම අත්‍යවශ්‍ය වේ. ශාක ප්‍රතිරෝධී හෝ ඔරොත්තු දෙන වගාවන්, අඩු ජලය දැමීමේ ක්‍රම, ප්‍රයෝජනවත් ක්ෂුද්‍ර ජීවීන්, සනීපාරක්ෂක මෙවලම් සහ බෝග මාරුව ද ප්රයෝජනවත් වේ.",
    "Grape__black_rot": "වැල් කප්පාදු කිරීම, ආසාදිත ශාක ද්‍රව්‍ය ඉවත් කිරීම, දිලීර නාශක භාවිතා කිරීම සහ ශාක-ප්‍රතිරෝධී මිදි වර්ග භාවිතා කිරීම නිර්දේශ කරයි. නිසි වාරිමාර්ග, පොහොර යෙදීම සහ පළිබෝධ පාලනය වැල් සෞඛ්‍ය සම්පන්නව පවත්වා ගැනීමට සහ දිලීර වර්ධනය වැළැක්වීම සඳහා ඉතා වැදගත් වේ. නියම වේලාවට දිලීර නාශක යොදන්න. සහ සාන්ද්රණය.",
    "Grape__healthy": "කිසිදු රෝගයක් අනාවරණය වී නොමැත.",
    "Grape__leaf_blight_(isariopsis_leaf_spot)": "දිලීර ආසාදන වැලැක්වීම සඳහා හිරු එළිය සහ වාතය වැඩි කිරීමට මිදි වැල් කප්පාදු කර තුනී කරන්න. තඹ, සල්ෆර් හෝ ට්‍රයිසෝල් දිලීර නාශක භාවිතා කරන්න. සංස්කෘතික භාවිතයන්, කාබනික පළිබෝධනාශක, සහ කෘමිනාශකවලට ප්‍රයෝජනවත් වන කෘමිනාශක ඇතුළත් IPM භාවිතා කරන්න. ආසාදිත පත්‍ර සහ පලතුරු, සහ ඵලදායී පිරිසිදු කිරීම සඳහා කප්පාදු මෙවලම් විෂබීජහරණය කරන්න.",
}
disease_name_Grape = {
    "Grape__black_measles": "black measles",
    "Grape__black_rot": "black rot",
    "Grape__healthy": "healthy",
    "Grape__leaf_blight_(isariopsis_leaf_spot)": "leaf blight",
}
disease_name_Grape_Si = {
    "Grape__black_measles": "කළු සරම්ප",
    "Grape__black_rot": "කලු කුණුවීම",
    "Grape__healthy": "ඔබේ ශාකය සෞඛ්‍ය සම්පන්නයි!",
    "Grape__leaf_blight_(isariopsis_leaf_spot)": "පත්‍ර  අංගමාරය",
}

CLASS_NAMES_Potato = ["Potato__early_blight", "Potato__healthy", "Potato__late_blight"]
disease_solution_Potato = {
    "Potato__early_blight": "Follow crop rotation, mulch, watering, fertilizing, pruning, and using fungicides. Plant potatoes in different locations each year, avoid over-fertilizing, prune off infected leaves and stems, and use appropriate fungicides if the disease persists. Prevent disease by planting resistant varieties and following good planting practices.",
    "Potato__healthy": "No disease detected. Your plant is healthy!",
    "Potato__late_blight": "Implement crop rotation, resistant varieties, sanitation, fungicide application, timely watering, adequate plant spacing, and regular inspections. Avoid planting consecutively, use disease-resistant varieties, follow label instructions, and ensure proper sanitation and watering.",
}
disease_solution_Potato_Si = {
    "Potato__early_blight": "බෝග මාරු කිරීම, වසුන් යෙදීම, ජලය දැමීම, පොහොර යෙදීම, කප්පාදු කිරීම සහ දිලීර නාශක භාවිතා කිරීම අනුගමනය කරන්න. සෑම වසරකම විවිධ ස්ථානවල අර්තාපල් සිටුවන්න, අධික පොහොර යෙදීමෙන් වළකින්න, ආසාදිත කොළ සහ කඳන් කප්පාදු කරන්න, සහ රෝගය දිගටම පවතී නම් සුදුසු දිලීර නාශක භාවිතා කරන්න. ප්‍රතිරෝධී ප්‍රභේද සිටුවීමෙන් සහ හොඳ රෝපණ පිළිවෙත් අනුගමනය කිරීමෙන් රෝග වළක්වා ගන්න.",
    "Potato__healthy": "කිසිදු රෝගයක් අනාවරණය වී නොමැත.",
    "Potato__late_blight": "බෝග භ්‍රමණය, ප්‍රතිරෝධී ප්‍රභේද, සනීපාරක්ෂාව, දිලීර නාශක යෙදීම, කලට වේලාවට ජලය දැමීම, ප්‍රමාණවත් ශාක පරතරයක් සහ නිරන්තර පරීක්‍ෂණ ක්‍රියාත්මක කරන්න. අඛණ්ඩව සිටුවීමෙන් වළකින්න, රෝග-ප්‍රතිරෝධී ප්‍රභේද භාවිතා කරන්න, ලේබල් උපදෙස් පිළිපදින්න, සහ නිසි සනීපාරක්ෂාව සහ ජලය සැපයීම සහතික කරන්න.",
}
disease_name_Potato = {
    "Potato__early_blight": "early blight",
    "Potato__healthy": "healthy",
    "Potato__late_blight": "late blight",
}
disease_name_Potato_Si = {
    "Potato__early_blight": "පෙර අංගමාරය",
    "Potato__healthy": "ඔබේ ශාකය සෞඛ්‍ය සම්පන්නයි!",
    "Patato__late_blight": "පසු අංගමාරය",
}
