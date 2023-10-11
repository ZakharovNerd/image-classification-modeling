from fastapi import FastAPI
from pydantic import BaseModel

from src.predict import predict_image

app = FastAPI()


# Define a Pydantic model for input validation
class ImageInput(BaseModel):
    image_path: str


@app.post("/classify/")
async def classify_image(input_data: ImageInput):
    # Call the predict_image function with the provided input
    result = predict_image(input_data.image_path)  # You can specify the device here

    # You can return the result as needed, e.g., as JSON
    return {"result": result}

if __name__ == "__main__":
    import uvicorn

    # Run the FastAPI app using Uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
