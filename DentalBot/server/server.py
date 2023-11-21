from fastapi import FastAPI, HTTPException

app = FastAPI()

# Import the prediction function
from DentalBot.model.showandnoshow import predict_show_no_show

# Define a route to handle incoming messages from the UI
@app.post("/webhooks/rest/webhook")
async def process_message(message: dict):
    # Extract user inputs from the incoming request
    customer_name = message["message"]["content"]["customer_name"]
    phone_number = message["message"]["content"]["phone_number"]
    email = message["message"]["content"]["email"]

    # Ensure that all required inputs are provided
    if not customer_name or not phone_number or not email:
        raise HTTPException(status_code=400, detail="Missing required inputs")

    # Call the prediction function
    prediction_result = predict_show_no_show(customer_name, phone_number, email)

    # Return the response back to the UI
    response = {"text": prediction_result}
    return response
