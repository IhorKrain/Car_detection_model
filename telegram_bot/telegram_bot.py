# Import necessary libraries
import os
import logging
import tempfile
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,  # Fixes the NameError
)
from ultralytics import YOLO
import cv2
from dotenv import load_dotenv

# Загрузите переменные из .env
load_dotenv()

# Вместо явного указания токена используйте
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load YOLOv8 model (replace with your custom model path)
model = YOLO("/car_detection_model/best.pt")  # Example: "runs/train/exp/weights/best.pt"

# Define class IDs for detection (adjust based on your model)
CAR_CLASSES = [2, 5, 7]  # Default IDs for car, bus, truck in COCO dataset


# /start command handler
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a welcome message when the /start command is issued."""
    await update.message.reply_text("🚗 Hello! Send me a photo/video, and I'll detect vehicles!")


# Photo processing handler
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Process incoming photos and return detection results."""
    try:
        # Download the photo
        photo_file = await update.message.photo[-1].get_file()

        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix=".jpg") as input_file, \
                tempfile.NamedTemporaryFile(suffix=".jpg") as output_file:

            # Download and process image
            await photo_file.download_to_drive(input_file.name)
            results = model.predict(input_file.name, classes=CAR_CLASSES)

            # Visualize and save results
            res_plotted = results[0].plot()
            cv2.imwrite(output_file.name, res_plotted)

            # Send processed image back
            await update.message.reply_photo(photo=output_file.name)

    except Exception as e:
        logger.error(f"Photo processing error: {e}")
        await update.message.reply_text("⚠️ Error processing photo. Please try again.")


# Video processing handler
async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Process incoming videos and return detection results."""
    try:
        video_file = await update.message.video.get_file()

        with tempfile.NamedTemporaryFile(suffix=".mp4") as input_file, \
                tempfile.NamedTemporaryFile(suffix=".mp4") as output_file:

            # Download video
            await video_file.download_to_drive(input_file.name)

            # Process video frames
            cap = cv2.VideoCapture(input_file.name)
            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))

            # Configure video writer
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(
                output_file.name,
                fourcc,
                20.0,
                (frame_width, frame_height))

            # Process each frame
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Perform detection
                results = model.predict(frame, classes=CAR_CLASSES)
                frame_with_boxes = results[0].plot()
                out.write(frame_with_boxes)

            # Release resources
            cap.release()
            out.release()

            # Send processed video
            await update.message.reply_video(video=output_file.name)

    except Exception as e:
        logger.error(f"Video processing error: {e}")
        await update.message.reply_text("⚠️ Error processing video. Please try again.")


# Error handler
async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log errors and notify users."""
    logger.error(f"Error: {context.error}")
    await update.message.reply_text("⚠️ An error occurred. Please try again later.")


# Main function
def main() -> None:
    """Start the bot."""
    # Replace with your bot token
    TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

    # Create application
    application = Application.builder().token(TOKEN).build()

    # Register handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(MessageHandler(filters.VIDEO, handle_video))
    application.add_error_handler(error_handler)

    # Start polling
    application.run_polling()


if __name__ == "__main__":
    main()