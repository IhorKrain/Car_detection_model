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
import tempfile

load_dotenv()

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load YOLOv8 model
model = YOLO("C:/Users/Public/pycharm/Yolo_detection/car_detection_model/bests.pt")

# Define class IDs for detection
CAR_CLASSES = [2, 5, 7]  # Default IDs for car, bus, truck in COCO dataset

# Create secure temp directory
TEMP_DIR = os.path.join(os.path.expanduser("~"), "yolo_temp")
os.makedirs(TEMP_DIR, exist_ok=True)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a welcome message when the /start command is issued."""
    await update.message.reply_text("🚗 Hello! Send me a photo/video, and I'll detect vehicles!")


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Process incoming photos and return detection results."""
    try:
        photo_file = await update.message.photo[-1].get_file()

        with tempfile.NamedTemporaryFile(dir=TEMP_DIR, suffix=".jpg", delete=False) as input_file, \
                tempfile.NamedTemporaryFile(dir=TEMP_DIR, suffix=".jpg", delete=False) as output_file:

            # Download and process image
            await photo_file.download_to_drive(input_file.name)

            # YOLO prediction
            results = model.predict(input_file.name)

            # Process and save results
            res_plotted = results[0].plot()
            cv2.imwrite(output_file.name, res_plotted)

            # Send processed image
            await update.message.reply_photo(photo=open(output_file.name, 'rb'))

    except Exception as e:
        logger.error(f"Photo processing error: {e}", exc_info=True)
        await update.message.reply_text("⚠️ Error processing photo. Please try again.")
    finally:
        # Cleanup files
        for f in [input_file.name, output_file.name]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception as e:
                    logger.warning(f"Failed to delete {f}: {e}")


async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Process incoming videos and return detection results."""
    try:
        video_file = await update.message.video.get_file()

        with tempfile.NamedTemporaryFile(dir=TEMP_DIR, suffix=".mp4", delete=False) as input_file, \
                tempfile.NamedTemporaryFile(dir=TEMP_DIR, suffix=".mp4", delete=False) as output_file:

            # Download video
            await video_file.download_to_drive(input_file.name)

            # Video processing
            cap = cv2.VideoCapture(input_file.name)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Video writer setup
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_file.name, fourcc, fps, (frame_width, frame_height))

            try:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Detection and writing
                    results = model.predict(frame, classes=CAR_CLASSES)
                    out.write(results[0].plot())
            finally:
                cap.release()
                out.release()

            # Send processed video
            await update.message.reply_video(video=open(output_file.name, 'rb'))

    except Exception as e:
        logger.error(f"Video processing error: {e}", exc_info=True)
        await update.message.reply_text("⚠️ Error processing video. Please try again.")
    finally:
        # Cleanup files
        for f in [input_file.name, output_file.name]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception as e:
                    logger.warning(f"Failed to delete {f}: {e}")


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log errors and notify users."""
    logger.error(f"Error: {context.error}", exc_info=True)
    await update.message.reply_text("⚠️ An error occurred. Please try again later.")


def main() -> None:
    """Start the bot."""
    application = Application.builder().token(os.getenv("TELEGRAM_BOT_TOKEN")).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(MessageHandler(filters.VIDEO, handle_video))
    application.add_error_handler(error_handler)

    application.run_polling()


if __name__ == "__main__":
    main()