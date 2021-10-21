import datetime as dt
import json
import os
from typing import Any, List, Tuple
from operator import attrgetter, itemgetter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader as web
import seaborn as sns
from functools import wraps
from dotenv import load_dotenv
from loguru import logger
from telegram import ChatAction
from telegram.ext import CommandHandler, Updater, MessageHandler, Filters, Dispatcher
from telegram.ext.callbackcontext import CallbackContext

from sklearn.preprocessing import MinMaxScaler
from telegram.ext.jobqueue import JobQueue
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

load_dotenv()

DB_FILE_PATH = "db.json"
TEST_PREDICTION_IMAGE_PATH = "test_prediction.png"
CHAT_ID_CRYPTO_INFO = "-698951802"
DB_INITIAL = {
    "config": {
        "crypto_currency": "BTC",
        "against_currency": "USD",
        "prediction_days": 60,
    }
}


def tg_error_handler(update, context: CallbackContext) -> None:
    """
    Error handler for telegram bot.
    See https://github.com/python-telegram-bot/python-telegram-bot/blob/master/examples/errorhandlerbot.py
    """

    logger.error("Exception occurred while handling an update:", exc_info=context.error)


def tg_all_handler(update, context: CallbackContext):
    logger.info("Chat ID: {}", update.effective_message.chat_id)
    logger.info("Received message: {}", update.effective_message.text)


# def start(update, context):
#     context.bot.send_message(
#         chat_id=update.effective_chat.id, text="I'm a bot, please talk to me!"
#     )


def send_typing_action(func):
    """Sends typing action while processing func command."""

    @wraps(func)
    def command_func(update, context, *args, **kwargs):
        context.bot.send_chat_action(
            chat_id=update.effective_message.chat_id, action=ChatAction.TYPING
        )
        return func(update, context, *args, **kwargs)

    return command_func


@logger.catch
def load_db() -> Any:
    with open(DB_FILE_PATH, "r") as db_file:
        return json.load(db_file)


@logger.catch
def save_db(db: Any) -> None:
    with open(DB_FILE_PATH, "w") as db_file:
        json.dump(db, db_file)


@logger.catch
@send_typing_action
def set_config(update, context: CallbackContext):
    logger.info("User requested setting config")

    db = load_db()
    if context.args is None or context.args == []:
        context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="ERROR: Setting name not specified.",
        )
        return
    args = context.args

    if args[0] in db["config"]:
        db["config"][args[0]] = args[1]
        save_db(db)

        # success
        context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=f"Successfully set {args[0]} to {args[1]}",
        )
        logger.info("Setting config successful")
    else:
        context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Please provide a valid setting name!",
        )


@logger.catch
def get_data(
    crypto_currency, against_currency, prediction_days
) -> Tuple[Any, Any, Any, MinMaxScaler]:
    """
    Fetch data from sources using DataReader and fit them into a feature range of 0 to 1.
    """

    start = dt.datetime(2016, 1, 1)
    end = dt.datetime.now()
    data = web.DataReader(f"{crypto_currency}-{against_currency}", "yahoo", start, end)
    logger.info("Yahoo finance data:\n{}", data.head())  # show first few rows of data

    # Prepare data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data["Close"].values.reshape(-1, 1))

    x_train, y_train = [], []

    for x in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[x - prediction_days : x, 0])
        y_train.append(scaled_data[x, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    logger.debug("x_train: {}", x_train)
    logger.debug("y_train: {}", y_train)

    return x_train, y_train, data, scaler


@logger.catch
def create_neural_net(x_train, y_train) -> Sequential:
    """
    Create the model that is used to predict from the data
    """

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(x_train, y_train, epochs=25, batch_size=32)

    return model


@logger.catch
def test_model(
    crypto_currency: str,
    against_currency: str,
    data,
    prediction_days,
    scaler: MinMaxScaler,
    model: Sequential,
):
    test_start = dt.datetime(2020, 1, 1)
    test_end = dt.datetime.now()

    test_data = web.DataReader(
        f"{crypto_currency}-{against_currency}", "yahoo", test_start, test_end
    )
    actual_prices = test_data["Close"].values

    total_dataset = pd.concat((data["Close"], test_data["Close"]), axis=0)

    global model_inputs
    model_inputs = total_dataset[
        len(total_dataset) - len(test_data) - prediction_days :
    ].values
    model_inputs = model_inputs.reshape(-1, 1)  # type: ignore
    model_inputs = scaler.fit_transform(model_inputs)

    x_test = []

    for x in range(prediction_days, len(model_inputs)):
        x_test.append(model_inputs[x - prediction_days : x, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    prediction_prices = model.predict(x_test)
    prediction_prices = scaler.inverse_transform(prediction_prices)

    sns.set_style("whitegrid")  # darkgrid, whitegrid, dark, white and ticks
    plt.rc("axes", titlesize=18)  # fontsize of the axes title
    plt.rc("axes", labelsize=14)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=13)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=13)  # fontsize of the tick labels
    plt.rc("legend", fontsize=13)  # legend fontsize
    plt.rc("font", size=13)  # controls default text sizes

    plt.plot(actual_prices, color="black", label="Actual Prices")
    plt.plot(prediction_prices, color="green", label="Predicted Prices")
    plt.title(f"{crypto_currency} price prediction")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend(loc="upper left")
    plt.savefig(TEST_PREDICTION_IMAGE_PATH)
    plt.clf()


@logger.catch
def predict_next_day(
    prediction_days: int, model: Sequential, scaler: MinMaxScaler
) -> float:
    real_data = [
        model_inputs[len(model_inputs) - prediction_days : len(model_inputs) + 1, 0]
    ]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)

    return prediction[0][0]


@logger.catch
@send_typing_action
def crypto_prediction(update, context: CallbackContext):
    db = load_db()
    logger.debug("loaded DB!")
    logger.debug("DB: {}", db)

    crypto_currency, against_currency, prediction_days = itemgetter(
        "crypto_currency", "against_currency", "prediction_days"
    )(db["config"])

    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=f"Predicting crypto prices for {crypto_currency} based on the past {prediction_days} days.",
    )

    # run the predictions
    x_train, y_train, data, scaler = get_data(
        crypto_currency, against_currency, prediction_days
    )
    model = create_neural_net(x_train, y_train)
    test_model(crypto_currency, against_currency, data, prediction_days, scaler, model)
    prediction = predict_next_day(prediction_days, model, scaler)

    logger.info("Prediction: {}", prediction)

    context.bot.send_photo(
        chat_id=update.effective_chat.id,
        photo=open(TEST_PREDICTION_IMAGE_PATH, "rb"),
    )
    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=f"Tomorrow's {crypto_currency} price is predicted to be at ${prediction}.",
    )
    os.remove(TEST_PREDICTION_IMAGE_PATH)


@send_typing_action
@logger.catch
def send_crypto_update(context: CallbackContext):
    context.bot.send_message(
        chat_id=CHAT_ID_CRYPTO_INFO,
        text="Good morning! Here's your crypto update for today:",
    )
    tg_update = {"effective_chat": {"id": CHAT_ID_CRYPTO_INFO}}
    crypto_prediction(tg_update, context)


@logger.catch
def main():
    updater = Updater(token=os.getenv("TELEGRAM_BOT_TOKEN"), use_context=True)
    dispatcher: Dispatcher = updater.dispatcher
    job_queue: JobQueue = updater.job_queue

    if not os.path.exists(DB_FILE_PATH):
        save_db(DB_INITIAL)

    # dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("predictcrypto", crypto_prediction))
    dispatcher.add_handler(CommandHandler("set", set_config))

    dispatcher.add_handler(MessageHandler(Filters.all, tg_all_handler), group=1)
    dispatcher.add_error_handler(tg_error_handler)

    job_queue.run_daily(send_crypto_update, time=dt.time(hour=7, minute=00, second=00))

    updater.start_polling()
    logger.info("Bot started polling!")
    updater.idle()


if __name__ == "__main__":
    main()
