import os
import sys
from typing import List

import numpy as np
import pandas as pd
import math
import uvicorn
from fastapi import FastAPI
from fastapi import File
from fastapi import UploadFile
from pydantic import BaseModel
from pydantic import Field

PREDICTIONS_LOCAL_PATH = os.path.join(sys.path[0], "data/predictions.csv")

app = FastAPI()
predictions = None


class SKUInfo(BaseModel):
    sku_id: int = Field(..., description="The SKU ud.")
    stock: int = Field(0, description="The current stock level.")


class SKURequest(BaseModel):
    sku: SKUInfo = Field(..., description="The sku and stock level.")
    horizon_days: int = Field(7, description="The number of days in the horizon.")
    confidence_level: float = Field(0.1, description="The confidence level.")


class LowStockSKURequest(BaseModel):
    confidence_level: float = Field(..., description="The confidence level.")
    horizon_days: int = Field(..., description="The number of days in the horizon.")
    # dict of sku and stock
    sku_stock: List[SKUInfo] = Field(..., description="The sku and stock level.")


@app.post("/api/predictions/upload")
def upload_predictions(file: UploadFile = File(...)) -> dict:
    """Upload predictions"""
    try:
        content = file.file.read()
        with open(PREDICTIONS_LOCAL_PATH, "wb") as f:
            f.write(content)

        df = pd.read_csv(PREDICTIONS_LOCAL_PATH)

        global predictions
        predictions = df

        return {"success": 1}
    except Exception as e:
        return {"success": 0, "error": str(e)}


@app.post("/api/how_much_to_order")
def how_much_to_order(request_data: SKURequest) -> dict:
    """Predict how much to order"""
    try:
        sku_id = request_data.sku.sku_id
        current_stock = request_data.sku.stock
        horizon_days = request_data.horizon_days
        confidence_level = request_data.confidence_level

        assert predictions is not None, "Predictions are not loaded"

        prediction = predictions.loc[predictions['sku_id'] == sku_id,
                                     f"pred_{horizon_days}d_q{int(confidence_level * 100)}"].values[0]
        recommended = max(0, math.ceil(prediction - current_stock))
        return {"quantity": recommended}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/stock_level_forecast")
def stock_level_forecast(request_data: SKURequest) -> dict:
    """Predict stock level"""
    try:
        sku_id = request_data.sku.sku_id
        current_stock = request_data.sku.stock
        horizon_days = request_data.horizon_days
        confidence_level = request_data.confidence_level

        assert predictions is not None, "Predictions are not loaded"

        prediction = predictions.loc[predictions['sku_id'] == sku_id,
                                     f"pred_{horizon_days}d_q{int(confidence_level * 100)}"].values[0]
        stock_level = max(0, math.ceil(current_stock - prediction))
        return {"stock_forecast": stock_level}

    except Exception as e:
        return {"error": str(e)}


@app.post("/api/low_stock_sku_list")
def low_stock_sku_list(request_data: LowStockSKURequest) -> dict:
    """Return sku list with low stock level"""
    try:
        confidence_level = request_data.confidence_level
        horizon_days = request_data.horizon_days
        skus = request_data.sku_stock

        assert predictions is not None, "Predictions are not loaded"

        low_stock_list = []
        for sku_info_ in skus:
            id_ = sku_info_.sku_id
            stock_ = sku_info_.stock
            prediction = predictions.loc[predictions['sku_id'] == id_,
                                         f"pred_{horizon_days}d_q{int(confidence_level * 100)}"].values[0]
            if stock_ < prediction:
                low_stock_list.append(id_)

        return {"sku_list": low_stock_list}
    except Exception as e:
        return {"error": str(e)}


def main() -> None:
    """Run application"""
    uvicorn.run("app:app", host="localhost", port=5000)


if __name__ == "__main__":
    main()
    # request_data_1 = SKURequest(sku=SKUInfo(sku_id=38, stock=20))
    # to_order = how_much_to_order(request_data_1)
    # stock_level = stock_level_forecast(request_data_1)
    #
    # import random
    # sku_stocks = [SKUInfo(sku_id=id_, stock=random.randint(0, 100)) for id_ in predictions_['sku_id'].unique()]
    # request_data_2 = LowStockSKURequest(confidence_level=0.1, horizon_days=7, sku_stock=sku_stocks)
    # low_stock = low_stock_sku_list(request_data_2)
    #
    # print()
