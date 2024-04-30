import quanto
import torch
from config import config
from torch.quantization import (
    default_observer,
    HistogramObserver,
    MinMaxObserver,
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
    PerChannelMinMaxObserver,
    QuantWrapper,
    QConfig,
)

# See https://pytorch.org/blog/quantization-in-practice/#affine-and-symmetric-quantization-schemes

def no_quantize(model, calibration_data):
    return model


def _pytorch_quantize(model, calibration_data, weight_observer):
    # Prepare the model for quantization, observing weights with the given observerl.
    # Note: we specifically do not quantize the activations.
    model = QuantWrapper(model)
    model.qconfig = QConfig(activation=default_observer, weight=weight_observer)
    torch.quantization.prepare(model, inplace=True)

    # Run calibration examples through the model. We throw away the results since the
    # goal is only to calibrate the quantization observer.
    with torch.no_grad():
        for batch, _ in calibration_data:
            model(batch)

    # Actually convert the model with the calibrated quantizer
    torch.quantization.convert(model, inplace=True)
    return model


def affine_minmax_per_tensor(model, calibration_data):
    return _pytorch_quantize(
        model,
        calibration_data,
        MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_affine),
    )


def symmetric_minmax_per_tensor(model, calibration_data):
    return _pytorch_quantize(
        model,
        calibration_data,
        MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric),
    )


def affine_moving_avg_per_tensor(model, calibration_data):
    return _pytorch_quantize(
        model,
        calibration_data,
        MovingAverageMinMaxObserver.with_args(
            dtype=torch.qint8, qscheme=torch.per_tensor_affine
        ),
    )


def symmetric_minmax_per_channel(model, calibration_data):
    return _pytorch_quantize(
        model,
        calibration_data,
        PerChannelMinMaxObserver.with_args(
            dtype=torch.qint8, qscheme=torch.per_channel_symmetric
        ),
    )


def affine_minmax_per_channel(model, calibration_data):
    return _pytorch_quantize(
        model,
        calibration_data,
        PerChannelMinMaxObserver.with_args(
            dtype=torch.qint8, qscheme=torch.per_channel_affine
        ),
    )


def symmetric_moving_avg_per_channel(model, calibration_data):
    return _pytorch_quantize(
        model,
        calibration_data,
        MovingAveragePerChannelMinMaxObserver.with_args(
            dtype=torch.qint8, qscheme=torch.per_channel_symmetric
        ),
    )


def affine_moving_avg_per_channel(model, calibration_data):
    return _pytorch_quantize(
        model,
        calibration_data,
        MovingAveragePerChannelMinMaxObserver.with_args(
            dtype=torch.qint8, qscheme=torch.per_channel_affine
        ),
    )


def affine_histogram_per_tensor(model, calibration_data):
    return _pytorch_quantize(
        model,
        calibration_data,
        HistogramObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_affine),
    )


def symmetric_histogram_per_tensor(model, calibration_data):
    return _pytorch_quantize(
        model,
        calibration_data,
        HistogramObserver.with_args(
            dtype=torch.qint8, qscheme=torch.per_tensor_symmetric
        ),
    )

def _quanto_quantize(model, calibration_data, weights_dtype):
    quanto.quantize(model, weights=weights_dtype)
    with quanto.Calibration():
        for batch, labels in calibration_data:
            batch = batch.to(config.runtime.device)
            labels = labels.to(config.runtime.device)
            model(batch)
    quanto.freeze(model)
    return model

def quanto_int8_quantize(model, calibration_data):
    return _quanto_quantize(model, calibration_data, quanto.qint8)

def quanto_int4_quantize(model, calibration_data):
    return _quanto_quantize(model, calibration_data, quanto.qint4)

def quanto_int2_quantize(model, calibration_data):
    return _quanto_quantize(model, calibration_data, quanto.qint2)

def quanto_float8_quantize(model, calibration_data):
    return _quanto_quantize(model, calibration_data, quanto.qfloat8)