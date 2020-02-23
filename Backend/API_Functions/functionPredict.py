# Requires google-cloud-automl
from google.api_core.client_options import ClientOptions
from google.cloud import automl_v1
from google.cloud.automl_v1.proto import service_pb2
import json
from google.protobuf.json_format import MessageToJson


def format_text_payload(content):
    return {'text_snippet': {'content': content, 'mime_type': 'text/plain'}}


def get_prediction(content, model_name):
    options = ClientOptions(api_endpoint='automl.googleapis.com')
    prediction_client = automl_v1.PredictionServiceClient(
        client_options=options)

    payload = format_text_payload(content)
    params = {}
    request = prediction_client.predict(model_name, payload, params)
    return request  # waits until request is returned


def predict(request):
    """Attempts to Predict Category.
    Args:
        request (flask.Request): HTTP request object.
    Returns:
        The response text or any set of values that can be turned into a
        Response object using
        `make_response <http://flask.pocoo.org/docs/1.0/api/#flask.Flask.make_response>`.
    """
    request_json = request.get_json()
    if request_json and 'review_body' in request_json:
        content = request_json['review_body']  # TODO add review_summary
        prediction = get_prediction(
            content, 'projects/207895552307/locations/us-central1/models/TCN5004391989450375168')
        classifications = []
        return MessageToJson(prediction)
    else:
        return f'ERROR: Missing review_body!'
