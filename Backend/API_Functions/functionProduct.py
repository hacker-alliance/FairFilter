# Requires pymongo
import pymongo

client = pymongo.MongoClient(
    "mongodb+srv://mongoUser:WBxiojjvekSwQLQuJYWn@fair-filter-cluster-qxumh.gcp.mongodb.net/test?retryWrites=true&w=majority")
db = client.fairfilter_db
col = db.fairfilter_collection

record = {
    "ASIN": "TEST",
    "category": "TEST"
}


def lookup(ASIN):
    # col.insert_one(record)
    return col.find_one({"ASIN": ASIN})


def product(request):
    """Responds to any HTTP request.
    Args:
        request (flask.Request): HTTP request object.
    Returns:
        The response text or any set of values that can be turned into a
        Response object using
        `make_response <http://flask.pocoo.org/docs/1.0/api/#flask.Flask.make_response>`.
    """
    print(lookup("TEST"))

    request_json = request.get_json()
    if request.args and 'message' in request.args:
        return request.args.get('message')
    elif request_json and 'message' in request_json:
        return request_json['message']
    else:
        return f'Hello World!'
