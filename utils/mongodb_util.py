import pymongo

myclient = pymongo.MongoClient("mongodb://10.10.10.17:27017/")

db = myclient.admin    #
db.authenticate("reader", "R@rtkp")
mydb = myclient["result"]
