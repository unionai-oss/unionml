import modelz

api_key = "mzi-921b803eb4931c569153b46dc8356a1a"
client = modelz.ModelzClient(key=api_key, deployment="modelz-test0-deployment-sd-a4mgodx2a4bqangp")
resp = client.inference(params="painting of Napoleon Bonaparte riding a horse", serde="msgpack")
resp.save_to_file("inference3.jpg")
