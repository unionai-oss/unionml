import modelz

api_key = "mzi-f1e7034393e5e5b3e0927f6c3c2ac58b"
client = modelz.ModelzClient(key=api_key, deployment="modelz-test3-stablediffusion-73kbi10qro7qhuzi")
resp = client.inference(params="a horse in a green field", serde="msgpack")
print(resp.resp.content)
resp.save_to_file("inference.jpg")
