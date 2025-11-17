class Index:
    def __init__(self, name, key, token, base_url):
        self.name = name
        self.key = key
        self.token = token
        self.base_url = base_url

    def __str__(self):
        return self.name

    def upsert(self, vectors):
        return Index(f"{self.token}[{upsert}]")

    def query(self, vector):
        return Index(f"{self.token}[{query}]")

    def delete(self, id):
        return Index(f"{self.token}[{delete}]")
