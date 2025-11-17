class VectorX:
    def __init__(self, token:string):
        self.token = token

    def __str__(self):
        return self.token
    
    def create_index(self, index:int):
        return VectorX(f"{self.token}[{index}]")

    def list_indexes(self, index:int):
        return VectorX(f"{self.token}[{index}]")
    
    def delete_index(self, index:int):
        return VectorX(f"{self.token}[{index}]")
    
    