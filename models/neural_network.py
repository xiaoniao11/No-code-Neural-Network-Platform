import json
import torch
import torch.nn as nn
from typing import List, Dict, Any
from database.db_manager import DatabaseManager

class NNLayer:
    def __init__(self, layer_type: str, params: Dict[str, Any]):
        self.type = layer_type
        self.params = params
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "params": self.params
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NNLayer':
        return cls(data["type"], data["params"])
    
    def to_pytorch(self) -> nn.Module:
        """将层配置转换为PyTorch层"""
        if self.type == "Conv2d":
            return nn.Conv2d(**self.params)
        elif self.type == "MaxPool2d":
            return nn.MaxPool2d(**self.params)
        elif self.type == "AvgPool2d":
            return nn.AvgPool2d(**self.params)
        elif self.type == "Linear":
            return nn.Linear(**self.params)
        elif self.type == "Relu":
            return nn.ReLU(inplace=True)
        elif self.type == "Sigmoid":
            return nn.Sigmoid()
        elif self.type == "Tanh":
            return nn.Tanh()
        else:
            raise ValueError(f"不支持的层类型: {self.type}")

class NNModel(nn.Module):
    def __init__(self):
        super(NNModel, self).__init__()
        self.layers: List[NNLayer] = []
        self.db = DatabaseManager()
        self.pytorch_layers = nn.ModuleList()
    
    def add_layer(self, layer: NNLayer):
        self.layers.append(layer)
        self.pytorch_layers.append(layer.to_pytorch())
    
    def forward(self, x):
        for layer in self.pytorch_layers:
            x = layer(x)
        return x
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "layers": [layer.to_dict() for layer in self.layers]
        }
    
    def save(self, name: str = "default_model"):
        """保存模型到数据库"""
        model_data = json.dumps(self.to_dict())
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO models (name, architecture, parameters)
                VALUES (?, ?, ?)
                """,
                (name, model_data, "{}")
            )
        # 同时保存PyTorch模型参数
        torch.save(self.state_dict(), f"{name}.pth")
    
    @classmethod
    def load(cls, model_id: int = None) -> 'NNModel':
        """从数据库加载模型"""
        model = cls()
        model_data = model.db.get_model_by_id(model_id)
        if model_data:
            data = json.loads(model_data["architecture"])
            for layer_data in data["layers"]:
                model.add_layer(NNLayer.from_dict(layer_data))
            # 如果存在对应的参数文件，加载参数
            try:
                model.load_state_dict(torch.load(f"{model_data['name']}.pth"))
            except:
                pass
            return model
        raise Exception("找不到指定的模型") 