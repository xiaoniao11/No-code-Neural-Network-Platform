import sqlite3
from typing import Dict, Any

class DatabaseManager:
    def __init__(self, db_path: str = "neural_network.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 创建模型表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    architecture TEXT NOT NULL,
                    parameters TEXT NOT NULL,
                    created_at INTEGER DEFAULT (strftime('%s', 'now'))
                )
            ''')
            
            # 创建数据集表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS datasets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    preprocessing_params TEXT,
                    created_at INTEGER DEFAULT (strftime('%s', 'now'))
                )
            ''')
            
            conn.commit() 
    
    def get_connection(self):
        """获取数据库连接"""
        return sqlite3.connect(self.db_path) 
    
    def get_all_models(self) -> list:
        """获取所有已保存的模型"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, name, created_at 
                FROM models 
                ORDER BY created_at DESC
                """
            )
            return cursor.fetchall()
    
    def get_model_by_id(self, model_id: int) -> dict:
        """根据ID获取模型"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT name, architecture, parameters 
                FROM models 
                WHERE id = ?
                """,
                (model_id,)
            )
            result = cursor.fetchone()
            if result:
                return {
                    "name": result[0],
                    "architecture": result[1],
                    "parameters": result[2]
                }
            return None 