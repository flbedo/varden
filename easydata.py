import sqlite3
from typing import Any, Union, Dict, List
from contextlib import contextmanager

@contextmanager
def _get_connection(f_name: str):
    conn = sqlite3.connect(f'{f_name}.db')
    try:
        yield conn
    finally:
        conn.close()

def _parse_value(value: str) -> Any:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            if value.lower() == 'true':
                return True
            elif value.lower() == 'false':
                return False
            elif value == 'None':
                return None
            else:
                return value

def _serialize_value(value: Any) -> str:
    if isinstance(value, bool):
        return 'true' if value else 'false'
    elif value is None:
        return 'None'
    else:
        return str(value)

def create_database(f_name: str) -> None:
    conn = sqlite3.connect(f'{f_name}.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS data (
            id TEXT NOT NULL,
            item TEXT NOT NULL,
            value TEXT,
            PRIMARY KEY (id, item)
        )
    ''')
    conn.commit()
    conn.close()

def get_id_data(f_name: str, id) -> Union[Dict[str, Any], None]:
    with _get_connection(f_name) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT item, value FROM data WHERE id = ?', (id,))
        rows = cursor.fetchall()
        
        if not rows: 
            return None
            
        # Оптимизированное преобразование значений
        return {
            item: _parse_value(value) 
            for item, value in rows
        }

def get_item_data(f_name: str, id, item: str) -> Any:
    conn = sqlite3.connect(f'{f_name}.db')
    cursor = conn.cursor()
    cursor.execute(
        'SELECT value FROM data WHERE id = ? AND item = ?', 
        (id, item)
    )
    row = cursor.fetchone()
    conn.close()
    return _parse_value(row[0]) if row else ''

def give_id_data(f_name: str, id, value: Dict[str, Any]) -> Dict[str, Any]:
    with _get_connection(f_name) as conn:
        cursor = conn.cursor()
        
        # Используем транзакцию для группировки операций
        cursor.execute('DELETE FROM data WHERE id = ?', (id,))
        
        # Batch-вставка вместо отдельных INSERT
        data_to_insert = [
            (id, item, _serialize_value(item_value)) 
            for item, item_value in value.items()
        ]
        cursor.executemany(
            'INSERT INTO data (id, item, value) VALUES (?, ?, ?)',
            data_to_insert
        )
        
        conn.commit()
    return value

def give_item_data(f_name: str, id, item: str, value: Any) -> Any:
    conn = sqlite3.connect(f'{f_name}.db')
    cursor = conn.cursor()
    cursor.execute(
        '''INSERT OR REPLACE INTO data (id, item, value)
           VALUES (?, ?, ?)''',
        (id, item, _serialize_value(value))
    )
    conn.commit()
    conn.close()
    return value

def give_all_item_data(f_name: str, item: str, value: Any) -> Any:
    with _get_connection(f_name) as conn:
        cursor = conn.cursor()
        # ОДИН запрос вместо множественных - основной выигрыш в производительности
        cursor.execute('''
            INSERT OR REPLACE INTO data (id, item, value)
            SELECT DISTINCT id, ?, ? FROM data
        ''', (item, _serialize_value(value)))
        conn.commit()
    return value

def delete_id_data(f_name: str, id) -> str:
    conn = sqlite3.connect(f'{f_name}.db')
    cursor = conn.cursor()
    cursor.execute('DELETE FROM data WHERE id = ?', (id,))
    conn.commit()
    conn.close()
    return id

def extremum_item_data(f_name: str, item: str, max_or_min: str) -> Any:
    with _get_connection(f_name) as conn:
        cursor = conn.cursor()
        
        if max_or_min == 'max':
            cursor.execute('''
                SELECT value FROM data 
                WHERE item = ? AND value GLOB '*[0-9]*'
                ORDER BY CAST(value AS REAL) DESC 
                LIMIT 1
            ''', (item,))
        else:  # min
            cursor.execute('''
                SELECT value FROM data 
                WHERE item = ? AND value GLOB '*[0-9]*'
                ORDER BY CAST(value AS REAL) ASC 
                LIMIT 1
            ''', (item,))
        
        result = cursor.fetchone()
        return _parse_value(result[0]) if result else None

def average_item_data(f_name: str, item: str) -> Union[float, None]:
    with _get_connection(f_name) as conn:
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT AVG(CAST(value AS REAL)) 
            FROM data 
            WHERE item = ? AND value GLOB '*[0-9]*'
        ''', (item,))
        
        result = cursor.fetchone()[0]
        return float(result) if result is not None else None
    
def get_ids_by_item(f_name: str, item: str, value: str) -> Dict[str, Dict[str, Any]]:
    with _get_connection(f_name) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT all_data.id, all_data.item, all_data.value
            FROM data all_data
            JOIN (
                SELECT DISTINCT id 
                FROM data 
                WHERE item = ? AND value = ?
            ) city_users ON all_data.id = city_users.id
        ''', (item, value,))
        
        users_data = {}
        for id, item, value in cursor.fetchall():
            if id not in users_data:
                users_data[id] = {}
            users_data[id][item] = _parse_value(value)
        
        return users_data


def ids(f_name: str) -> List[str]:
    conn = sqlite3.connect(f'{f_name}.db')
    cursor = conn.cursor()
    cursor.execute('SELECT DISTINCT id FROM data')
    ids = [row[0] for row in cursor.fetchall()]
    conn.close()
    return ids

def is_id_exist(f_name: str, id) -> bool:
    conn = sqlite3.connect(f'{f_name}.db')
    cursor = conn.cursor()
    cursor.execute('SELECT 1 FROM data WHERE id = ? LIMIT 1', (id,))
    exists = cursor.fetchone() is not None
    conn.close()
    return exists

def is_item_exist(f_name: str, id, item: str) -> bool:
    conn = sqlite3.connect(f'{f_name}.db')
    cursor = conn.cursor()
    cursor.execute(
        'SELECT 1 FROM data WHERE id = ? AND item = ? LIMIT 1', 
        (id, item)
    )
    exists = cursor.fetchone() is not None
    conn.close()
    return exists


if __name__ == '__main__':
    create_database('data')