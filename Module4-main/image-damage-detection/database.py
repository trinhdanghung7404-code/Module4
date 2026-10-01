import psycopg2
import json

class Database:

    def _normalize_texture_feature(self, value):

        if not value:
            return {"keypoint_count": 0, "keypoints": [], "descriptors": [], "bbox": None}

        if isinstance(value, dict):
            return value

        return json.loads(value)

    def __init__(self):

        self.conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database="damage_detection",
            user="postgres",
            password="123"
        )

        self.cursor = self.conn.cursor()

    def close(self):
        self.cursor.close()
        self.conn.close()

    def save_product(self, feature):

        sql = """
        INSERT INTO product_image_feature(
            image_path,
            object_area,
            perimeter,
            width,
            height,
            texture_feature
        )
        VALUES(%s,%s,%s,%s,%s,%s)
        """

        self.cursor.execute(
            sql,
            (
                feature["image_path"],
                feature["object_area"],
                feature["perimeter"],
                feature["width"],
                feature["height"],
                json.dumps(feature.get("texture_feature", {}))
            )
        )

        self.conn.commit()

    def get_products(self):

        self.cursor.execute("""
            SELECT id, image_path
            FROM product_image_feature
            ORDER BY id
        """)

        rows = self.cursor.fetchall()

        return [
            {
                "id": row[0],
                "image_path": row[1]
            }
            for row in rows
        ]
    
    def get_product_feature(self, product_id):

        self.cursor.execute("""
            SELECT
                id,
                image_path,
                object_area,
                perimeter,
                width,
                height,
                texture_feature
            FROM product_image_feature
            WHERE id = %s
        """, (product_id,))

        row = self.cursor.fetchone()

        if row is None:
            return None

        return {
            "id": row[0],
            "image_path": row[1],
            "object_area": row[2],
            "perimeter": row[3],
            "width": row[4],
            "height": row[5],
            "texture_feature": self._normalize_texture_feature(row[6])
        }
    
    def save_return(self, feature, result):

        sql = """
        INSERT INTO return_image_analysis(
            product_id,
            image_path,
            object_area,
            perimeter,
            width,
            height,
            difference_area,
            largest_damage_area,
            damage_score
        )
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """

        self.cursor.execute(sql, (
            feature["product_id"],
            feature["image_path"],
            feature["object_area"],
            feature["perimeter"],
            feature["width"],
            feature["height"],
            result["difference_area"],
            result["largest_damage_area"],
            result["damage_score"]
        ))

        self.conn.commit()