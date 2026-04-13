import pandas as pd
import os

class StudentData:
    def __init__(self, csv_path="data/student_records/students.csv"):
        self.csv_path = csv_path
        if not os.path.exists(csv_path):
             raise FileNotFoundError(f"Student data file not found at {csv_path}")
        self.df = pd.read_csv(csv_path)

    def verify_student(self, sid, pin):
        """
        Verifies if SID and PIN match a record.
        """
        sid = int(sid)
        pin = int(pin)
        student = self.df[(self.df['SID'] == sid) & (self.df['PIN'] == pin)]
        if not student.empty:
            return student.iloc[0].to_dict()
        return None

    def get_student_record(self, sid):
        """
        Gets a student record by SID (after auth).
        """
        sid = int(sid)
        student = self.df[self.df['SID'] == sid]
        if not student.empty:
            return student.iloc[0].to_dict()
        return None

if __name__ == "__main__":
    sd = StudentData()
    print(sd.verify_student(12345678, 4821))
