#!/usr/bin/env python3
"""
UrbanEye - Clear All Database Events and Incidents
Resets the SQLite database (data/events.db) to a completely clean state.
"""
import os
import sqlite3

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "events.db")

def clear_all_data():
    if not os.path.exists(DB_PATH):
        print(f"[INFO] Database file not found at {DB_PATH}.")
        return

    print("\n" + "="*55)
    print("🧹 CLEARING ALL URBANEYE DATABASE RECORDS...")
    print("="*55)
    print(f"Target Database: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    tables = ['events', 'incidents', 'incident_events', 'incident_notes', 'incident_history']
    total_cleared = 0

    for table in tables:
        try:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            cnt = cur.fetchone()[0]
            cur.execute(f"DELETE FROM {table}")
            print(f"  ✓ Cleared {cnt:4d} records from table '{table}'")
            total_cleared += cnt
        except sqlite3.OperationalError:
            pass

    conn.commit()
    conn.isolation_level = None
    cur.execute("VACUUM")
    conn.close()

    print("-"*55)
    print(f"✨ Successfully cleared {total_cleared} total records!")
    print("🌐 The web dashboard (http://127.0.0.1:3000) is now reset to 0.")
    print("💡 Note: Run 'python scripts/inject_fake_gps_data.py' if you ever want fresh sample data back.")
    print("="*55 + "\n")

if __name__ == "__main__":
    clear_all_data()
