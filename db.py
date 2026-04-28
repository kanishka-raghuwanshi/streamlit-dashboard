import sqlite3
import pandas as pd
from pathlib import Path
import streamlit as st
import numpy as np

DB_PATH = Path("smart_leads.db")

def get_connection():
    """Get database connection with dict-like rows."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Create all tables if they don't exist."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Table 1: Leads (with ML score column)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS leads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT,
            company TEXT,
            segment TEXT CHECK(segment IN ('SMB', 'Enterprise', 'Startup')),
            region TEXT CHECK(region IN ('APAC', 'EMEA', 'NA', 'LATAM')),
            source TEXT CHECK(source IN ('Ads', 'Organic', 'Referral', 'Events')),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            lead_score REAL DEFAULT 0,
            status TEXT DEFAULT 'New' CHECK(status IN ('New', 'Contacted', 'Qualified', 'Won', 'Lost'))
        )
    """)
    
    # Table 2: Customers (with ML churn column)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            company TEXT,
            segment TEXT CHECK(segment IN ('SMB', 'Enterprise', 'Startup')),
            region TEXT CHECK(region IN ('APAC', 'EMEA', 'NA', 'LATAM')),
            plan TEXT,
            mrr REAL,
            tenure_months INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            churn_risk REAL DEFAULT 0,
            status TEXT DEFAULT 'Active'
        )
    """)
    
    conn.commit()
    conn.close()
    print("✅ Database tables created!")

def add_lead(name, email, company, segment, region, source):
    """Add single lead from form."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO leads (name, email, company, segment, region, source)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (name, email, company, segment, region, source))
    conn.commit()
    lead_id = cursor.lastrowid
    conn.close()
    return lead_id

def get_leads():
    """Get all leads as DataFrame."""
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM leads ORDER BY lead_score DESC, created_at DESC", conn)
    conn.close()
    return df

def get_customers():
    """Get all customers as DataFrame."""
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM customers ORDER BY churn_risk DESC", conn)
    conn.close()
    return df

def get_filtered_leads(segment=None, region=None, source=None):
    """Server-side filtering for leads."""
    conn = get_connection()
    query = "SELECT * FROM leads WHERE 1=1"
    params = []
    
    if segment and segment != "All":
        query += " AND segment = ?"
        params.append(segment)
    if region and region != "All":
        query += " AND region = ?"
        params.append(region)
    if source and len(source) > 0:
        placeholders = ",".join("?" * len(source))
        query += f" AND source IN ({placeholders})"
        params.extend(source)
    
    query += " ORDER BY lead_score DESC"
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df

def update_lead_scores(scores_dict):
    """Update lead scores in database."""
    conn = get_connection()
    for lead_id, score in scores_dict.items():
        conn.execute("UPDATE leads SET lead_score = ? WHERE id = ?", (score, lead_id))
    conn.commit()
    conn.close()

def import_sample_data():
    """Add 50 sample leads."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM leads")
    
    sample_leads = [
        ("John Doe", "john@example.com", "TechCorp", "Enterprise", "NA", "Ads"),
        ("Jane Smith", "jane@test.com", "StartupX", "Startup", "APAC", "Organic"),
        ("Mike Johnson", "mike@biz.com", "GrowEasy", "SMB", "EMEA", "Referral"),
        ("Sarah Wilson", "sarah@enterprise.com", "BigCorp", "Enterprise", "LATAM", "Events"),
        ("Tom Brown", "tom@startup.io", "Innovate", "Startup", "NA", "Ads"),
    ] * 10
    
    cursor.executemany("""
        INSERT INTO leads (name, email, company, segment, region, source)
        VALUES (?, ?, ?, ?, ?, ?)
    """, sample_leads)
    conn.commit()
    conn.close()
    print("✅ Added 50 sample leads!")

if __name__ == "__main__":
    init_db()
    import_sample_data()
    print("🚀 Database + Sample Data Ready!")


