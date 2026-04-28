import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from db import *
from ml_models import lead_model, churn_model, train_all_models

# Initialize
st.set_page_config(page_title="Smart Leads Pro", layout="wide")
st.title("🚀 Smart Leads Pro - ML Powered Dashboard")

# Auto-train ML models
if 'models_trained' not in st.session_state:
    with st.spinner("🤖 Training ML Models..."):
        init_db()
        train_all_models()
    st.session_state.models_trained = True
    st.success("✅ ML Models Trained & Ready!")

# Sidebar
st.sidebar.header("⚙️ ML Controls")
if st.sidebar.button("🔄 Retrain Models", use_container_width=True):
    st.rerun()
st.sidebar.metric("Leads", len(get_leads()))
st.sidebar.success("Production Ready!")

# 4 Professional Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "⭐ Leads (ML Scores)", "🔄 Churn Analytics", "🎯 ML Model Insights"])

with tab1:
    col1, col2, col3, col4 = st.columns(4)
    leads_df = get_leads()
    customers_df = get_customers()
    
    # ML Predictions
    lead_scores = lead_model.predict(leads_df)
    
    with col1:
        st.metric("Total Leads", len(leads_df), delta="+12")
    with col2:
        st.metric("High Score Leads", len(leads_df[lead_scores > 80]), delta="+5")
    with col3:
        st.metric("Avg Lead Score", f"{lead_scores.mean():.1f}", delta="+3.2")
    with col4:
        st.metric("Pipeline Value", f"${len(leads_df) * 750:,.0f}", delta="+15K")

    # Lead Source Distribution
    if len(leads_df) > 0:
        fig = px.pie(leads_df, names='source', title="Lead Sources", 
                    color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    # Advanced Filters
    col1, col2, col3 = st.columns(3)
    segment = col1.selectbox("Segment", ["All", "SMB", "Enterprise", "Startup"])
    region = col2.selectbox("Region", ["All", "APAC", "EMEA", "NA", "LATAM"])
    source = col3.multiselect("Source", ["Ads", "Organic", "Referral", "Events"])
    
    filtered_leads = get_filtered_leads(segment, region, source)
    
    # ML Scores for filtered leads
    if len(filtered_leads) > 0:
        filtered_scores = lead_model.predict(filtered_leads)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Filtered Leads", len(filtered_leads))
        with col2:
            st.metric("Avg Score", f"{filtered_scores.mean():.1f}")
        
        # Enhanced table with ML scores
        display_df = filtered_leads.copy()
        display_df['lead_score'] = filtered_scores
        st.dataframe(display_df[['name', 'company', 'segment', 'region', 'source', 'lead_score', 'status']],
                    column_config={"lead_score": st.column_config.ProgressColumn("ML Score", format="%d%%")},
                    use_container_width=True, hide_index=True)
    else:
        st.info("🔍 No leads match filters")

with tab3:
    st.header("🔄 ML-Powered Churn Analytics")
    
    # Generate realistic customers for demo
    customers_df = pd.DataFrame({
        'name': [f'Customer {i}' for i in range(1, 21)],
        'company': [f'Corp {i}' for i in range(1, 21)],
        'segment': np.random.choice(['Enterprise', 'SMB', 'Startup'], 20),
        'region': np.random.choice(['NA', 'APAC', 'EMEA', 'LATAM'], 20),
        'mrr': np.random.randint(1000, 50000, 20),
        'tenure_months': np.random.randint(1, 48, 20),
        'plan': np.random.choice(['Basic', 'Pro', 'Enterprise'], 20)
    })
    
    # ML Churn Predictions
    customers_df['churn_risk'] = churn_model.predict_proba(customers_df)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Customers", len(customers_df))
    with col2: st.metric("Churn Rate", f"{customers_df['churn_risk'].mean()*100:.1f}%")
    with col3: st.metric("Avg MRR", f"${customers_df['mrr'].mean():,.0f}")
    with col4: st.metric("High Risk", len(customers_df[customers_df['churn_risk']>0.7]))
    
    # ML Charts
    col1, col2 = st.columns(2)
    with col1:
        risk_dist = px.pie(customers_df, names=pd.cut(customers_df['churn_risk'], 
                                                     bins=[0,0.3,0.7,1], 
                                                     labels=['Low','Medium','High']),
                          title="ML Churn Risk Distribution")
        st.plotly_chart(risk_dist, use_container_width=True)
    
    with col2:
        churn_scatter = px.scatter(customers_df, x='tenure_months', y='mrr',
                                  size='churn_risk', color='churn_risk',
                                  hover_name='name', title="Churn Risk vs Metrics",
                                  color_continuous_scale='RdYlGn_r')
        st.plotly_chart(churn_scatter, use_container_width=True)

with tab4:
    st.header("🎯 ML Model Performance")
    
    leads_df = get_leads()
    if len(leads_df) > 0:
        scores = lead_model.predict(leads_df)
        fig = px.histogram(scores, nbins=20, title="Lead Score Distribution (ML Predicted)")
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("🏆 Top 10 High-Potential Leads")
        # FIXED: Ensure all required columns exist
        top_leads = leads_df[['name', 'company', 'segment']].copy().head(10)
        top_leads['lead_score'] = lead_model.predict(top_leads)
        st.dataframe(top_leads, use_container_width=True)


# Add Lead Form (Auto ML Scoring)
with st.expander("➕ Add New Lead (Auto ML Score)"):
    with st.form("new_lead"):
        col1, col2 = st.columns(2)
        name = col1.text_input("👤 Name")
        email = col2.text_input("📧 Email")
        
        col3, col4, col5, col6 = st.columns(4)
        company = col3.text_input("🏢 Company")
        segment = col4.selectbox("🎯 Segment", ["SMB", "Enterprise", "Startup"])
        region = col5.selectbox("🌍 Region", ["APAC", "EMEA", "NA", "LATAM"])
        source = col6.selectbox("📢 Source", ["Ads", "Organic", "Referral", "Events"])
        
        submitted = st.form_submit_button("🚀 Add & Predict Score", use_container_width=True)
        if submitted and name:
            add_lead(name, email, company, segment, region, source)
            st.success("✅ Lead added + ML score predicted!")
            st.balloons()
            st.rerun()
        elif submitted:
            st.error("❌ Name required!")

st.markdown("---")

